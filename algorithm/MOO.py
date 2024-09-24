import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
import copy
from tqdm import tqdm
# Set your OpenAI API key
import random
import torch
from functools import partial
import os
from algorithm.base import Item,History_Buffer
from openai import AzureOpenAI
from tdc.generation import MolGen

from eval import get_evaluation
import time
from model.util import nsga2_selection,so_selection
from algorithm import prompt_template

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_smiles_from_string(text):
    pattern = r"<mol>(.*?)</mol>"
    smiles_list = re.findall(pattern, text)
    return smiles_list

def split_list(lst, n):
    """Splits the list lst into n nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

class MOO:
    def __init__(self, reward_system, llm,property_list,config):
        self.reward_system = reward_system
        self.config = config
        self.llm = llm
        self.history = History_Buffer()
        self.property_list = property_list
        self.moles_df = None
        self.pop_size = self.config.get('optimization.pop_size')
        self.init_mol_dataset()
        self.prompt_module = getattr(prompt_template ,self.config.get('model.prompt_module',default='Prompt'))

    def init_mol_dataset(self):
        print('Loading ZINC dataset...')
        data = MolGen(name='ZINC')
        split = data.get_split()
        self.moles_df = split['train']

    def generate_initial_population(self, mol1, n):
        num_blocks = 200
        combs = [[mol1, mol2] for mol2 in self.moles_df.smiles]
        combs_blocks = split_list(combs, num_blocks)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_evaluation, ['similarity'], block) for block in combs_blocks]
            results = [future.result() for future in futures]

        combined_results = []
        for result in results:
            combined_results.extend(result['similarity'])

        self.moles_df['similarity'] = combined_results
        top_n = self.moles_df.nlargest(n - 1, 'similarity').smiles.values.tolist()
        top_n.append(mol1)
        return [Item(i,self.property_list) for i in top_n]

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_crossover_prompt(parent_list)
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response

    def evaluate(self, smiles_list):
        ops = self.property_list
        res = self.reward_system.evaluate(ops,smiles_list)
        results = np.zeros([len(smiles_list),len(ops)])
        raw_results = np.zeros([len(smiles_list),len(ops)])
        for i,op in enumerate(ops):
            part_res = res[op]
            for j,inst in enumerate(part_res):
                inst = inst[0]
                raw_results[j,i] = inst
                if op=='qed':
                    if 'increase' in self.requirement_meta['qed_requ']['requirement']:
                        results[j,i] = -inst
                    else:
                        results[j,i] = inst
                elif op=='logp':
                    logp_requ = self.requirement_meta['logp_requ']['requirement']
                    if 'increase' in logp_requ:
                        results[j,i] = -inst / 10
                    elif 'range' in logp_requ:
                        a, b = [int(x) for x in logp_requ.split(',')[1:]]
                        mid = (b+a)/2
                        results[j,i] = np.clip(abs(inst-mid) * 1/ ( (b-a)/2), a_min=0,a_max=1)    # 2-3    2.5  3-2.5=0.5 * 2     1 0
                        #if a<=inst<=b:
                        #    results[j,i] = 0
                        #else:
                        #    results[j,i] = 1     
                    else:
                        results[j,i] = inst/10
                elif op=='donor':
                    #print('donor num',inst,donor_num,j,i)
                    donor_requ = self.requirement_meta['donor_requ']['requirement']
                    donor_num = self.requirement_meta['donor_num']
                    if donor_requ == 'increase' and inst - donor_num>0:
                        results[j,i] = 0
                    elif donor_requ == 'decrease' and donor_num - inst>0:
                        results[j,i] = 0
                    elif donor_requ == 'same' and donor_num == inst:
                        results[j,i] = 0
                    elif donor_requ == 'increase, >=2' and inst - donor_num>=2:
                        results[j,i] = 0
                    else:
                        results[j,i] = 1  
                else:
                    raise NotImplementedError
        return results,raw_results

    def evaluate_all(self,items):
        smiles_list = [i.value for i in items]
        smiles_list = [[i,smiles_list[0]] for i in smiles_list]
        fitnesses,raw_results = self.evaluate(smiles_list)
        for i,ind in enumerate(items):
            ind.scores = fitnesses[i]
            ind.assign_raw_scores(raw_results[i])
            #ind.raw_scores = raw_results[i]

    def run(self, prompt,requirements):
        """High level logic"""
        self.requirement_meta = requirements
        ngen= self.config.get('optimization.ngen')
        
        #initialization 
        mol = extract_smiles_from_string(prompt)[0]
        self.original_mol = mol
        self.prompt_generator = self.prompt_module(self.original_mol,self.requirement_meta,self.property_list)


        population = self.generate_initial_population(mol1=mol, n=self.pop_size)
        donor_num = get_evaluation(['donor'], [[mol, mol]])['donor'][0][0]
        self.requirement_meta['donor_num'] = donor_num
        self.evaluate_all(population)

        init_pops = copy.deepcopy(population)

        #offspring_times = self.config.get('optimization.eval_budge') // ngen //2
        offspring_times = self.pop_size //2
        for gen in tqdm(range(ngen)):
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(population, offspring, self.pop_size)
        return init_pops,population

    def generate_offspring(self, population, offspring_times):
        #for _ in range(offspring_times): # 20 10 crossver+mutation 20 
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        while True:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.crossover, parent_list=parent_list) for parent_list in parents]
                    results = [future.result() for future in futures]
                    children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
                    break
            except Exception as e:
                print('retry in 60s, exception ',e)
                time.sleep(90)
        for child_pair in children:
            self.evaluate_all(child_pair)
        # check if the child is valid
        offspring = self.check_valid(children)
        self.history.push(prompts,children,responses) 
        return offspring

    def check_valid(self,children):
        tmp_offspring = []
        offspring = []
        for child_pair in children:
            tmp_offspring.extend(child_pair)
        for child in tmp_offspring:
            if self.is_valid(child):
                offspring.append(child)
        return offspring
    
    def is_valid(self,child):
        for idx, op in enumerate(child.property_list):
            if op == 'qed' and child.raw_scores[idx]==0:
                return False
            if op == 'logp' and child.raw_scores[idx]==-100:
                return False
        return True

    def select_next_population(self, population, offspring, pop_size):
        combined_population = offspring + population 
        #return self.no_selection(combined_population,pop_size)

        if len(self.property_list)>1:
            return nsga2_selection(combined_population, pop_size)
        else:
            return so_selection(combined_population, pop_size)
    
    def no_selection(self,combined_population, pop_size):
        return combined_population[:pop_size] 
c,p,r = None,None,None