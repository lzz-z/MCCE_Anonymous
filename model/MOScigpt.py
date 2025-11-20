import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from deap import base, creator, tools, algorithms
import re
import copy
from tqdm import tqdm
import random
import torch
from functools import partial
import os
from openai import AzureOpenAI
from tdc import Oracle
import time
from MOLLM import *
from load_Scigpt import load_scigpt

model,tokenizer = load_scigpt()
def query_sci(prompt,model,tokenizer,num_decode=20):
    input_mol = prompt.split('</mol>')[0].split('<mol>')[-1]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids, do_sample=True, temperature=0.75, top_p=0.95, max_new_tokens=300, num_return_sequences=num_decode)
    tmp_list = []
    for s in output:
        s = tokenizer.decode(s)
        response  = s.split(' : ')[-1]
        output_mol = response.split('</mol>')[0].split('<mol>')[-1].replace('<m>', '').replace(' ', '')
        tmp_list.append(output_mol)
    return tmp_list,s

def crossover_sci(prompt,ind_list):
    pop_content = prepare_pop_content(ind_list)
    original_mol = extract_smiles_from_string(prompt)[0]
    prompt = (
        prompt + 
        "I have some molecules with their objective values. "
        + pop_content +
        " Give me two new molecules that are different from all points above, and not dominated by any of the above. "
        "You can do it by applying crossover on the points I give to you. "
        f"Please note when you try to achieving these objectives, the molecules given you propose should be similar to the original molecule <mol>{original_mol}</mol>: <mol>"
    )
    tmp_list,s = query_sci(prompt,model,tokenizer,num_decode=2)
    new_ind_list = [Individual(i) for i in tmp_list]
    if len(new_ind_list)==0:
        print('cannot decode any molecules from the respones')
        print('prompt:',prompt)
        print('response:',tmp_list)
        new_ind_list = ind_list
    return new_ind_list

pop_size=30

def main(population, prompt,qed_requ,logp_requ,donor_requ,pop_size=20,eval_budget=200,ngen=10,seed=42,scigpt=False):
    set_seed(42)
    mol = extract_smiles_from_string(prompt)[0]
    donor_num = get_evaluation(['donor'],[[mol,mol]])['donor'][0][0]
    evaluate_all(population,qed_requ=qed_requ,logp_requ=logp_requ,donor_requ=donor_requ,donor_num=donor_num)
    init_population = [copy.deepcopy(i) for i in population]
    cxpb = 0.7
    mutpb = 0.2

    opers_per_gen = eval_budget // ngen //2
    for gen in tqdm(range(ngen)):
        offspring = []
        
        parents = [list(sample_candidates(population,2)) for _ in range(opers_per_gen)]
        if scigpt:
            new_inds = [crossover_sci(prompt,parent_list) for parent_list in parents]
        else:
            while True:
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(crossover, prompt=prompt,ind_list=parent_list) for parent_list in parents]
                        new_inds = [future.result() for future in futures]
                        break
                except Exception as e:
                    print('retry in 60s, exception ',e)
                    time.sleep(90)
        new_inds = [smile for result in new_inds for smile in result]
        while True:
            try:
                evaluate_all(new_inds,qed_requ=qed_requ,logp_requ=logp_requ,donor_requ=donor_requ,donor_num=donor_num)
                break
            except Exception as e:
                print('retry in 30s, exception ',e)
                time.sleep(30)

        for ind in new_inds:
            if ind.fitness[0] != 0.0 and abs(ind.fitness[1]) != 100:
                offspring.append(ind)
        if len(offspring) > len(population):
            offspring = offspring[:len(population)]
        population[:] = nsga2_selection(population + offspring, pop_size)

    return init_population, population

