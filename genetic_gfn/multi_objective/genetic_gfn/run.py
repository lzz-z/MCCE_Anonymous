import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from genetic_gfn.multi_objective.optimizer import BaseOptimizer
from genetic_gfn.multi_objective.genetic_gfn.utils import Variable, seq_to_smiles, unique
from genetic_gfn.multi_objective.genetic_gfn.model import RNN
from genetic_gfn.multi_objective.genetic_gfn.data_structs import Vocabulary, Experience
from algorithm.base import Item
import torch
from rdkit import Chem
from copy import deepcopy
from joblib import Parallel
from genetic_gfn.multi_objective.genetic_gfn.graph_ga_expert import GeneticOperatorHandler


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


class Genetic_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"
        self.history_smiles = []
    
    def setup_model(self, oracle, config):
        self.oracle.set_objectives(*(oracle))
        self.config = config

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from = restore_prior_from
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        self.voc = voc
        self.Prior = RNN(voc)
        self.Agent = RNN(voc)

        if torch.cuda.is_available():
            self.Prior.rnn.load_state_dict(torch.load(restore_prior_from))
            self.Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            self.Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location='cpu'))
            self.Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location='cpu'))

        for param in self.Prior.rnn.parameters():
            param.requires_grad = False

        self.log_z = torch.nn.Parameter(torch.tensor([5.]).cuda() if torch.cuda.is_available() else torch.tensor([5.]))
        self.optimizer = torch.optim.Adam([
            {'params': self.Agent.rnn.parameters(), 'lr': config['learning_rate']},
            {'params': self.log_z, 'lr': config['lr_z']}
        ])

        self.experience = Experience(voc, max_size=config['num_keep'])
        self.model_initialized = True
        self.ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        self.pool = Parallel(n_jobs=config['num_jobs'])
        print("Model setup complete.")


    def train_on_smiles(self, smiles: list, scores: list, loop: int,time_step=0,mol_buffer = None):
        """
        用外部提供的SMILES和分数训练一次
        """
        if not self.model_initialized:
            raise RuntimeError("Please call setup_model() before training.")

        smiles = sanitize(smiles)
        
        
        self.experience.add_experience(zip(smiles, scores))
        au_smiles = [i[0] for i in self.experience.memory]
        repeat_au = len(au_smiles) - len(np.unique(au_smiles))
        print(f'step {time_step}: repeat {repeat_au} {len(smiles)} smiles added to experience after sanitizing! experience length {len(self.experience)}')
    
        config = self.config
        Agent = self.Agent
        Prior = self.Prior
        experience = self.experience
        log_z = self.log_z
        optimizer = self.optimizer
        
        if len(experience) > config['experience_replay']:
            for _ in range(loop): # config['experience_loop']
                if config['rank_coefficient'] > 0:
                    exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                else:
                    exp_seqs, exp_score = experience.sample(config['experience_replay'])
                exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                reward = torch.tensor(exp_score).cuda() 

                exp_forward_flow = exp_agent_likelihood + log_z
                exp_backward_flow = reward * config['beta']
                loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                if config['penalty'] == 'prior_kl':
                    loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                    loss += config['kl_coefficient'] * loss_p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            print("Not enough experience to train. Current:", len(experience))
        
    def mol_buffer_store(self,mol_buffer,smiles):
        all_smiles = [i[0].value for i in mol_buffer]
        scores = np.array(self.oracle(smiles))
        for i,child in enumerate(smiles):
            mol = Chem.MolFromSmiles(child) 
            if mol is None: # check if valid
                pass
            else:
                child = Chem.MolToSmiles(mol,canonical=True)
                # check if repeated
                if child in all_smiles:
                    pass
                else:
                    all_smiles.append(child)
                    item = Item(child,['sa','drd2','qed','gsk3b','jnk3'])
                    item.total = scores[i]
                    mol_buffer.append([item,len(mol_buffer)+1])
        return mol_buffer

    def sample_n_smiles(self, n: int,mol_buffer: list) -> list:
        """
        用当前Agent策略生成n个SMILES字符串
        """
        mol_buffer = deepcopy(mol_buffer)
        if not self.model_initialized:
            raise RuntimeError("Please call setup_model() before sampling.")
        all_smiles = []
        config = self.config

        seqs, _, _ = self.Agent.sample(self.config['batch_size'])
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        smiles = seq_to_smiles(seqs, self.voc)
        smiles = sanitize(smiles)

        all_smiles.extend(smiles)
        self.mol_buffer_store(mol_buffer,smiles)
        
        
        if config['population_size'] and len(mol_buffer) > config['population_size']:
            mol_buffer = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)
            #pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in mol_buffer.items()])))
            pop_smis = [i[0].value for i in mol_buffer]
            pop_scores = [i[0].total for i in mol_buffer]
            
            populations = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])
            # populations = select_pop(pop_smis, pop_scores, config['population_size'], rank_coefficient=config['rank_coefficient'])

            for g in range(config['ga_generations']):
                child_smis, child_n_atoms, pop_smis, pop_scores = self.ga_handler.query(
                        query_size=config['offspring_size'], mating_pool=populations, pool=self.pool, 
                        rank_coefficient=config['rank_coefficient'], 
                    )

                child_score = np.array(self.oracle(child_smis))
                all_smiles.extend(child_smis)
                new_experience = zip(child_smis, child_score)
                self.experience.add_experience(new_experience)

                # import pdb; pdb.set_trace()
                populations = (pop_smis+child_smis, pop_scores+child_score.tolist())
        return all_smiles

    def _optimize(self, oracle, config):

        # self.oracle.assign_evaluator(oracle)
        self.oracle.set_objectives(*(oracle))

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        log_z = torch.nn.Parameter(torch.tensor([5.]).cuda())
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': config['learning_rate']},
                                    {'params': log_z, 
                                        'lr': config['lr_z']}])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc, max_size=config['num_keep'])

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            smiles = seq_to_smiles(seqs, voc)
            if config['valid_only']:
                smiles = sanitize(smiles)
            
            score = np.array(self.oracle(smiles))
            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # early stopping
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            # Calculate augmented likelihood
            # augmented_likelihood = prior_likelihood.float() + 500 * Variable(score).float()
            # reinvent_loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # print('REINVENT:', reinvent_loss.mean().item())

            # Then add new experience
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            if config['population_size'] and len(self.oracle) > config['population_size']:
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))

                populations = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])
                # populations = select_pop(pop_smis, pop_scores, config['population_size'], rank_coefficient=config['rank_coefficient'])

                for g in range(config['ga_generations']):
                    child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=populations, pool=pool, 
                            rank_coefficient=config['rank_coefficient'], 
                        )

                    child_score = np.array(self.oracle(child_smis))
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    # import pdb; pdb.set_trace()
                    populations = (pop_smis+child_smis, pop_scores+child_score.tolist())
 
                    if self.finish:
                        print('max oracle hit')
                        break
                
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] > 0:
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])

                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty
                    if config['penalty'] == 'prior_kl':
                        loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                        loss += config['kl_coefficient']*loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            step += 1

