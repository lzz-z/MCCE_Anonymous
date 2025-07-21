import numpy as np
import pickle
import os
from typing import List, Dict, Any

class Item:
    def __init__(self, value: str, property_list: List[str]) -> None:
        """
        Initializes an Item object representing a molecule with specified properties.

        Args:
            value (str): The molecule's SMILES string or identifier.
            property_list (List[str]): List of property names to be assigned to the molecule.
        """
        self.value = value
        self.property_list = property_list
        self.assign_raw_scores([0.0 for _ in self.property_list])
        self.scores = [0.0 for _ in self.property_list]
        self.total = 0.0

    def assign_raw_scores(self, scores: List[float]) -> None:
        """
        Assigns raw (original) scores to each property and computes the total score.

        Args:
            scores (List[float]): Values corresponding to each property in property_list.
        """
        self.raw_scores = scores
        self.property = {
            self.property_list[i]: scores[i] for i in range(len(self.property_list))
        }
        self.cal_sum()

    def cal_sum(self) -> None:
        """
        Calculates the overall total score based on property-specific scoring rules.
        """
        self.total = 0.0
        for p in self.property_list:
            val = self.property[p]
            if p in ['qed', 'jnk3', 'bbbp1']:
                self.total += val
            elif p == 'sa':
                self.total += 1 - (val - 1) / 9
            elif p in ['gsk3b', 'drd2', 'smarts_filter']:
                self.total += 1 - val
            elif p == 'logs':
                self.total += (val + 8) / 9
            elif p == 'reduction_potential':
                self.total += 1 - abs(np.clip(val, -2.3, -0.3) + 1.3)
            else:
                raise NotImplementedError(f"Property '{p}' is not defined in Item scoring rules.")


class HistoryBuffer:
    def __init__(self) -> None:
        """
        Initializes a history buffer for storing prompts, generations, and responses.
        """
        self.prompts = []
        self.generations= []
        self.responses = []
        self.successful_molecules = []
        self.failed_molecules = []
        self.save_path = 'checkpoint/'

    def save_to_pkl(self, filename: str) -> None:
        """
        Saves the entire buffer object to a pickle file.

        Args:
            filename (str): The name of the output pickle file.
        """
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, filename), 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {filename}")

    def load_from_pkl(self, filename: str) -> 'HistoryBuffer':
        """
        Loads a HistoryBuffer object from a pickle file.

        Args:
            filename (str): The pickle file name to load from.

        Returns:
            HistoryBuffer: Loaded history buffer instance.
        """
        with open(os.path.join(self.save_path, filename), 'rb') as f:
            obj = pickle.load(f)
        print(f"Data loaded from {filename}")
        return obj

    def push(self, prompts: Any, generation: Any, responses: Any) -> None:
        """
        Appends a new generation record to the buffer.

        Args:
            prompts (Any): The prompt(s) used to generate molecules.
            generation (Any): Generated molecule(s).
            responses (Any): Model's response(s), e.g., SMILES strings.
        """
        self.prompts.append(prompts)
        self.generations.append(generation)
        self.responses.append(responses)


'''
    'celecoxib_rediscovery', 'troglitazone_rediscovery','thiothixene_rediscovery',
'albuterol_similarity','mestranol_similarity',
'isomers_c7h8n2o2','isomers_c9h10n2o2pf2cl','median1','median2', 'osimertinib_mpo',
'fexofenadine_mpo','ranolazine_mpo','perindopril_mpo', 'amlodipine_mpo',
'sitagliptin_mpo','zaleplon_mpo','valsartan_smarts', 'deco_hop', 'scaffold_hop'
'''