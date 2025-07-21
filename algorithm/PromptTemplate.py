from model.util import nsga2_selection
import numpy as np

from algorithm.base import Item
import pygmo as pg
from typing import List, Dict, Optional, Union
import json
with open('data/descriptions.json','r') as f:
    descriptions = json.load(f) 

class Prompt:
    def __init__(self, original_mol: Item, requirements: Dict,
                 properties: List[str], experience_prob: float = 0.5):
        """
        Args:
            original_mol (Item): The original Item used as a reference.
            requirements (dict): Dictionary specifying the property optimization requirements.
            properties (List[str]): List of properties to optimize.
            experience_prob (float): Probability of using previous experience in the prompt.
        """
        self.original_mol = original_mol
        self.requirements = requirements
        self.properties = properties
        self.experience: Optional[str] = None
        self.pure_experience: Optional[str] = None
        self.exp_times = 0
        self.experience_prob = experience_prob

    def get_prompt(self, prompt_type: str, ind_list: List[Item], history_moles: List[Item]) -> str:
        """
        Args:
            prompt_type (str): Type of operation ('crossover', 'mutation', or 'explore').
            ind_list (List[Item]): List of current individuals (e.g.molecules).
            history_moles (List[Item]): List of previously generated items (e.g.molecules).

        Returns:
            str: A formatted prompt string.
        """
        experience = self.experience if self.experience and np.random.random() < self.experience_prob else ""
        if prompt_type == 'crossover':
            return self._get_crossover_prompt(ind_list, history_moles, experience)
        elif prompt_type == 'mutation':
            return self._get_mutation_prompt(ind_list, history_moles, experience)
        elif prompt_type == 'explore':
            return self._get_exploration_prompt(history_moles)
        else:
            raise NotImplementedError(f'Unsupported operation type: {prompt_type}')

    def _get_crossover_prompt(self, ind_list: List[Item], history_moles: List[Item], experience: str) -> str:
        return self._compose_prompt(ind_list, experience, 'crossover')

    def _get_mutation_prompt(self, ind_list: List[Item], history_moles: List[Item], experience: str) -> str:
        return self._compose_prompt(ind_list[:1], experience, 'mutation')

    def _get_exploration_prompt(self, history_moles: List[Item]) -> str:
        top100 = sorted(history_moles, key=lambda x: x.total, reverse=True)[:100]
        worst10 = sorted(history_moles, key=lambda x: x.total)[self.exp_times * 10:(self.exp_times + 1) * 10]
        random10 = np.random.choice(top100, size=10, replace=False)

        prompt = self._compose_prompt(random10, "", 'explore')
        prompt += "There are also some bad candidates, don't propose new candidates like the candidates below:\n"
        prompt += self._make_history_prompt(worst10, experience=False)
        return prompt

    def _compose_prompt(self, ind_list: List[Item], experience: str, oper_type: str) -> str:
        parts = [
            self._make_requirement_prompt(),
            self._make_description_prompt(),
            experience,
            self._make_history_prompt(ind_list),
            self._make_instruction_prompt(oper_type)
        ]
        return ''.join(parts)

    def _make_description_prompt(self) -> str:
        return ''.join([f"{p}: {descriptions[p]}\n" for p in self.properties])

    def _make_history_prompt(self, ind_list: List[Item], experience: bool = False) -> str:
        header = "" if experience else (
            "I have some candidates with their objective values. The total score is the integration of all property values; a higher total score means a better candidate.\n")
        entries = [
            f"<mol>{ind.value}</mol>, its property values are: " +
            ", ".join([f"{prop}:{score:.4f}" for prop, score in zip(ind.property_list, ind.raw_scores)]) +
            f", total: {ind.total:.4f}\n"
            for ind in ind_list
        ]
        return header + ''.join(entries)

    def _make_instruction_prompt(self, oper_type: str) -> str:
        common_tail = ("Do not write code. Do not give any explanation. "
                       "Each output new molecule must start with <mol> and end with </mol> in SMILES form.\n")

        if oper_type == 'mutation':
            return (
                "Generate 2 new better molecules in SMILES format through mutation, ensuring they are different from all points provided above and not dominated by any of them.\n"
                "The molecules must be valid. Example operations include:\n"
                "1. Modify functional groups\n"
                "2. Replace atoms or bonds\n"
                "3. Add/remove small substituents\n"
                "4. Ring modifications\n"
                "5. Stereochemistry changes\n"
                "6. Property-specific optimizations\n" + common_tail)
        elif oper_type == 'crossover':
            return (
                "Give me 2 new better molecules that are different from all points above and not dominated by them.\n"
                "Use crossover and your knowledge to create valid molecules.\n" + common_tail)
        elif oper_type == 'explore':
            return (
                "Confidently propose two novel and better molecules different from the given ones, leveraging your expertise.\n"
                "The molecule should be valid.\n" + common_tail)
        else:
            raise NotImplementedError(f'Unsupported instruction type: {oper_type}')

    def _make_requirement_prompt(self) -> str:
        sentences = [
            f"{i+1}. {self._translate_requirement(prop)}"
            for i, prop in enumerate(self.properties)
        ]
        header = "suggest new molecules that satisfy the following requirements:\n"
        if 'reduction_potential' in self.properties:
            header += 'reduction_potential is the most important objective, make sure it is as close to -1.3 as possible.\n'
        return header + '\n'.join(sentences) + '\n'

    def _translate_requirement(self, prop: str) -> str:
        if prop == 'similarity':
            return f"make sure the new molecules have a similarity over 0.4 to the original molecule <mol>{self.original_mol.value}</mol>"

        req = self.requirements[f'{prop}_requ']['requirement']
        pname = self.requirements[f'{prop}_requ']['property']

        if 'increase' in req:
            return f"increase the {pname} value{' by at least ' + req.split('>=')[-1] if '>=' in req else '.'}"
        elif 'decrease' in req:
            return f"decrease the {pname} value{' to at most ' + req.split('<=')[-1] if '<=' in req else '.'}"
        elif 'range' in req:
            low, high = req.split(',')[1:3]
            return f"keep the {pname} value within the range {low.strip()} to {high.strip()}"
        elif 'equal' in req or 'towards' in req:
            value = req.split(',')[1]
            return f"make the {pname} value close to {value.strip()}"
        else:
            return f"optimize the {pname} as appropriate"

    def make_experience_prompt(self, all_mols: List[tuple]) -> tuple[str, str, str]:
        """
        Create a learning-based experience prompt from previous molecules.

        Args:
            all_mols (List[tuple]): A list of tuples, where each tuple's first element is an Item.

        Returns:
            tuple[str, str, str]: A summary prompt, a top-5 example block, and a worst-5 example block.
        """
        all_items = [i[0] for i in all_mols]
        experience_type = np.random.choice(['best_f', 'hvc', 'pareto'], p=[0.5, 0., 0.5])

        # Worst molecules (for negative examples)
        worst10 = sorted(all_items, key=lambda x: x.total)[self.exp_times * 10:(self.exp_times + 1) * 10]

        # Best molecules (based on different strategies)
        if experience_type == 'best_f':
            top100 = sorted(all_items, key=lambda x: x.total, reverse=True)[:100]
            best10 = list(np.random.choice(top100, size=10, replace=False))

        elif experience_type == 'pareto':
            best100, _ = nsga2_selection(all_items, pop_size=100, return_fronts=True)
            best10 = list(np.random.choice(best100, size=10, replace=False))

        elif experience_type == 'hvc':
            best100, fronts = nsga2_selection(all_items, pop_size=100, return_fronts=True)
            if len(fronts[0]) <= 10:
                best10 = [all_items[i] for i in fronts[0]]
            else:
                scores = np.array([all_items[i].scores for i in fronts[0]])
                points = [all_items[i] for i in fronts[0]]
                hv = pg.hypervolume(scores)
                contrib = hv.contributions(ref_point=np.array([2.0] * scores.shape[1]))
                sorted_idx = np.argsort(contrib)[::-1]
                best10 = [points[i] for i in sorted_idx[:10]]

        else:
            raise NotImplementedError(f"Unsupported experience type: {experience_type}")

        # Compose prompts
        best_prompt = self._make_history_prompt(best10, experience=True)
        worst_prompt = self._make_history_prompt(worst10, experience=True)
        requirement_text = self._make_requirement_prompt()

        summary_prompt = (
            f"I am optimizing molecular properties based on the following requirements:\n{requirement_text}\n"
            f"Here are some excellent non-dominated molecules and their associated property values:\n{best_prompt}\n"
            f"Here are some poorly performing molecules and their associated property values:\n{worst_prompt}\n"
            "Please analyze the patterns and characteristics of the excellent molecules. Summarize what makes them excel and suggest strategies "
            "to create new molecules with similar and better properties. Additionally, identify the reasons why the poorly performing molecules "
            "have suboptimal properties and provide guidance on how to avoid such issues in the future.\n"
        )

        if self.experience:
            summary_prompt += (
                "When you write the experience, also integrate the important information from my old experience:\n"
                f"<old experience>{self.pure_experience}</old experience>\n"
                "In your reply, only write the concise integrated experience without repetitive sentences or conclusions.\n"
                "You can also abandon less useful or inappropriate experience in my old experience.\n"
            )

        summary_prompt += (
            "Keep the summary concise (within 200 words), focusing on actionable insights and avoiding redundancy. "
            "Don't describe the given molecules, directly state the experience."
        )

        self.exp_times += 1
        return summary_prompt, self._make_history_prompt(best10[:5], experience=True), self._make_history_prompt(worst10[:5], experience=True)


metadata = {
    "qed_requ": {
        "source_smiles": "O=C([C@H]1CCCC[C@H]1N1CCCC1=O)N1CC2(CC(F)C2)C1",
        "reference_smiles": "NNC(=O)C(=O)NC1CC2(C1)CN(C(=O)[C@H]1CCCC[C@H]1N1CCCC1=O)C2",
        "property": "QED",
        "requirement": "decrease <=2"
    },
    "logp_requ": {
        "source_smiles": "CCCCC(CC)COCOc1ccc([C@@H](O)C(=O)N[C@@H]2[C@H]3COC[C@@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "reference_smiles": "COc1ccc([C@@H](O)C(=O)N[C@H]2[C@@H]3COC[C@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "property": "logP",
        "requirement": "range, 2, 3"
    },
    "donor_requ": {
        "source_smiles": "O=C(NC[C@H]1CCOc2ccccc21)c1ccc(F)c(C(F)(F)F)c1",
        "reference_smiles": "CC(C)C[NH+](CC(=O)[O-])C(F)(F)c1cc(C(=O)NC[C@H]2CCOc3ccccc32)ccc1F",
        "property": "donor",
        "requirement": "increase >= 1"
    },
}
import json
if __name__ == '__main__':
    with open('/home/hp/src/MOLLM/data/goals5.json','r') as f:
        metadata = json.load(f)
    metadata = metadata['requirements'][0]
    ops = ['qed','sa','jnk3']
    parents = [Item('CCFF',ops),Item('FFFFA',ops)]
    parents[0].raw_scores = [0,1,2,3]
    parents[1].raw_scores = [4,5,6,7]
    parents[0].assign_raw_scores([0,1,2,3])
    p = Prompt(parents[0],metadata,ops)
    prompt = p.get_prompt( prompt_type='mutation', ind_list=parents, history_moles=parents)
    print(prompt)
