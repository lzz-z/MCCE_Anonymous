class Prompt:
    def __init__(self,original_mol,requirements,properties):
        self.requirements = requirements
        self.property = properties
        self.original_mol = original_mol

    def get_first_prompt(self):
        pass

    def get_final_prompt(self):
        pass 
    
    def get_crossover_prompt(self,ind_list):
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        history_prompt = self.make_history_prompt(ind_list)
        instruction_prompt = self.make_instruction_prompt()
        final_prompt = requirement_prompt + history_prompt + instruction_prompt
        #final_prompt = requirement_prompt  + instruction_prompt
        return final_prompt

    def make_history_prompt(self, ind_list):
        pop_content = "I have some molecules with their objective values. "
        for ind in ind_list:
            pop_content += f"<mol>{ind.value}</mol>,"
            for index,property in enumerate(ind.property_list):
                pop_content += f"{property}:{ind.raw_scores[index]},  "
            pop_content += '\n'
        return pop_content
    
    def make_instruction_prompt(self):
        prompt = (" Give me two new molecules that are different from all points above, and not dominated by any of the above. "
        "You can do it by applying crossover on the points I give to you. "
        f"Please note when you try to achieving these objectives, the molecules you propose should be similar to the original molecule <mol>{self.original_mol}</mol>. "
        "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLE form"
        )
        return prompt
    

    def make_requirement_prompt(self,original_mol,requirements,properties):
        sentences = []
        for property in properties:
            value = requirements[property+'_requ']
            property_name = value["property"]
            source_smiles = value["source_smiles"]
            requirement = value["requirement"]

            # Check for specific requirement patterns directly using symbols
            if "increase" in requirement:
                if ">=" in requirement:
                    threshold = requirement.split(">=")[-1].strip()
                    sentence = f"help me increase the {property_name} value by at least {threshold}."
                elif ">" in requirement:
                    threshold = requirement.split(">")[-1].strip()
                    sentence = f"help me increase the {property_name} value to more than {threshold}."
                else:
                    sentence = f"help me increase the {property_name} value."

            elif "decrease" in requirement:
                if "<=" in requirement:
                    threshold = requirement.split("<=")[-1].strip()
                    sentence = f"help me decrease the {property_name} value to at most {threshold}."
                elif "<" in requirement:
                    threshold = requirement.split("<")[-1].strip()
                    sentence = f"help me decrease the {property_name} value to less than {threshold}."
                else:
                    sentence = f"help me decrease the {property_name} value."

            elif "range" in requirement:
                # Extract the range values from the string
                range_values = requirement.split(",")[1:]
                range_start = range_values[0].strip()
                range_end = range_values[1].strip()
                sentence = f"help me keep the {property_name} value within the range {range_start} to {range_end}."

            elif "the same" in requirement:
                sentence = f"help me keep the {property_name} value the same."

            elif any(op in requirement for op in [">=", "<=", "=", ">", "<"]):
                # Directly use the symbols for constraints
                sentence = f"help me ensure the {property_name} value is {requirement}."
                
            else:
                sentence = f"help me modify the {property_name} value."
            sentences.append(sentence)
        sentences = f'Suggest new molecules based on molecule <mol>{original_mol}</mol> and ' + 'and '.join(sentences) 
        return sentences

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

# Generate sentences based on metadata
#generate_sentence('CCH',metadata,['qed','donor'])
