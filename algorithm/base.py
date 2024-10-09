class Item:
    #property_list = ['qed', 'logp', 'donor']

    def __init__(self, value, property_list):
        self.value = value
        self.property_list = property_list if property_list is not None else self.property_list
        # raw scores are the original objective values
        self.assign_raw_scores([ 0 for prop in self.property_list])
        # scores are the objective values (after judgement) for MOO
        self.scores = [ 0 for prop in self.property_list]
    
    def assign_raw_scores(self,scores):
        self.raw_scores = scores
        self.property = {self.property_list[i]:scores[i] for i in range(len(self.property_list))}

import pickle
import os
class HistoryBuffer:
    def __init__(self):
        self.prompts = []
        self.generations = []
        self.responses = [] # <mol> </mol>
        self.save_path = 'checkpoint/'

    def save_to_pkl(self, filename):
        with open(os.path.join(self.save_path,filename), 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {filename}")

    def load_from_pkl(self,filename):
        with open(os.path.join(self.save_path,filename), 'rb') as f:
            obj = pickle.load(f)
        print(f"Data loaded from {filename}")
        return obj

    def push(self,prompts,generation,responses):
        self.prompts.append(prompts)
        self.generations.append(generation)
        self.responses.append(responses)