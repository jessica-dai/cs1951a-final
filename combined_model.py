import numpy as np 

from data_arts_ed.model import get_models as get_arts
from data_crdc.model import get_models as get_crdc
from data_cte_progs.model import get_models as get_cte
from data_edtech_access.model import get_models as get_edtech

#todo: key / decoder

class combinedModel():

    """
    Assumed order of process is always CRDC ARTS CTE EDTECH
    """

    def __init__(self):
        self.raw_scores = None
        self.weights = None
        self.models = None
        self.keys = self.init_keys()
    
    def init_keys():

        keys = {}

        comms = ["urban", "suburban", "town", "rural"]
        regions = ["northeast", "southeast", "central", "west"]

        for i in range(4):
            comm = comms[i]

            for j in range(4):
                region = regions[i]

                keys[4*i + j] = comm + ", " + region


        return keys

    def init_models():

        for data_models in (get_crdc(), get_arts(), get_cte(), get_edtech()):
            model, score = get_best_model(data_models)
            self.raw_scores.append(score)
            self.models.append(model)
    
    def get_best_model(returned_models): # other ways to do this would be to do so for a particular type of model (e.g. just linear regression)
        max_score = 0
        best_model = None
        for model, score in returned_models():
            if score > max_score:
                best_model = model
                max_score = score
        
        return best_model, max_score

    def calc_weights():
        self.raw_scores[0] *= 4 # multiply this weight by 4 because CRDC model predicts only location
        total_raw = np.sum(np.array(self.raw_scores))

        self.weights = np.array(self.raw_scores)/total_raw
    
    def train():
        self.init_models()
        self.calc_weights()

    def predict(self, data_per_model):
        """
        Note: the data_per_model parameter expects a length-4 array of data to be passed into each model, 
        i.e. [[arts_data], [crdc_data], [cte_data], [edtech_data]]. 

        Returns 
        """
        if self.models == None or self.weights == None:
            print("untrained! call model.train() before attempting to predict.")
        

        preds = []

        for i in range(4):
            curr_model = self.models[i]
            curr_data = data_per_model[i]

            preds.append(curr_model.fit(curr_data))
        
        # vote
        votes = [0]*16

        # add crdc votes
        for i in range(4):
            poss = preds[0]*4 + i
            votes[poss] += self.weights[0]
        
        # for the rest of the models
        for i in range(3):
            poss = preds[1 + i]
            votes[poss] += self.weights[i]
        
        votes = np.array(votes)
        # print results
        print("the highest-voted classification is:")
        print(keys[np.argmax(votes)])

        print("overall, here are the full votes:")
        sorted_votes = np.argsort(votes)
        for i in range(16):
            if (votes[sorted_votes[i]] > 0):
                print(str(votes[sorted_votes[i]]) + ", " + keys[sorted_votes[i]])
    

if __name__ == "__main__":
    model = combinedModel()
    model.train()
    print("This is a trained model that will predict a region and community classification for a given set of input.")
    print("To use, modify the main method to pass in (hypothetical) datasets to model.predict() -- arts, crdc, cte, and edtech.")

