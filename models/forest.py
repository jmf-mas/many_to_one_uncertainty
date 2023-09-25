from sklearn.ensemble import RandomForestRegressor
import pickle
from pathlib import Path

class FOREST():
    
    def __init__(self, max_depth, name):
        
        self.regressor = RandomForestRegressor(max_depth=max_depth, random_state=0)
        self.name = name
    
    
    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name+"/"+self.name+".pickle", "wb") as fp:
            pickle.dump(self.regressor, fp)


    def load(self):
        parent_name = "checkpoints"
        with open(parent_name+"/"+self.name+".pickle", "rb") as fp:
            self.regressor = pickle.load(fp)