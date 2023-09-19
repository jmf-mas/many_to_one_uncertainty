import torch.nn as nn
import pickle
from pathlib import Path

class MLP(nn.Module):
    
    def __init__(self, in_dim, name):
        super(MLP, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.name = name
    
    def forward(self, x):
        return self.regressor(x)
    
    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name+"/"+self.name+".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/"+self.name+".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))