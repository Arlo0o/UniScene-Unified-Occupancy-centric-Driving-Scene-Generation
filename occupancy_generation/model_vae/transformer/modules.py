import torch
from torch import nn
import torch.nn.functional as F

class FFN(nn.Module):
    
    def __init__(self, dims, hidden_dims, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(dims, hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dims, dims)
        self.drop = nn.Dropout(p=drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x        
    