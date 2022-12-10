import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.dtype = torch.float32
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=36, dtype=self.dtype)
        self.fc2 = nn.Linear(in_features=36, out_features=36, dtype=self.dtype)
        self.fc3 = nn.Linear(in_features=36, out_features=36, dtype=self.dtype)
        self.q = nn.Linear(in_features=36, out_features=1, dtype=self.dtype)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.q.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        res = torch.relu(self.fc1(state))
        res = torch.relu(self.fc2(res))
        # res = torch.relu(self.fc3(res))
        
        q_val = self.q(res)
        
        return q_val

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.reparam_noise = 1e-8
        self.dtype = torch.float32
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=36, dtype=self.dtype)
        self.fc2 = nn.Linear(in_features=36, out_features=36, dtype=self.dtype)
        self.fc3 = nn.Linear(in_features=36, out_features=n_actions, dtype=self.dtype)
        # self.dist = nn.Linear(in_features=256, out_features=n_actions, dtype=self.dtype)
        self.softmax = nn.Softmax(dim=-1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        res = torch.relu(self.fc1(state))
        res = torch.relu(self.fc2(res))
        # res = torch.relu(self.fc3(res))
        res = self.fc3(res)
        # res = self.dist(res)
        
        action_probs = self.softmax(res)
        dist = Categorical(action_probs)
        action = dist.sample().view(-1, 1)
        
        z = (action_probs == 0.0).float() * self.reparam_noise
        log_action_probs = torch.log(action_probs + z)
        
        return action, action_probs, log_action_probs