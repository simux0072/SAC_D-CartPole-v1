import torch
import torch.nn as nn
from torch.distributions import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        
        self.fc1 = nn.Linear(in_features=input_dims[1], out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.q = nn.Linear(in_features=256, out_features=1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        res = torch.relu(self.fc1(state))
        res = torch.relu(self.fc2(res))
        res = torch.relu(self.fc3(res))
        
        q_val = self.q(res)
        
        return q_val

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.reparam_noise = 1e-8
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.dist = nn.Linear(in_features=256, out_features=n_actions)
        self.softmax = nn.Softmax(dim=-1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        res = torch.relu(self.fc1(state))
        res = torch.relu(self.fc2(res))
        res = torch.relu(self.fc3(res))
        
        action_probs = self.softmax(res)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        z = (action_probs == 0.0).float() * self.reparam_noise
        log_action_probs = torch.log(action_probs + z)
        
        return action, action_probs, log_action_probs