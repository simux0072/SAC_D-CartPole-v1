import Network
import torch.optim as optim
import copy

class Agent():
    def __init__(self, lr, input_dims, tau, n_actions):
        self.critic_local = Network.CriticNetwork(input_dims)
        self.critic_local2 = Network.CriticNetwork(input_dims)
        
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr)
        self.critic_optim2 = optim.Adam(self.critic_local2.parameters(), lr)
        
        self.critic_target = copy.deepcopy(self.critic_local)
        self.critic_target2 = copy.deepcopy(self.critic_local2)
        
        self.actor = Network.ActorNetwork(input_dims, n_actions)
        self.actor_optim = optim.Adam(self.actor.parameters())
        
        self.tau = tau
        
    def get_action(self, state):
        action, action_probs, log_action_probs = self.actor(state)
        return action, action_probs, log_action_probs
        
    def soft_update_target_networks(self):
        self.soft_update(self.critic_target, self.critic_local)
        self.soft_update(self.critic_target2, self.critic_local2)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        