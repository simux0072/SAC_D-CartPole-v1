import Network
import torch.optim as optim
import copy
import torch
import numpy as np

class Agent():
    def __init__(self, lr, input_dims, tau, n_actions, Memory, delta, device):
        self.critic_local = Network.CriticNetwork(input_dims, n_actions)
        self.critic_local2 = Network.CriticNetwork(input_dims, n_actions)
        
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr)
        self.critic_optim2 = optim.Adam(self.critic_local2.parameters(), lr)
        
        self.critic_target = Network.CriticNetwork(input_dims, n_actions)
        self.copy_model_over(self.critic_local, self.critic_target)
        self.critic_target2 = Network.CriticNetwork(input_dims, n_actions)
        self.copy_model_over(self.critic_local2, self.critic_target2)

        self.actor = Network.ActorNetwork(input_dims, n_actions)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr)
        
        self.tau = tau
        self.memory = Memory
        self.device = device
        
        self.target_entropy = -0.98 * np.log(1/n_actions)  # -dim(A)
        self.log_alpha = torch.tensor(np.log(1), requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optim = optim.Adam([self.log_alpha], lr)
                
        self.delta = delta
        
    def get_action(self, state):
        action, action_probs, log_action_probs = self.actor(state)
        return action, action_probs, log_action_probs
    
    def train(self, batch_size):
        if self.memory.mem_cntr >= batch_size:
            states, actions, rewards, states_, dones = self.memory.sample_buffer(batch_size)
            
            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            states_ = torch.from_numpy(states_).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            dones = torch.from_numpy(dones).to(self.device)
            
            self.critic_optim.zero_grad()
            self.critic_optim2.zero_grad()
            self.actor_optim.zero_grad()
            self.alpha_optim.zero_grad()
            
            _, next_state_action_probs, next_state_log_probs = self.get_action(states_)
            critic_loss, critic_loss2 = self.critic_loss(next_state_action_probs, next_state_log_probs, states, states_, rewards, dones, actions)
                
            critic_loss.backward(retain_graph=True)
            critic_loss2.backward(retain_graph=True)
            self.critic_optim.step()
            self.critic_optim2.step()
            
            actor_loss, log_action_probs = self.actor_loss(states)
            
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            
            alpha_loss = self.temp_loss(log_action_probs)
            
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            
            self.soft_update_target_networks()
            return actor_loss.item(), critic_loss.item(), critic_loss2.item(), alpha_loss.item()
        return 0, 0, 0, 0
        
    def critic_loss(self, action_probs, log_probs, states, states_, rewards, dones, actions):
        
        with torch.no_grad():
            next_q_values_target = self.critic_target.forward(states_)
            next_q_value_target_2 = self.critic_target2.forward(states_)
            
            soft_state_values = (action_probs * (torch.min(next_q_values_target, next_q_value_target_2) - self.alpha * log_probs))
            
            next_q_values = (rewards + (1 - dones.to(dtype=torch.int)) * self.delta * soft_state_values.sum(dim=-1)).unsqueeze(-1)
        
        soft_q_values = self.critic_local(states).gather(1, actions.unsqueeze(-1).long())
        soft_q_values2 = self.critic_local2(states).gather(1, actions.unsqueeze(-1).long())
        critic_loss = torch.nn.functional.mse_loss(soft_q_values, next_q_values)
        critic_loss2 =  torch.nn.functional.mse_loss(soft_q_values2, next_q_values)
        
        return critic_loss, critic_loss2
    
    def actor_loss(self, states):
        _, action_probs, log_action_probs = self.get_action(states)
        q_values_local = self.critic_local(states)
        q_values_local2 = self.critic_local2(states)
        
        
        inside_term = self.alpha * log_action_probs * torch.min(q_values_local, q_values_local2)
        actor_loss = (action_probs * inside_term.detach()).sum(-1).mean()
        return actor_loss, log_action_probs
    
    def temp_loss(self, log_action_probs):
        alpha_loss = -(self.log_alpha.exp() * (log_action_probs + self.target_entropy).detach()).mean()
        return alpha_loss
    
    def soft_update_target_networks(self):
        self.soft_update(self.critic_target, self.critic_local)
        self.soft_update(self.critic_target2, self.critic_local2)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    @staticmethod
    def copy_model_over(from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())