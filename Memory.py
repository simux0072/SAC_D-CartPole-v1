import numpy as np

class Memory():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.dtype = np.float32
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=self.dtype)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=self.dtype)
        self.action_memory = np.zeros((self.mem_size, 1), dtype=self.dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=self.dtype)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, state_, terminal):
        idx = self.mem_cntr % self.mem_size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = terminal
        
        self.mem_cntr += 1
        
    def sample_transitions(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, states_, rewards, dones