import Memory
import Agent
import gym
import torch
import data

lr = 0.0001
tau = 0.01
alpha = 1
delta = 0.99

batch_size = 400

max_size = 10000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()
input_dims = env.observation_space.shape
n_actions = 2

memory = Memory.Memory(max_size, input_dims, n_actions)
agent = Agent.Agent(lr, input_dims[0], tau, n_actions, memory, delta, device)
data = data.Data()

gen = 0
score = []
actor_loss = []
critic_loss = []
alpha_loss = []

while True:
    state = torch.from_numpy(state[0])
    step = 0
    curr_reward = 0
    while True:
        step += 1
        action, _, _ = agent.get_action(state.to(device))
        new_state, reward, is_dead, _, _ = env.step(action.item())
        
        new_state = torch.from_numpy(new_state)
        
        memory.store_transition(state, action.item(), reward, new_state, is_dead)
        state = new_state
        curr_reward += reward
        
        if is_dead:
            break
    
    actor_temp, critic_temp, alpha_temp = agent.train(batch_size)
    gen += 1
    actor_loss.append(actor_temp)
    critic_loss.append(critic_temp)
    alpha_loss.append(alpha_temp)
    score.append(curr_reward)
    
    if len(score) > 100:
        score.pop(0)
        actor_loss.pop(0)
        critic_loss.pop(0)
        alpha_loss.pop(0)
        data.update_data([gen, score[-1], sum(score)/100, actor_loss[-1], sum(actor_loss)/100, critic_loss[-1], sum(critic_loss)/100, alpha_loss[-1], sum(alpha_loss)/100, agent.alpha.item()])
    else:
        data.update_data([gen, score[-1], 0, actor_loss[-1], 0, critic_loss[-1], 0, alpha_loss[-1], 0, agent.alpha.item()])

    state = env.reset()