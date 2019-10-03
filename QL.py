import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from collections import namedtuple

from env import treasure_hunter

EPISODES = 1000000
EPSILON = 1
MIN_EPSILON = 0.05
EPSILON_decay = 0.9995
GAMMA = 0.99
LEARNING_RATE = 5e-4
UPDATE_EVERY = 4
tau = 1e-3
ENV = 'LunarLander-v2'

state_dim=2
action_dim= 8
hidden_dim = 64
BATCH_SIZE = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.optimizer = opt.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(device)

    def forward(self,state):
        x = torch.tensor(state).float().to(device)
        x = F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x = self.fc3(x) # Q value for each action
        return x
    def save_checkpoint(self):
        torch.save(self.state_dict(), 'models/DQN'+ENV+'.pt')

    def load_checkpoint(self):
        self.load_state_dict(torch.load( 'models/DQN'+ENV+'.pt'))



class memory():
    def __init__(self, size):
        self.memory = []
        self.size = size
        self.mem_counter = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self,state,action,reward,next_state,done):
        e = self.experience(state, action, reward, next_state, done)
        if self.mem_counter < self.size:
            self.memory.append(e)
            self.mem_counter +=1
        else:
            self.memory[self.mem_counter%self.size] = e

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states =np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        return (states, actions, rewards, next_states, dones)




class agent():
    def __init__(self,memorySize):
        self.model_target = DQN()#target net
        self.model = DQN() #policy net
        self.memory = memory(memorySize)
        # duplicate weights in begin
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        self.steps = 0
        self.swap_netwerk_per = 100

    def choose_action(self,obs, eps):
        rand = np.random.random()
        actions = self.model(obs)
        if rand < 1-eps:
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(action_dim)
        return action

    def update(self, batch_size):
        if ((self.steps%self.swap_netwerk_per)==0):
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        if self.memory.mem_counter>batch_size:
            experiences = self.memory.sample(batch_size)
            states, actions, rewards, next_states, dones = experiences
            Q_targets_next = self.model_target(next_states).detach().max(1)[0].unsqueeze(1)
            #convert to tensors
            dones = torch.tensor((1 - dones)).to(device).float()
            rewards = torch.tensor(rewards).to(device).float()
            actions = torch.from_numpy(actions).long().to(device)
            #compute target based on value of next state
            Q_targets = rewards + (GAMMA * Q_targets_next * dones)
            # get prediction from current model
            Q_expected = self.model(states).gather(1, actions)
            # update current model
            loss = F.mse_loss(Q_expected, Q_targets)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            return loss.item()
        else:
            return 0

    def add(self,state,action,reward,next_state,done):
        return self.memory.add(state,action,reward,next_state,done)




def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]

def main():
    # Create the Gym environment. A good first environment is FrozenLake-v0
    env = treasure_hunter()
    print(env.action_space)
    print(env.observation_space)
    ag = agent(60000)
    HISTORY = []
    # Act randomly in the environment
    average_cumulative_reward = 0.0
    EPSILON = 1

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        t=0
        # Loop over time-steps
        while not terminate:
            a = ag.choose_action(state,EPSILON)
            # Perform the action
            next_state, treasure_reward, step_reward, terminate = env.step(a)
            r = treasure_reward# + step_reward
            #ag.add(r, next_state)
            #add experience to buffer
            ag.add(state,a,r,next_state,terminate)
            # Update statistics
            t = t+1
            if t%UPDATE_EVERY == 0:
                ag.update(BATCH_SIZE)
            cumulative_reward += r
            state = next_state
            # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
        EPSILON = max(EPSILON * EPSILON_decay, MIN_EPSILON)
        print(i, average_cumulative_reward)
        HISTORY.append(average_cumulative_reward)
        print("epsilon = " + str(EPSILON))
        print ("episode finished")
        print ("steps : "+ str(t) + " treasure reward " + str(treasure_reward) + " cumulative reward " + str(cumulative_reward))
        #if average_cumulative_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(average_cumulative_reward, t))
        #    ag.model.save_checkpoint()
        #    break
    return HISTORY


if __name__ == '__main__':
    result = main()
    name = 'results/' + ENV +'-DQN-cumulative-rewards.npy'
    np.save(name,result)