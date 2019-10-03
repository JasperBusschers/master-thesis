import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Categorical
import numpy as np
from collections import namedtuple
import random

from env import treasure_hunter

LOG_INTERVAL = 10
RENDER = False
GAMMA = 1#0.99
LEARNING_RATE = 5e-4
BETAS = (0.9, 0.99)
CLIP_GRAD = 0.1

ALGORITH = 'ActorCritic'

EPISODES = 10000
EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_decay = 0.995
UPDATE_EVERY =4
tau = 1e-3
ENV = 'LunarLander-v2'

state_dim=20
action_dim= 8
hidden_dim = 64
BATCH_SIZE = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#to try 2 different networks

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.optimizer = opt.Adam(self.parameters(), lr=LEARNING_RATE)
        # hold log_probs and rewards of on policy trajectory
        self.log_probs = []
        self.values =[]
        self.rewards = []
        self.next_state_values = []



        # actor
        self.policy = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # critic
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(device)
    def forward(self,state):
        shape = state.shape
        x = torch.zeros([shape[0], 20]).float()
        for i, s in enumerate(state):
            x[i][s[0].int().item()] = 1
            x[i][s[1].int().item() + 10] = 1
        state= x.to(device)
        state = self.fc1(state.to(device))
        x = self.policy(state)
        value = self.critic(state)
        return F.softmax(x, dim=1), value

    def save_checkpoint(self):
        torch.save(self.state_dict(), 'models/'+ALGORITH+'-'+ENV+'.pt')

    def load_checkpoint(self):
        self.load_state_dict(torch.load( 'models/'+ALGORITH+ENV+'-'+ENV+'.pt'))





class agent():
    def __init__(self, memorySize):
        self.model = Model()

    def choose_action(self,state):
        state = torch.from_numpy(state).float().view(1,2)
        probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.log_probs.append(m.log_prob(action))
        self.model.values.append(value)
        return action.item() , value.item()

    def update(self):
        R = 0
        #calculate R-V(si)
        returns = []
        for r in self.model.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, [R])
        returns  = torch.tensor(returns).float().to(device)
        pol_loss = torch.tensor([0]).float().to(device)
        actor_loss = torch.tensor([0]).float().to(device)

        for log_prob, value,next_state_value , reward in zip(self.model.log_probs, self.model.values,self.model.next_state_values, returns):
            value = value[0]
            actor_loss += F.mse_loss(value,reward)
            advantage = (reward-value)
            pol_loss += -log_prob * advantage
        loss = 0.5*pol_loss+ 0.5*actor_loss
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.model.rewards = []
        self.model.values = []
        self.model.log_probs = []
        self.model.next_state_values = []

    def add(self, reward,next_state):
        state = torch.from_numpy(next_state).float().unsqueeze(0)
        probs, value = self.model(state)
        self.model.next_state_values.append(value)
        self.model.rewards.append(reward)



def main():
    # Create the Gym environment. A good first environment is FrozenLake-v0
    env = treasure_hunter()
    print(env.action_space)
    print(env.observation_space)
    ag = agent(6000)
    HISTORY = []
    # Act randomly in the environment
    average_cumulative_reward = 0.0
    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        t=0
        # Loop over time-steps
        while not terminate:
            a , value = ag.choose_action(state)
            # Perform the action
            next_state, treasure_reward , step_reward , terminate = env.step(a)
            r = 0.99*treasure_reward + 0.01* step_reward
            ag.add( r, next_state)
            # Update statistics
            t = t+1
            cumulative_reward += r
            state = next_state
            # Per-episode statistics
        print ("episode finished")
        print treasure_reward
        print cumulative_reward
        ag.update()
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
        print(i, average_cumulative_reward)
        HISTORY.append(average_cumulative_reward)
        #if average_cumulative_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(average_cumulative_reward, t))
        #    ag.model.save_checkpoint()
        #    break
    return HISTORY




if __name__ == '__main__':
    result = main()
    name = 'results/' + ENV + '-'+ALGORITH+ '-cumulative-rewards.npy'
    np.save(name, result)