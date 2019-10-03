import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Categorical
import numpy as np

from env import treasure_hunter

LOG_INTERVAL = 10
RENDER = False
GAMMA = 0.99
LEARNING_RATE = 5e-4
BETAS = (0.9, 0.99)

ALGORITH = 'Policy_gradient'


EPISODES = 100000
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



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        #dropout to reduce variance in policy gradient
        self.dropout = nn.Dropout(p=0.5)
        self.optimizer = opt.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(device)
        # hold log_probs and rewards of on policy trajectory
        self.log_probs = []
        self.rewards = []



    def forward(self,state):
        shape =  state.shape
        x = torch.zeros([shape[0],  20]).float()
        for i , s in enumerate(state):
            x[i][s[0].int().item()] = 1
            x[i][s[1].int().item()+10] = 1
        x = x.to(device)# torch.tensor(state).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3(x)
        return F.softmax(action_scores, dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), 'models/' + ALGORITH + '-' + ENV + '.pt')

    def load_checkpoint(self):
        self.load_state_dict(torch.load('models/' + ALGORITH + ENV + '-' + ENV + '.pt'))


class agent():
    def __init__(self):
        self.model = Model()
    def choose_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        loss = torch.tensor([0]).to(device).float()
        returns = []
        for r in self.model.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        #subtract the mean as baseline
        returns = returns - returns.mean()
        #policy_loss[i] = -log_prob * R[i]
        for log_prob, R in zip(self.model.log_probs, returns):
            loss += -log_prob * R

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.model.rewards = []
        self.model.log_probs = []

    def add(self,reward):
        self.model.rewards.append(reward)



def main():
    # Create the Gym environment. A good first environment is FrozenLake-v0
    env = treasure_hunter()
    print(env.action_space)
    print(env.observation_space)
    ag = agent()
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
            a = ag.choose_action(state)
            # Perform the action
            next_state, treasure_reward, step_reward, terminate = env.step(a)
            r = 0.7* treasure_reward #+ 0.1*step_reward

            ag.add(r)
            # Update statistics
            t = t+1
            cumulative_reward += r
            state = next_state
            # Per-episode statistics
        ag.update()
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
        print(i, average_cumulative_reward)
        print ("steps : " + str(t) + " treasure reward " + str(treasure_reward) + " cumulative reward " + str(
            cumulative_reward))
        HISTORY.append(average_cumulative_reward)
       # if average_cumulative_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
         #         "the last episode runs to {} time steps!".format(average_cumulative_reward, t))
          #  ag.model.save_checkpoint()
           # break
    return HISTORY




if __name__ == '__main__':
    result = main()
    name = 'results/' + ENV + '-'+ALGORITH+ '-cumulative-rewards.npy'
    np.save(name, result)