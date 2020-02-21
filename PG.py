import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util import linear, chebychev, init_weights
from torch.distributions import Categorical
import torch
from torch.autograd import Variable

class PolicyNet(nn.Module):
    def __init__(self,state_dim, action_dim ):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(1, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, action_dim)  # Prob of Left
        self.apply(init_weights)
    def forward(self, x):
        x = torch.Tensor(x).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class PG_agent():
    def __init__(self, state_dim, action_dim , args):
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer =  torch.optim.RMSprop(self.policy.parameters(), lr=args.lr)
        self.args = args

    def act(self, state):
        probs = self.policy([[state]])
        m = Categorical(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0]
        return action

    def update(self, states,actions,rewards):
        # Discount reward
        running_add = 0
        for i in reversed(range(len(states))):
            if rewards[i] == 0:
                running_add = 0
            else:
                running_add = running_add *self.args.gamma + rewards[i]
                rewards[i] = running_add

        # Normalize reward
        #reward_mean = np.mean(rewards)
        #reward_std = np.std(rewards)
        #for i in range(len(states)):
        #    rewards[i] = (rewards[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(states)):
            state = states[i]
            action = Variable(torch.FloatTensor([actions[i]]))
            reward = rewards[i]
            probs = self.policy([[state]])[0]
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            loss.backward()
