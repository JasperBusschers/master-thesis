import numpy as np

from util import linear, chebychev


class tabularQL():
    def __init__(self, state_dim, action_dim , args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = args.lr
        self.eps = args.eps
        self.gamma = args.gamma
        self.eps_decay = args.eps_decay
        self.args = args
        self.Q = np.random.uniform(args.low , args.high ,[self.state_dim,action_dim])
        self.min_eps = args.min_eps

    def act(self, state):
        random = np.random.uniform()
        if random < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update(self, state, next_state, action ,reward,done):
        best = np.argmax(self.Q[state])
        if not done:
            self.Q[state,action] += self.lr * (reward + self.gamma * self.Q[next_state,best] - self.Q[state,action])
        else:
            self.Q[state, action] += self.lr * (reward - self.Q[state, action])
    def decay(self):
        if self.eps > self.min_eps:
            self.eps = self.eps * self.eps_decay