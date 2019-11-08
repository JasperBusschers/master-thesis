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
        self.Q = np.random.uniform(args.low , args.high ,[self.state_dim,action_dim,2])
        self.min_eps = args.min_eps

    def act(self, state):
        random = np.random.uniform()
        if random < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            if self.args.scalarization_method == 'Linear':
                qvals = [linear(self.args,q_vals) for q_vals in self.Q[state]]
            elif self.args.scalarization_method == 'Chebyshev':
                qvals = [chebychev(self.args,q_vals) for q_vals in self.Q[state]]
            action = np.argmax(qvals)
        return action , self.Q[state,action]

    def update(self, state, next_state, action ,reward,done):
        reward1, reward2 = reward
        if self.args.scalarization_method == 'Linear':
            qvals = [linear(self.args, q_vals) for q_vals in self.Q[next_state]]
        elif self.args.scalarization_method == 'Chebyshev':
            qvals = [chebychev(self.args, q_vals) for q_vals in self.Q[next_state]]
        best = np.argmax(qvals)
        if not done:
            self.Q[state,action,0] += self.lr * (reward1 + self.gamma * self.Q[next_state,best,0] - self.Q[state,action,0])
            self.Q[state, action, 1] += self.lr * (reward2 + self.gamma * self.Q[next_state, best, 1] - self.Q[state, action,1])
        else:
            self.Q[state, action, 0] += self.lr * (reward1 - self.Q[state, action, 0])
            self.Q[state, action, 1] += self.lr * (reward2 - self.Q[state, action, 1])
