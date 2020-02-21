

import numpy as np

class tabular_discriminator_module():

    def __init__(self, state_dim, action_dim, args):
        self.Q = np.random.uniform(0, 1, [state_dim, action_dim])
        self.min_eps = args.min_eps
        self.args = args
        self.lr = 0.1#args.lr

    def update(self, memory , disc_index, memory_index):
        batch_size = self.args.batch_size
        batch_expert = memory.sample_dom_buffer(batch_size*5,memory_index,also_agent=True)
        agent_batch = memory.sample_experiences(batch_size)
        for s,a in batch_expert:
            reward = memory.search(s,a)
            self.Q[s,a] +=  self.lr * (reward - self.Q[s, a])
        for s,a in agent_batch:
            reward = memory.search(s, a)
            self.Q[s, a] += self.lr * (reward - self.Q[s, a])


    def get_reward(self,sample,method,memory):
        obs, action = sample
        reward = self.Q[obs,action]
        reward = memory.search(obs, action)
        return reward