import random

from agent_buffer import Agent_buffer
from buffer import buffer


class Memory(object):
    def __init__(self,number_of_buffers,buffer_size,args):
        self.buffers = [buffer(buffer_size) for _ in range(number_of_buffers)]
        self.args = args
        self.agent_buffer = Agent_buffer(args.agent_buffer_size)


    def save_memory(self,dir = 'checkpoints/'):
        for i , b in enumerate(self.buffers):
            b.save(dir, i)

    def load_memory(self,dir = 'checkpoints/'):
        for i ,b in enumerate(self.buffers):
            b.load(dir, i)


    def add_agent_experience(self,sample):
        self.agent_buffer.add(sample)

    def search(self,state,action):
        reward = -10
        for b in self.buffers:
            if b.search(state,action):
                reward =1

        return  reward

    def add_dom_buffer(self,  idx, trajectory, reward1,reward2):
        added , amount, amount_visited= self.buffers[idx].add(trajectory , reward1,reward2)
        return  added, amount , amount_visited

    def sample_dom_buffer(self, amount, idx , policy= 'random', also_agent = False):
        samples = []
        for i in range(amount):
            if policy == 'random':
                rand = random.random()
                if rand > 0.5 and also_agent and not self.agent_buffer.empty():
                    sample = self.sample_experiences(1)[0]
                else:
                    sample = self.buffers[idx].sample(policy,self.args.number_of_steps)
                samples.append(sample)
        return samples

    def get_rewards(self):
        result = []
        for buf in self.buffers:
            rewards  = buf.get_rewards()
            if len(rewards) != 0 :
                result.append(rewards)
        return result

    def empty(self,idx):
        return self.buffers[idx].empty()

    def length_agent_buffer(self):
        return self.agent_buffer.length()
    def sample_experiences(self, amount):
        return self.agent_buffer.sample(amount)