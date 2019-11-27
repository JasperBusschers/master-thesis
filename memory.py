
from buffer import buffer


class Memory(object):
    def __init__(self,number_of_buffers,buffer_size):
        self.buffers = [buffer(buffer_size) for i in range(number_of_buffers)]



    def add(self,  idx, trajectory, reward1,reward2):
        self.buffers[idx].add(trajectory , reward1,reward2)

    def sample(self, amount, idx , policy= 'random'):
        samples = []
        for i in range(amount):
            if policy == 'random':
                sample = self.buffers[idx].sample(policy)
                samples.append(sample)
        return samples

    def get_rewards(self):
        return [self.buffers[idx].get_rewards() for idx in range(len(self.buffers))]

    def empty(self,idx):
        return self.buffers[idx].empty()