
import random



class Agent_buffer():
    def __init__(self,size):
        self.size = size
        self.buffer = []

    def add(self,sample):
        if len(self.buffer) > self.size:
            del self.buffer[-1]
        self.buffer.insert(0, sample)

    def sample(self,batch_size):
        return random.sample(self.buffer, batch_size)

    def length(self):
        return len(self.buffer)