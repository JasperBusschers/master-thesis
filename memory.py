from agent_buffer import Agent_buffer
from buffer import buffer


class Memory(object):
    def __init__(self,number_of_buffers,buffer_size,args):
        self.buffers = [buffer(buffer_size) for i in range(number_of_buffers)]
        self.args = args
        self.agent_buffer = Agent_buffer(args.agent_buffer_size)


    def add_agent_experience(self,sample):
        self.agent_buffer.add(sample)

    def add_dom_buffer(self,  idx, trajectory, reward1,reward2):
        self.buffers[idx].add(trajectory , reward1,reward2)

    def sample_dom_buffer(self, amount, idx , policy= 'random'):
        samples = []
        for i in range(amount):
            if policy == 'random':
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