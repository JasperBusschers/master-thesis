from util import pareto_dominates
import numpy as np
import random

class buffer():
    def __init__(self,size):
        self.data = []
        self.size = size

    def add(self,trajectory , reward1,reward2):
        dominating = []
        dominated_by = []
        for i ,sample in enumerate (self.data):
            t, r1, r2 = sample
            if pareto_dominates(reward1,reward2 , r1,r2):
                dominating.append(self.data[i])
                self.data.remove(self.data[i])
            if pareto_dominates(r1,r2,reward1,reward2):
                dominated_by.append(self.data[i])
            if r1 == reward1 and r2 == reward2 :
                dominated_by.append(self.data[i])
        if len(dominated_by)==0:
            self.data.append([trajectory,reward1,reward2])



    def get_rewards(self):
        rewards = []
        for i, sample in enumerate(self.data):
            t, r1, r2 = sample
            rewards.append( [r1, r2] )
        return rewards

    def sample(self,policy,number_of_steps=2):
        result = []
        if policy == 'random':
            idx = random.randint(0,len(self.data)-(1))
            data = self.data[idx]
            trajectory = data[0]
            idx = random.randint(0,max(0,len(trajectory)-(1+number_of_steps)))
            for i in range(number_of_steps):
                if len(trajectory) > idx+i:
                    sample = trajectory[idx+i]
                else:
                    sample = [-1,-1]
                result.extend(sample)
        return result

    def empty(self):
        return len(self.data)==0
