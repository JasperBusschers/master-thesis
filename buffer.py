from util import pareto_dominates
import numpy as np
import random
import pickle

class buffer():
    def __init__(self,size):
        self.data = []
        self.size = size

    def save(self,dir,idx):
        with open(dir+str(idx)+"memory.pkl", "wb") as fp:
            pickle.dump(self.data, fp)
    def load(self,dir,idx):
        with open(dir+str(idx)+"memory.pkl", "rb") as fp:  # Unpickling
            self.data = pickle.load(fp)


    def add(self,trajectory , reward1,reward2):
        dominating = []
        dominated_by = []
        equal = []
        amount_visited = 1
        added = False
        for i ,sample in enumerate (self.data):
            t, r1, r2 , count = sample
            if pareto_dominates(reward1,reward2 , r1,r2):
                dominating.append(self.data[i])
                self.data.remove(self.data[i])
            if pareto_dominates(r1,r2,reward1,reward2):
                dominated_by.append(self.data[i])
            if r1 == reward1 and r2 == reward2 :
                equal.append(self.data[i])
                self.data[i][3]+=1
                amount_visited += self.data[i][3]
        if len(dominated_by)==0:
            if len(equal) == 0:
                self.data.append([trajectory,reward1,reward2, 0])

            added = True
        return added, len(dominating), amount_visited

    def search(self,state,action, index):
        found = False
        amount = 0
        times = 0
        total = 0
        for i, sample in enumerate(self.data):
            t, r1, r2,visited = sample
            for i, [s,a]  in enumerate(t):
                total += 1
                v =visited + 1
                if s == state and a == action :
                    found=True
                    times += 1
                    if  i == index:
                        amount +=1/v
                    else :
                        amount += (1/np.abs(i-index) -0.5)/v

        return found, amount

    def get_rewards(self):
        rewards = []
        for i, sample in enumerate(self.data):
            t, r1, r2,_ = sample
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


    def __sizeof__(self):
        for i, sample in enumerate(self.data):
            t, r1, r2,_ = sample
        return len(self.data)