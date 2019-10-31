from util import pareto_dominates


class buffer():
    def __init__(self,size):
        self.data = []
        self.size = size

    def add(self,trajectory , reward1,reward2):
        dominated_set = []
        for i , t,r1, r2 in enumerate (self.data):
            if pareto_dominates(reward1,reward2 , r1,r2):
                dominated_set.append(self.data[i])
                self.data.remove(self.data[i])
