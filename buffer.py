

class buffer():
    def __init__(self,size):
        self.data = []
        self.size = size

    def add(self,trajectory , reward):
        for t,r in self.data:
