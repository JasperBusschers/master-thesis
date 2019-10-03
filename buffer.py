

class buffer():
    def __init__(self,size):
        self.data = []
        self.size = size


    def insert(self,data):
        self.data.append(data)