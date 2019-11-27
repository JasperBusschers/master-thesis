
import numpy as np
import matplotlib.pyplot as plt



class visualizer():
    def plot_pareto_front(self,memory, name):
        for i,  rewards in enumerate(memory.get_rewards()):
            x, y = zip(*rewards)
            plt.scatter(x, y)
            plt.title(name)
            plt.xlabel("reward objective 1")
            plt.ylabel("reward objective 2")
            plt.savefig('plots/'+name+'_' +str(i)+'.png')
            plt.close()

