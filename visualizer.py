
import numpy as np
import matplotlib.pyplot as plt
import imageio


class visualizer():
    def __init__(self):
        self.file_names = []
    def plot_pareto_front(self,memory, name):
        for i,  rewards in enumerate(memory.get_rewards()):
            x, y = zip(*rewards)
            plt.scatter(x, y)
            plt.title(name)
            plt.xlabel("reward objective 1")
            plt.ylabel("reward objective 2")
            filename = 'plots/'+name+'_' +str(i)+'.png'
            self.file_names.append(filename)
            plt.savefig(filename)
            plt.close()
    def make_gif(self, name):
        images =[]
        for i, filename in enumerate(self.file_names):
            images.append(imageio.imread(filename))
        imageio.mimsave('plots/gifs/'+name+'.gif', images,format='GIF', duration=2)

    def plot_rewards(self,rewards,name):
        plt.plot(rewards)
        plt.title(name)
        plt.ylabel('reward')
        plt.xlabel('episode')
        filename = 'plots/' + name + '.png'
        plt.savefig(filename)
        plt.close()

