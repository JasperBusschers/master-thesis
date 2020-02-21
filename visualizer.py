
import numpy as np
import matplotlib.pyplot as plt
import imageio


class visualizer():
    def __init__(self):
        self.file_names = []
    def plot_pareto_front(self,memory, name, correct =[]):
        for i,  rewards in enumerate(memory.get_rewards()):
            x, y = zip(*rewards)
            for X,Y in zip(x,y):
                if [X,Y] in correct:
                    color = 'green'
                else:
                    color = 'red'
                axes = plt.gca()
                axes.set_xlim([0, 130])
                axes.set_ylim([0, -30])

                plt.plot(X, Y, 'ro',color = color)
            plt.title(name+'_' + str(i))
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

    def plot_losses(self, name, losses, extension = 'disc'):
        for i,loss in enumerate(losses):
            plt.plot(loss , label = 'disc '+ str(i))
        plt.title(extension+'-' + str(i) )
        plt.legend(loc='upper left')
        plt.savefig('plots/'+name + '-'+extension + str(i) +'.png')
    def distribution(self,data, name):
        plt.hist(data, color='blue', edgecolor='black',)
        filename = 'plots/' + name + '.png'
        plt.savefig(filename)
        plt.close()