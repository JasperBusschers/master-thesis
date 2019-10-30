from time import sleep

import numpy as np
from threading import Thread as thread
import gui as interface


class treasure_hunter:
    def __init__(self,render=True):
        self.to_render = render
        self.map = np.zeros([10,10])
        self.pos = [0,0]
        self.action_space = 8
        self.observation_space = 2
        self.initialize()
        if render == True:
            self.gui = interface.gui()


    def reset(self):
        self.pos = [0,0]
        return np.asarray(self.pos)



    def initialize(self):
        self.map[1,0] = 1
        self.map[2,1] = 2
        self.map[3,2] = 3
        self.map[4,3] = 5
        self.map[4,4] = 8
        self.map[4,5] = 16
        self.map[6,6] = 24
        self.map[6,7] = 50
        self.map[8,8] = 74
        self.map[9,9] = 124


    def step(self, action):
        step_reward, treasure_reward = 0,0
        #go north
        if action == 0 and self.pos[0] > 0 :
                self.pos[0] -=1
        # go north east
        if action ==1 and self.pos[0]>0 and self.pos[1] <9:
            self.pos[0] += -1
            self.pos[1] += 1
        # go east
        if action ==2 and self.pos[1] <9 :
            self.pos[1]+=1
        # go south east
        if action ==3 and self.pos[0]<9 and self.pos[1] <9:
            self.pos[0] += 1
            self.pos[1] += 1
        # go south
        if action == 4 and self.pos[0] <9 :
            self.pos[0] += 1
        # go south-west
        if action == 5 and self.pos[0]<9 and self.pos[1] >0:
            self.pos[0] += 1
            self.pos[1] += -1
        # go west
        if action == 6 and self.pos[1] > 0:
            self.pos[1] += -1
        # go north west
        if action == 7 and self.pos[0] > 0 and self.pos[1] >0:
            self.pos[0] += -1
            self.pos[1] += -1
        step_reward = -1
        treasure_reward = self.map[self.pos[0],self.pos[1]]
        done = treasure_reward > 0
        return np.asarray(self.pos), treasure_reward, step_reward, done




    def test(self):
        self.reset()
        actions = [3,0,2,4,6,7,5,1,4,4,4]
        print("testing environment actions")
        for a in actions:
            print ("action taken " + str(a))
            state, treas_reward , step_reward , done  =self.step(a)
            self.render()
        assert done  and treas_reward == 3 ,  " expected episode to end on reward 3"


    def render(self):
        if self.to_render:
            map = np.copy(self.map)
            map[self.pos[0], self.pos[1]] = -5
            self.gui.create_grid(map)
            self.gui.render()
        map = np.copy(self.map)
        map[self.pos[0],self.pos[1]] = -5
        #print(map)



game = treasure_hunter()
game.test()

