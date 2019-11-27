import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np

from memory import Memory
from tabularQL import tabularQL
import cv2

from util import linear, chebychev
from visualizer import visualizer


def main(args):
    memory = Memory(1,20)
    plotter = visualizer()
    env = gym.make(args.environment)
    agent = tabularQL(env.observation_space.n, env.action_space.n ,args)
    rewards_o1, rewards_o2 = [],[]
    for e in range(args.episodes):
        state = env.reset()
        trajectory = []
        done = False
        total_reward = np.zeros([2])
        while not done:
            action = agent.act(state)
            next_state , rewards ,done , info =env.step(action)
            trajectory.append([state,action])
            agent.update(state, next_state, action, rewards ,done)
            state = next_state
            total_reward += np.asarray(rewards)
        if agent.eps > agent.min_eps:
            agent.eps = agent.eps * agent.eps_decay
        memory.add(0,trajectory,total_reward[0],total_reward[1])
        if not memory.empty(0):
            print("10 state action pairs sampled"+str(memory.sample(10,0)))
        print('rewards in buffer' + str(memory.get_rewards()[0]))
        print("episode " + str(e))
        print("total rewards " + str(total_reward))
        print("epsilon  " + str(agent.eps))
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])
        if e%args.plot_every == 0:
            plotter.plot_pareto_front(memory,args.name +str(e))
    plotter.make_gif(args.name)
    plotter.plot_rewards(rewards_o1, args.name + str('-rewards-objective1'))
    plotter.plot_rewards(rewards_o2, args.name + str('-rewards-objective2'))

