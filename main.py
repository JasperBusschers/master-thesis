import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np
from tabularQL import tabularQL
import cv2

from util import linear, chebychev



def main(args):

    env = gym.make(args.environment)
    agent = tabularQL(env.observation_space.n, env.action_space.n ,args)
    for e in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = np.zeros([2])
        cumulative_reward = 0
        while not done:
            action , Qval= agent.act(state)
            next_state , rewards ,done , info =env.step(action)

            agent.update(state, next_state, action, rewards ,done)
            state = next_state
            total_reward += np.asarray(rewards)
        if agent.eps > agent.min_eps:
            agent.eps = agent.eps * agent.eps_decay
        print("episode " + str(e))
        print("total rewards " + str(total_reward))
        print("epsilon  " + str(agent.eps))


