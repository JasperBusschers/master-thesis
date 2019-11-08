import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np

from buffer import buffer
from tabularQL import tabularQL
import cv2

from util import linear, chebychev



def main(args):
    memory = buffer(10)
    env = gym.make(args.environment)
    agent = tabularQL(env.observation_space.n, env.action_space.n ,args)
    for e in range(args.episodes):
        state = env.reset()
        trajectory = []
        done = False
        total_reward = np.zeros([2])
        cumulative_reward = 0
        while not done:
            action , Qval= agent.act(state)
            next_state , rewards ,done , info =env.step(action)
            trajectory.append([state,action])
            agent.update(state, next_state, action, rewards ,done)
            state = next_state
            total_reward += np.asarray(rewards)
        if agent.eps > agent.min_eps:
            agent.eps = agent.eps * agent.eps_decay
        memory.add(trajectory,total_reward[0],total_reward[1])
        print('rewards in buffer' + str(memory.get_rewards()))
        print("episode " + str(e))
        print("total rewards " + str(total_reward))
        print("epsilon  " + str(agent.eps))


