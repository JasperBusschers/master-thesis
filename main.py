import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np
from tabularQL import tabularQL
import cv2

from util import linear, chebychev


def record_frame(self, episode, frames=None):
        frame = self.env.render('rgb_array')
        w, h = frame.shape[:2]
        r = h / w
        w = np.minimum(w, 100)
        h = int(r * w)
        frame = cv2.resize(frame, (h, w))
        frame = np.expand_dims(frame, axis=0)
        if frames is None:
            return frame
        else:
            return np.append(frames, frame, axis=0)

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

            agent.update(state, next_state, action, rewards )
            state = next_state
            total_reward += np.asarray(rewards)
        if agent.eps > agent.min_eps:
            agent.eps = agent.eps * agent.eps_decay
        print("episode " + str(e))
        print("total rewards " + str(total_reward))
        print("epsilon  " + str(agent.eps))


