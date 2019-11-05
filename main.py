import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np
from tabularQL import tabularQL
import cv2


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
            if True:#args.render:
                frame = env.render('rgb_array')
                cv2.imshow('lol',frame)
            action = agent.act(state)
            next_state , rewards ,done , info =env.step(action)
            if args.scalarization_method == 'Linear':
                reward = args.weight1 * rewards[0] + (1 - args.weight1) * rewards[1]
            elif args.scalarization_method == 'Chebyshev':
                reward = args.weight1 * np.abs(rewards[0] - args.attraction1) + (1 - args.weight1) * np.abs(rewards[1] - args.attraction2)
            agent.update(state, next_state, action, reward )
            state = next_state
            total_reward += np.asarray(rewards)
        if agent.eps > agent.min_eps:
            agent.eps = agent.eps * agent.eps_decay
        print("episode " + str(e))
        print("total rewards " + str(total_reward))
        print("epsilon  " + str(agent.eps))


