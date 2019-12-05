import gym
import deep_sea_treasure
from gym import wrappers
import numpy as np

from PG import PG_agent
from disc_module import discriminator_module
from memory import Memory
from SQL import tabularSQL
from QL import tabularQL
import cv2

from util import linear, chebychev
from visualizer import visualizer


def train_QL_agent(args):
    env = gym.make(args.environment)
    agent = tabularSQL(env.observation_space.n, env.action_space.n, args)
    for e in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state , rewards ,done , info =env.step(action)
            agent.update(state, next_state, action, rewards ,done)
            state = next_state
    return agent

def sample_policy(policy,memory,mem_idx,args):
    env = gym.make(args.environment)
    for e in range(500):
        state = env.reset()
        trajectory = []
        done = False
        total_reward = np.zeros([2])
        while not done:
            action = policy.act(state)
            next_state , rewards ,done , info =env.step(action)
            trajectory.append([state, action])
            total_reward += np.asarray(rewards)
        memory.add_dom_buffer(mem_idx,trajectory,total_reward[0],total_reward[1])


def QL_with_disc(args, discriminators,memory,log= True):
    rewards_o1, rewards_o2 = [], []
    env = gym.make(args.environment)
    if args.policy == "QL":
        agent = tabularQL(env.observation_space.n, env.action_space.n, args)
    for e in range(args.episodes):
        total_reward = np.zeros([2])
        sample = []
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, rewards, done, info = env.step(action)
            reward = discriminators.get_reward([state,action],args.method)
            agent.update(state, next_state, action, reward, done)
            total_reward += np.asarray(rewards)
            state = next_state
            trajectory.append([state, action])
            if len(sample) < 2*args.number_of_steps:
                sample.extend([state, action])
            else:
                memory.add_agent_experience(sample)
                sample = []
        memory.add_dom_buffer(2, trajectory, total_reward[0], total_reward[1])
        if log:
            print("----------------EPISODE " + str(e) + "----------------")
            print('rewards in buffer' + str(memory.get_rewards()[2]))
            print("episode " + str(e))
            print("total rewards " + str(total_reward))
            print("epsilon  " + str(agent.eps))
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])
        # update discriminator
        if memory.length_agent_buffer() >= args.batch_size:
            discriminators.update(memory, 0, 0)
            discriminators.update(memory, 1, 1)
    return agent




def PG_with_disc(args, discriminators,memory,log= True):
    rewards_o1, rewards_o2 = [], []
    env = gym.make(args.environment)
    agent = PG_agent(env.observation_space.n, env.action_space.n, args)
    state_pool = []
    action_pool = []
    reward_pool = []
    for e in range(args.episodes):
        total_reward = np.zeros([2])
        sample = []
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, rewards, done, info = env.step(action)
            reward = discriminators.get_reward([state,action],args.method)
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            total_reward += np.asarray(rewards)
            state = next_state
            trajectory.append([state, action])
            if len(sample) < 2*args.number_of_steps:
                sample.extend([state, action])
            else:
                memory.add_agent_experience(sample)
                sample = []
        memory.add_dom_buffer(2, trajectory, total_reward[0], total_reward[1])
        if log:
            print("----------------EPISODE " + str(e) + "----------------")
            print('rewards in buffer' + str(memory.get_rewards()[2]))
            print("episode " + str(e))
            print("total rewards " + str(total_reward))
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])
        # update discriminator
        if memory.length_agent_buffer() >= args.batch_size:
            discriminators.update(memory, 0, 0)
            discriminators.update(memory, 1, 1)
        if len(state_pool) >= args.batch_size:
            agent.update(state_pool,action_pool,reward_pool)
            state_pool = []
            action_pool = []
            reward_pool = []
    return agent

def main(args):
    # step 1 train 2 extreme policies
    print("started training QL agents for extreme policies")
    policy1 = train_QL_agent(args)
    args.weight1 = 1- args.weight1
    policy2 = train_QL_agent(args)
    memory = Memory(3,20,args)
    # step 2 sample the policies in the buffer
    print("sampling extreme policies in buffer")
    sample_policy(policy1,memory,0,args)
    sample_policy(policy2, memory, 1, args)
    # step 3 train new policy using discriminators
    discriminators = discriminator_module(args)
    if args.policy == 'QL':
        print("started training using discriminator reward scheme and QL backend")
        agent  = QL_with_disc(args, discriminators,memory)
    elif args.policy == 'PG':
        print("started training using discriminator reward scheme and PG backend")
        agent = PG_with_disc(args, discriminators, memory)
    # visualize buffers for extreme policy and new policy
    plotter = visualizer()
    plotter.plot_pareto_front(memory, args.name + "extreme_pol")






