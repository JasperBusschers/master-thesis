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

from tabular_disc_module import tabular_discriminator_module
from util import linear, chebychev
from visualizer import visualizer

Correct_solutions = [[1,-1],[2,-3], [3,-5] , [5,-7] ,[8,-8], [16,-9] , [24,-13], [50,-14], [74,-17],[124,-19] ]

def train_QL_agent(args, memory, index):
    print("training QL agent using " + args.scalarization_method)
    best_traj, best_reward = [] , [0,-100]
    env = gym.make(args.environment)
    agent = tabularSQL(env.observation_space.n, env.action_space.n, args)
    rewards_o1, rewards_o2 = [] , []
    for e in range(args.episodes):
        total_reward = [0,0]
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state , rewards ,done , info =env.step(action)
            total_reward += rewards
            trajectory.append([state, action])
            agent.update(state, next_state, action, rewards ,done)
            state = next_state
        agent.decay()
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])
        if total_reward[index] >= best_reward[index] :
            best_traj = trajectory
            best_reward = total_reward
        if args.log and e % args.log_every == 0:
            print("final state reward = " + str(total_reward))
            print("best final state reward = " + str(best_reward))
            print("epsilon  " + str(agent.eps))
    memory.add_dom_buffer(index, best_traj, total_reward[0], total_reward[1])
    return agent, rewards_o1

def sample_policy(policy,memory,mem_idx,args):
    print("sampling policy")
    env = gym.make(args.environment)
    for e in range(500):
        state = env.reset()
        trajectory = []
        done = False
        total_reward = np.zeros([2])
        steps = 0
        while not done:
            action = policy.act(state)
            next_state , rewards ,done , info =env.step(action)
            trajectory.append([state, action])
            total_reward += np.asarray(rewards)
            steps+=1
        if args.log and e % 50 == 0:
            print("final state reward = " + str(total_reward))
            print(policy.eps)
            print(steps)
        memory.add_dom_buffer(mem_idx,trajectory,total_reward[0],total_reward[1])


def QL_with_disc(args, discriminators,memory):
    rewards_o1, rewards_o2 = [], []
    Percentage_non_dominating = []
    counts = np.zeros([len(Correct_solutions)])
    total = 0
    non_dominated = 0
    env = gym.make(args.environment)
    if args.policy == "QL":
        args.high = 100
        args.low = 0
        agent = tabularQL(env.observation_space.n, env.action_space.n, args)
    for e in range(args.episodes):
        total_reward = np.zeros([2])
        sample = []
        trajectory = []
        state = env.reset()
        done = False
        disc_reward = 0
        step = 0
        while not done:
            action = agent.act(state)
            next_state, rewards, done, info = env.step(action)
            reward = discriminators.get_reward([state,action],args.method,memory, step)
            total_reward += np.asarray(rewards)
            trajectory.append([state, action])
            if len(sample) < 2*args.number_of_steps:
                sample.extend([state, action])
            else:
                memory.add_agent_experience(sample)
                sample = []
            if done:
                added , amount, visted = memory.add_dom_buffer(2, trajectory, total_reward[0], total_reward[1])
                if [total_reward[0], total_reward[1]] in Correct_solutions:
                    non_dominated += 1
                    index = Correct_solutions.index([total_reward[0], total_reward[1]] )
                    counts[index] +=1
                total += 1
                Percentage_non_dominating.append(non_dominated/total)
            agent.update(state, next_state, action, reward, done)
            state = next_state
            disc_reward += reward
            step += 1
        agent.decay()
        if args.log and e % args.log_every == 0:
            print("----------------EPISODE " + str(e) + "----------------" + args.name)
            print('rewards in buffer' + str(memory.get_rewards()[2]))
            print("episode " + str(e))
            print("final state reward = " + str(reward))
            print ("reward achieved from disc =  " + str(disc_reward))
            print("total rewards " + str(total_reward))
            print("times visited " + str(visted))
            print("epsilon  " + str(agent.eps))
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])

        # update discriminator
        if memory.length_agent_buffer() >= args.batch_size:
            for _ in range(10):
                discriminators.update(memory, 0, 0)
                discriminators.update(memory, 1, 1)
    return agent, rewards_o1,rewards_o2 ,Percentage_non_dominating, counts




def PG_with_disc(args, discriminators,memory):
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
        disc_reward=0
        step = 0
        while not done :
            action = agent.act(state)
            next_state, rewards, done, info = env.step(action)
            reward = discriminators.get_reward([state,action],args.method,memory,step)
            state_pool.append(state)
            disc_reward+=reward
            action_pool.append(float(action))
            total_reward += np.asarray(rewards)
            state = next_state
            trajectory.append([state, action])
            step += 1
            if step == 200:
                done = True
            if len(sample) < 2*args.number_of_steps:
                sample.extend([state, action])
            else:
                memory.add_agent_experience(sample)
                sample = []
            if done:
                added , amount, visted = memory.add_dom_buffer(2, trajectory, total_reward[0], total_reward[1])
            reward_pool.append(reward)
        if args.log and e % args.log_every == 0:
            print("----------------EPISODE " + str(e) + "----------------"+args.name)
            print('rewards in buffer' + str(memory.get_rewards()[2]))
            print("episode " + str(e))
            print("total rewards " + str(total_reward))
            print("final state reward = " + str(reward))
            print ("reward achieved from disc =  " + str(disc_reward))
            print("total rewards " + str(total_reward))
        rewards_o1.append(total_reward[0])
        rewards_o2.append(total_reward[1])
        # update discriminator
        if memory.length_agent_buffer() >= args.batch_size:
            for _ in range(5):
                discriminators.update(memory, 0, 0)
                discriminators.update(memory, 1, 1)
        if len(state_pool) >= args.batch_size:
            agent.update(state_pool,action_pool,reward_pool)
            state_pool = []
            action_pool = []
            reward_pool = []
    return agent,  rewards_o1,rewards_o2

def test(args):
    # step 1 train 2 extreme policies
    print("started training QL agents for extreme policies")
    args2 = args
    memory = Memory(3, 20, args)
    if not args.load_memory:
        if args.scalarization_method == 'linear':
            policy1 = train_QL_agent(args,memory,0)
            args.weight1 = 1- args.weight1
            policy2 = train_QL_agent(args,memory,1)
        else :
            policy1, r1 = train_QL_agent(args,memory,0)
            args2.attraction1 = 1
            args2.attraction2 = -1
            args2.high = 1
            args2.low = 0
            policy2, r2 = train_QL_agent(args2, memory, 1)
        memory.save_memory()
    else:
        memory.load_memory()
    # step 3 train new policy using discriminators
    if args.module == 'disc':
        discriminators = discriminator_module(args)
    if args.module == 'tab_disc':
        discriminators = tabular_discriminator_module(policy1.state_dim, policy1.action_dim, args)
    if args.policy == 'QL':
        print("started training using discriminator reward scheme and QL backend")
        agent, r1, r2 , percentage_non_dominated,counts= QL_with_disc(args, discriminators, memory)
    elif args.policy == 'PG':
        print("started training using discriminator reward scheme and PG backend")
        agent, r1, r2 = PG_with_disc(args, discriminators, memory)
    # visualize buffers for extreme policy and new policy
    plotter = visualizer()
    plotter.distribution(r1, args.name + "_rewards1_")
    #plotter.distribution(r2, args.name + "_rewards2_")
    plotter.plot_losses(args.name, discriminators.get_losses())
    print(counts)
    plotter.plot_success_percentage(args.name, percentage_non_dominated, "percentage_non_dominated")
    plotter.plot_pareto_front(memory, args.name + "extreme_pol", correct=Correct_solutions)






