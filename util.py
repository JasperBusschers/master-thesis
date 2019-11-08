import numpy as np


def pareto_dominates(r11,r12, r21,r22):
    if r11 > r21 and r12 >= r22:
        return True
    elif r12 > r22 and r11 >= r21:
        return True
    else:
        return False

def chebychev(args, rewards):
    return - np.max ([args.weight1 * np.abs(rewards[0] - args.attraction1) , (1 - args.weight1) * np.abs(rewards[1] - args.attraction2)])

def linear(args, rewards):
    return args.weight1 * rewards[0]  + (1 - args.weight1) * rewards[1]