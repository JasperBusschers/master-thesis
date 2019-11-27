import argparse

from main import main


environments = ['bountiful-sea-treasure-v0' , 'deep-sea-treasure-v0' ]
strategy = ['Linear', 'Chebyshev']

def arguments(env,strat):
    parse = argparse.ArgumentParser()
    #parameters for environment
    parse.add_argument('--environment', type=str, default=env,help='environment to use')
    parse.add_argument('--name', type=str, default= env+"-"+strat, help='name to use when saving plots')
    parse.add_argument('--plot_every', type=int, default=100, help='how often to save plots')
    #parameters for Q agent
    parse.add_argument('--episodes', type=int, default=5000, help='amount of episodes')
    parse.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parse.add_argument('--eps', type=float, default=1, help='epsilon')
    parse.add_argument('--min_eps', type=float, default=0.1, help='minimum value epsilon')
    parse.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parse.add_argument('--eps_decay', type=float, default=0.999, help='learning rate')
    parse.add_argument('--high', type=int, default=124, help='higher boundary for Q_val initialization')
    parse.add_argument('--low', type=int, default=0, help='lower boundary for Q_val initialization')
    #multi objective Parameters
    parse.add_argument('--scalarization_method', type=str, default=strat,help='Linear , Chebyshev ')
    parse.add_argument('--weight1', type=float, default=1, help='weight for objective 1')
    parse.add_argument('--attraction1', type=float, default=124, help='attraction point1')
    parse.add_argument('--attraction2', type=float, default=19, help='attraction point2')
    args = parse.parse_args()
    return args


for env in environments:
    for strat in strategy:
        args = arguments(env,strat)
        main(args)