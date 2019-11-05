import argparse

from main import main


def arguments():
    parse = argparse.ArgumentParser()
    #parameters for environment
    parse.add_argument('--environment', type=str, default='deep-sea-treasure-v0',help='environment to use')
    #parameters for Q agent
    parse.add_argument('--episodes', type=int, default=100000, help='amount of episodes')
    parse.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parse.add_argument('--eps', type=float, default=0.90, help='epsilon')
    parse.add_argument('--min_eps', type=float, default=0.05, help='minimum value epsilon')
    parse.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parse.add_argument('--eps_decay', type=float, default=0.999, help='learning rate')
    parse.add_argument('--high', type=int, default=1, help='higher boundary for Q_val initialization')
    parse.add_argument('--low', type=int, default=0, help='lower boundary for Q_val initialization')
    #multi objective Parameters
    parse.add_argument('--scalarization_method', type=str, default='Chebyshev',help='Linear , Chebyshev ')
    parse.add_argument('--weight1', type=float, default=1., help='weight for objective 1')
    parse.add_argument('--attraction1', type=int, default=1, help='attraction point1')
    parse.add_argument('--attraction2', type=int, default=1, help='attraction point2')
    args = parse.parse_args()
    return args

args = arguments()
main(args)