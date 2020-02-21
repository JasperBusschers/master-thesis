import argparse

from main import main
from test import test

environments = [ 'deep-sea-treasure-v0' ] #'bountiful-sea-treasure-v0' ,
strategy = ['Linear', 'Chebyshev']
methods = ['sum', 'logsum', 'logsumdiff']
def arguments(env,strat = 'Linear', method = 'logsum' , pol = 'QL'):
    parse = argparse.ArgumentParser()
    #parameters for environment
    parse.add_argument('--environment', type=str, default=env,help='environment to use')
    parse.add_argument('--name', type=str, default= env+"-"+strat+'-'+method+'-'+pol, help='name to use when saving plots')
    parse.add_argument('--plot_every', type=int, default=100, help='how often to save plots')
    parse.add_argument('--log_every', type=int, default=1000, help='how often to log')
    parse.add_argument('--log',   default=True, type=lambda x: (str(x).lower() == 'true'), help='whether to display console debugging output')
    parse.add_argument('--load_memory', default=True, type=lambda x: (str(x).lower() == 'true'),
                       help='whether to load buffer from memory')
    #parameters for Q agent
    parse.add_argument('--episodes', type=int, default=5000, help='amount of episodes')
    parse.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parse.add_argument('--eps', type=float, default=0.9, help='epsilon')
    parse.add_argument('--min_eps', type=float, default=0.05, help='minimum value epsilon')
    parse.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parse.add_argument('--eps_decay', type=float, default=0.999, help='epsilon decay')
    parse.add_argument('--high', type=int, default=124, help='higher boundary for Q_val initialization')
    parse.add_argument('--low', type=int, default=100, help='lower boundary for Q_val initialization')
    #multi objective Parameters
    parse.add_argument('--scalarization_method', type=str, default=strat,help='Linear , Chebyshev ')
    parse.add_argument('--weight1', type=float, default=1, help='weight for objective 1')
    parse.add_argument('--attraction1', type=float, default=124, help='attraction point1')
    parse.add_argument('--attraction2', type=float, default=-19, help='attraction point2')
    #discriminator Parameters
    parse.add_argument('--policy', type=str, default=pol, help='learning approach for policy')
    parse.add_argument('--module', type=str, default="disc", help='normal or tabular discriminator')
    parse.add_argument('--method', type=str, default=method, help='approach for giving reward signal')
    parse.add_argument('--number_of_steps', type=int, default=1, help='how many state action pairs per sample')
    parse.add_argument('--agent_buffer_size', type=int, default=30, help='how many states to keep in memeory')
    parse.add_argument('--amount_of_disc', type=int, default=2, help='how many discriminators')
    parse.add_argument('--batch_size', type=int, default=10, help='batch size for updating discriminator')
    parse.add_argument('--disc_lr', type=float, default=0.05, help='learning rate discriminators')
    args = parse.parse_args()
    return args

def run1(env):
    for m in methods:
        for policy in ['QL','PG']:
            args = arguments(env,strat="Chebyshev",method=m , pol=policy)
            test(args)

def run2():
    for env in environments:
        strat = 'Linear'
        for m in methods:
            args = arguments(env,strat,m)
            main(args)


args = arguments('deep-sea-treasure-v0','Chebyshev','logsum')
run1('deep-sea-treasure-v0')