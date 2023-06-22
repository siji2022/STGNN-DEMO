from os import path
import configparser
import numpy as np
import random
import gym
from envs.flocking import *
import torch
import sys
from learner.gnn_stgnn import train_stgnn
from learner.gnn_dagger import train_dagger
import datetime


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    # if isinstance(env.env, gym_flock.envs.FlockingRelativeEnv):
    env.env.params_from_cfg(args)
    fname = args.get('fname')

    sys.stdout = open(f'train_{fname}', "a")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f'start testing: {fname}, {env_name}, {args.get("n_agents")}')

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    # initialize params tuple
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.get('device')

    alg = args.get('alg').lower()
    if alg == 'dagger':
        stats = train_dagger(env, args, device)
    elif alg == 'dagger_siji':
        stats = train_stgnn(env, args, device)
    else:
        raise Exception('Invalid algorithm/mode name')
    return stats


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)


    if config.sections():
        for section_name in config.sections():

            print(f'start training model: {section_name}')
            stats = run_experiment(config[section_name])
            print(section_name + ", " +
                  str(stats['mean']) + ", " + str(stats['std']))
    else:
        val = run_experiment(config[config.default_section])
        print(val)



if __name__ == "__main__":
    main()
