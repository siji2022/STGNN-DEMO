from os import path
import configparser
import numpy as np
import random
import gym
from envs.flocking import *
import torch
import sys
import pandas as pd
from learner.gnn_dagger_siji import DAGGER
import torch.nn.functional as F
import datetime
import threading
from multiprocessing.pool import ThreadPool
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


def test(args, actor_path, render=True):
    # initialize gym env
    env_name = args.get('env')
    fname = args.get('fname')
    n_agents = args.get('n_agents')

    # env = Monitor(gym.make(env_name),
    #               f'./video/{env_name}/{fname}/', force=True)
    # env.env.env.params_from_cfg(args)
    env = gym.make(env_name)
    env.env.params_from_cfg(args)

    sys.stdout = open(f'test_{fname}', "a")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f'start testing: {fname}, {env_name}, {n_agents}')

    # use seed
    seed = args.getint('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

    # initialize params tuple
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.get('device')
    learner = DAGGER(device, args)
    n_test_episodes = args.getint('n_test_episodes')

    if args.getboolean('transfer_test'):
        transfer_env_from = args.get('transfer_env_from')
        actor_path = f'{actor_path}_{transfer_env_from}_{fname}'
    else:
        actor_path = f'{actor_path}_{env_name}_{fname}'
    learner.load_model(actor_path, device)
    if args.getboolean('test_gt') and args.getboolean('gt_centralized'):
        print('Run as Ground Truch. Tanner')
        fname = 'BM_Tanner'
    elif args.getboolean('test_gt'):
        print('Run as Ground Truch. Saber')
        fname = 'BM_Saber'
    result = None
    n_leader=args.getint('n_leader')
    
    for i in range(n_test_episodes):
        episode_reward = 0
        env.reset()
        state = env.env.get_model_input()
        # state = env.env.env.get_model_input()
        done = False
        steps = 0
        test_loss_mae = []
        test_loss_mse = []
        while not done:
            steps += 1
            u = env.env.controller()
            u_pred = learner.select_action(state).cpu().numpy()
            # if args.getboolean('test_gt'):
            #     test_loss_step_mae = 0
            #     test_loss_step_mse = 0
            # else:
            test_loss_step_mae = MAE(u[n_leader:], u_pred[n_leader:])
            test_loss_step_mse = MSE(u[n_leader:], u_pred[n_leader:])
            test_loss_mae.append(test_loss_step_mae)
            test_loss_mse.append(test_loss_step_mse)
            # if args.getboolean('test_gt')  and np.random.binomial(1, 0.8) > 0:
            if args.getboolean('test_gt'):
                # next_state, reward, done, _ = env.step(env.env.env.controller())
                next_state, reward, done, _ = env.step(u)
            else:
                next_state, reward, done, _ = env.step(u_pred)

            # next_state = env.env.env.get_model_input()
            next_state = env.env.get_model_input()
            episode_reward += reward
            state = next_state

            # making movie slow down test
            # if render and steps % 5 == 0:
            #     env.render(mode="")
        print(f'episode_reward: {episode_reward}')
        # print(env.env.env.get_stats())
        stats = env.env.get_stats()
        # stats['episode_reward'] = episode_reward
        stats['mae'] = np.mean(test_loss_mae)
        stats['mse'] = np.mean(test_loss_mse)
        print(stats)
        if result is None:
            result = {}
            for key in stats.keys():
                result[key] = []
        for key in stats.keys():
            result[key] = result[key]+[stats[key]]
        # env.env.env.plot(i,fname=fname)
        env.env.plot(i, fname=fname)
    df = pd.DataFrame(result)
    print(df.describe())
    df.to_csv(f'./plots/result_test_{fname}_{n_agents}')
    env.close()

    # sys.stdout.close()


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False
    actor_path = 'models/actor'
    # pool = ThreadPool(4)
    results = []
    if config.sections():
        for section_name in config.sections():
            # if not printed_header:
            # print(config[section_name].get('header'))
            # printed_header = True

            # print(section_name)
            print(f'start testing model: {section_name}')
            # results+=[pool.apply_async(test,args=(config[section_name], actor_path))]
            # threading.Thread(target=test,args=(config[section_name], actor_path)).start()
            test(config[section_name], actor_path)
    else:
        test(config[config.default_section], actor_path)

    # for x in results:
    #     x.get()


if __name__ == "__main__":
    main()
