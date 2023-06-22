from os import path
import configparser
import numpy as np
import random
import gym
from envs.flocking import *
import torch
import sys
import pandas as pd
from learner.state_with_delay import MultiAgentStateWithDelay
from learner.gnn_dagger import DAGGER
from gym.wrappers import Monitor
import datetime
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

# def test(args, actor_path, render=True):
#     # initialize gym env
#     env_name = args.get('env')
#     fname = args.get('fname')
#     env = Monitor(gym.make(env_name),
#                   f'./video/{env_name}/{fname}/', force=True)
#     # if isinstance(env.env, gym_flock.envs.FlockingRelativeEnv):
#     env.env.env.params_from_cfg(args)

#     sys.stdout=open(f'test_{fname}',"a")
#     print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
#     print(f'start testing: {fname}, {env_name}, {args.get("n_agents")}')

#     # use seed
#     seed = args.getint('seed')
#     env.seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.enabled=False
#     torch.backends.cudnn.deterministic=True

#     # initialize params tuple
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     learner = DAGGER(device, args)
#     n_test_episodes = args.getint('n_test_episodes')

#     if args.getboolean('transfer_test'):
#         transfer_env_from=args.get('transfer_env_from')
#         actor_path = f'{actor_path}_{transfer_env_from}_{fname}'
#     else:
#         actor_path = f'{actor_path}_{env_name}_{fname}'
#         learner.load_model(actor_path, device)
#     if args.getboolean('test_gt'):
#         print('Run as Ground Truch.')

#     for i in range(n_test_episodes):
#         episode_reward = 0
#         state = MultiAgentStateWithDelay(
#             device, args, env.reset(), prev_state=None)
#         done = False
#         while not done:
#             if args.getboolean('test_gt'):
#                 next_state, reward, done, _ = env.step(env.env.env.controller())
#             else:
#                 action = learner.select_action(state)
#                 next_state, reward, done, _ = env.step(action.cpu().numpy())
#             next_state = MultiAgentStateWithDelay(
#                 device, args, next_state, prev_state=state)
#             episode_reward += reward
#             state = next_state
#             if render:
#                 env.render(mode="")
#         print(episode_reward)
#         print(env.env.env.get_stats())
#         env.env.env.plot(i, fname=fname)
#     env.close()

#     sys.stdout.close()


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

    for i in range(n_test_episodes):
        episode_reward = 0
        state = MultiAgentStateWithDelay(
            device, args, env.reset(), prev_state=None)
        done = False
        steps = 0
        test_loss_mae = []
        test_loss_mse = []
        while not done:
            steps += 1
            u = env.env.controller()
            u_pred = learner.select_action(state).cpu().numpy()
            test_loss_step_mae = MAE(u, u_pred)
            test_loss_step_mse = MSE(u, u_pred)
            test_loss_mae.append(test_loss_step_mae)
            test_loss_mse.append(test_loss_step_mse)
            if args.getboolean('test_gt'):
                # next_state, reward, done, _ = env.step(env.env.env.controller())
                next_state, reward, done, _ = env.step(u)
            else:
                next_state, reward, done, _ = env.step(u_pred)

            next_state = MultiAgentStateWithDelay(
                device, args, next_state, prev_state=state)
            episode_reward += reward
            state = next_state

            # making movie slow down test
            # if render and steps % 5 == 0:
            #     env.render(mode="")
        print(f'episode_reward: {episode_reward}')
        # print(env.env.env.get_stats())
        stats = env.env.get_stats()
        stats['episode_reward'] = episode_reward
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

    sys.stdout.close()


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False
    actor_path = 'models/actor'

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            test(config[section_name], actor_path)
    else:
        test(config[config.default_section], actor_path)


if __name__ == "__main__":
    main()
