from collections import deque
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable


from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.actor_siji import Actor as Actor
# from learner.actor_siji1 import Actor as ActorNew
from learner.actor_delayedST import Actor as ActorNew

#
# # TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class DAGGER(object):

    # , n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
    def __init__(self, device, args, k=None):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """

        n_s = args.getint('n_states')
        n_a = args.getint('n_actions')
        k = k or args.getint('k')
        hidden_size = args.getint('hidden_size')
        n_layers = args.getint('n_layers') or 2
        gamma = args.getfloat('gamma')

        self.n_agents = args.getint('n_agents')
        self.n_states = n_s
        self.n_actions = n_a
        self.len=args.getint('len')
        self.len_history = args.getint('len')+k-1

        # Device
        self.device = device

        self.enable_alpha = args.getboolean('enable_alpha')
        self.enable_beta = args.getboolean('enable_beta')
        self.enable_gamma = args.getboolean('enable_gamma')

        # Define Networks
        if args.get('new_model'):
            self.actor = ActorNew(self.device, self.len, hidden_size, k, self.enable_alpha,
                                  self.enable_beta, self.enable_gamma, args.getboolean('centralized')).to(self.device)
        else:
            self.actor = Actor(self.device, self.len, hidden_size, self.enable_alpha,
                               self.enable_beta, self.enable_gamma).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(),
                                lr=args.getfloat('actor_lr'))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optim, gamma=0.5)

        # Constants
        self.gamma = gamma

    def select_action(self, state):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        :param state: The current state.
        :param graph_shift_op: History of graph shift operators
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        state_values, state_network, state_values_queue, state_network_queue, obs_values_queue, obs_network_queue, gamma_queue = state
        self_loop = torch.tensor([[i for i in range(self.n_agents)],
                                  [i for i in range(self.n_agents)]])
        mu = self.actor(self_loop, state_values_queue, state_network_queue,
                        obs_values_queue, obs_network_queue, gamma_queue)  # .to(self.device)

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        :only support mini-batch(batch-size is 1)
        """
        # state=batch.state[0]
        # state_values_batch = Variable(torch.cat(tuple([s[0] for s in batch.state]))).to(self.device)
        # state_network_batch = Variable(torch.cat(tuple([s[1] for s in batch.state]))).to(self.device)
        # state_values_queue_batch = Variable(torch.cat(tuple([s[2] for s in batch.state])))
        # state_network_queue_batch = Variable(torch.cat(tuple([s[3] for s in batch.state])))
        # actor_batch = self.actor(state_values_batch, state_network_batch,state_values_queue_batch,state_network_queue_batch)

        n_size = self.n_agents
        batch_size = len(batch.state)
        # a=torch.cat([ s[1]+n_size*i for i,s in enumerate (batch.state)],dim=1)
        # x=torch.cat([s[0] for s in batch.state])
        # u_gamma=torch.cat([s[6] for s in batch.state])
        state_values_queue = deque()
        state_network_queue = deque()
        obs_values_queue = deque()
        obs_network_queue = deque()
        gamma_queue = deque()

        for i in range(self.len_history):
            # x_h=torch.block_diag(*[s[2][i] for j,s in enumerate (batch.state)])
            # batch*n_agents,n_agents,features
            x_h = torch.cat([s[2][i] for j, s in enumerate(batch.state)])
            state_values_queue.append(x_h)
            a_h = torch.cat(
                [s[3][i]+n_size*j for j, s in enumerate(batch.state)], dim=1)
            state_network_queue.append(a_h)
            if self.enable_beta:
                o_h = torch.cat([s[4][i] for s in batch.state])
                obs_values_queue.append(o_h)
                o_a_h = torch.cat(
                    [s[5][i]+n_size*j for j, s in enumerate(batch.state)], dim=1)
                obs_network_queue.append(o_a_h)
            if self.enable_gamma:
                # batch*n_agents,n_agents,features
                g_h = torch.cat([s[6][i] for j, s in enumerate(batch.state)])
                gamma_queue.append(g_h)

        self_loop = torch.tensor([[i for i in range(batch_size*n_size)],
                                  [i for i in range(batch_size*n_size)]])
        optimal_action_batch = torch.cat(batch.action).to(self.device)
        mu = self.actor(self_loop, state_values_queue, state_network_queue,
                        obs_values_queue, obs_network_queue, gamma_queue)

        # Optimize Actor
        self.actor_optim.zero_grad()
        # Loss related to sampled Actor Gradient.
        policy_loss = F.mse_loss(mu, optimal_action_batch) 
        policy_loss.backward()
        self.actor_optim.step()
        # End Optimize Actor

        return policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None):
        """
        Save the Actor Model after training is completed.
        :param env_name: The environment name.
        :param suffix: The optional suffix.
        :param actor_path: The path to save the actor.
        :return: None
        """
        if not os.path.exists('./models/'):
            os.makedirs('./models/')

        if actor_path is None:
            actor_path = "./models/actor_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)

    def load_model(self, actor_path, map_location):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        # print('Loading model from {}'.format(actor_path))
        if actor_path is not None:
            state_dict=torch.load(actor_path, map_location)
            # remove_prefix = 'layers_alpha.1.'
            # state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

            self.actor.load_state_dict(state_dict,strict=False)
            # self.actor.load_state_dict(torch.load(actor_path, map_location))
            # params= [ p.numel() for p in self.actor.parameters() if p.requires_grad ]
            # print(params)
            # print(f'model size: {sum(p.numel() for p in self.actor.parameters() if p.requires_grad)}')
            self.actor.to(self.device)


def train_dagger_siji(env, args, device):
    debug = args.getboolean('debug')
    n_a = args.getint('n_actions')
    n_agents = args.getint('n_agents')
    batch_size = args.getint('batch_size')
    len = args.getint('len')
    if args.getboolean('use_presaved_data'):
        try:
            fileopen = open(f'./presaved_data/{n_agents}_{len}', 'rb')
            memory = pickle.load(fileopen)
            fileopen.close()
        except:
            memory = ReplayBuffer(max_size=args.getint('buffer_size'))
    else:
        memory = ReplayBuffer(max_size=args.getint('buffer_size'))

    learner = DAGGER(device, args)
    env_name = args.get('env')
    fname = args.get('fname')
    if args.getboolean('continue_training'):
        print('load model before training.')
        learner.load_model(f'./models/actor_{env_name}_{fname}', device)

    n_train_episodes = args.getint('n_train_episodes')
    beta_coeff = args.getfloat('beta_coeff')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')

    total_numsteps = 0
    updates = 0
    beta = 1

    stats = {'mean': -1.0 * np.Inf, 'std': 0}
    prev_policy_loss = np.Inf
    prev_reward =-100000
    prev_mse_loss = 90+100000
    for i in range(n_train_episodes):

        beta = max(beta * beta_coeff, 0)
        env.reset()
        state = env.env.get_model_input()

        done = False

        policy_loss_sum = 0
        skip = 0
        j = 0
        while not done:

            optimal_action = env.env.controller()
            if np.random.binomial(1, beta) > 0:
                action = optimal_action
            else:
                action = learner.select_action(state)
                action = action.cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            next_state = env.env.get_model_input()

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            optimal_action = torch.Tensor(optimal_action).to(device)
            # optimal_action = optimal_action.transpose(1, 0)
            # optimal_action = optimal_action.reshape((1, 1, n_a, n_agents))
            j += 1
            if j > skip or i % 5 == 0:
                # if np.sum(np.abs(action)) >0.1 or np.random.binomial(1, 0.5) > 0: # so the train sample has useful output, intead of 0.
                memory.insert(Transition(state, optimal_action,
                              notdone, next_state, reward))
            total_numsteps += 1

            state = next_state

        if memory.curr_size > 4000:
        # if memory.curr_size > 40000:
            total_steps = args.getint('updates_per_step')
            for _ in range(total_steps):
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                policy_loss = learner.gradient_step(batch)
                policy_loss_sum += policy_loss
                updates += 1

            if i % 100 == 0:
                learner.scheduler.step()
            print(
                f'{i} lr: {learner.scheduler.get_last_lr()}, {policy_loss_sum/total_steps}')
            prev_policy_loss = policy_loss_sum

        
        if i % test_interval == 0 and debug:
            test_rewards = []
            mse_list=[]        
            for _ in range(n_test_episodes):
                test_loss = []  
                
                ep_reward = 0
                env.reset()
                state = env.env.get_model_input()
                done = False
                while not done:
                    action = learner.select_action(state)
                    test_loss_step = F.mse_loss(action, torch.tensor(
                        env.env.controller()).to(device)).item()
                    test_loss.append(test_loss_step)
                    next_state, reward, done, _ = env.step(
                        action.cpu().numpy())
                    next_state = env.env.get_model_input()
                    ep_reward += reward
                    state = next_state
                    # env.render()
                test_rewards.append(ep_reward)
                mse_list.append(np.mean(test_loss))
            mean_reward = np.mean(test_rewards)
            std_reward = np.std(test_rewards)
            mean_mse=np.mean(mse_list)
            std_mse = np.std(mse_list)
            env.env.plot(i, fname)

            # learner.save_model(env_name, suffix=args.get('fname'))
            mean_reward_lb = mean_reward-std_reward/2
            mean_mse_ub = mean_mse+std_mse/2
            # save better model
            #  
            # if mean_reward_lb > prev_reward-20:
            #     learner.save_model(env_name, suffix=args.get('fname'))
            #     prev_reward = mean_reward_lb
            if mean_mse_ub < prev_mse_loss+10:
                learner.save_model(env_name, suffix=args.get('fname'))
                prev_mse_loss = mean_mse_ub
            else:
                learner.load_model(
                    f'./models/actor_{env_name}_{fname}', device)

            # if stats['mean'] < mean_reward:
            #     stats['mean'] = mean_reward
            #     stats['std'] = np.std(test_rewards)
            #
            #     if debug and args.get('fname'):  # save the best model
            #         learner.save_model(args.get('env'), suffix=args.get('fname'))

            if debug:
                print(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {:.2f}, std: {:.2f}, test MSE: {:.2f}, mse std: {:.2f}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        std_reward,
                        mean_mse,std_mse))
        sys.stdout.flush()
        if learner.scheduler.get_last_lr()[0] < 1e-7:
            break

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        env.reset()
        state = env.env.get_model_input()

        done = False
        while not done:
            action = learner.select_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            # next_state, reward, done, _ = env.step(env.env.controller())
            next_state = env.env.get_model_input()
            ep_reward += reward
            state = next_state
            # env.render()
        test_rewards.append(ep_reward)
    # env.env.plot(i)
    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)

    # if debug and args.get('fname'):  # save the best model
    #     learner.save_model(args.get('env'), suffix=args.get('fname'))
    if args.getboolean('use_presaved_data'):
        fileopen = open(f'./presaved_data/{n_agents}_{len}', 'wb')
        pickle.dump(memory, fileopen)
        fileopen.close()

    env.close()
    return stats
