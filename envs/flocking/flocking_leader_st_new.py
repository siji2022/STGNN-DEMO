from collections import deque

import gym
import io
from PIL import Image
from gym import spaces, error, utils
from gym.utils import seeding
from envs.flocking.flocking_relative_st import FlockingRelativeSTEnv
from envs.flocking.saber_utils import *
from envs.flocking.utils import constrant_initialization, plot_details, step_helper
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops
import torch

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

class FlockingLeaderSTNewEnv(FlockingRelativeSTEnv):

    def __init__(self):
        super(FlockingLeaderSTNewEnv, self).__init__()
        self.quiver = None

    def params_from_cfg(self, args):
        super(FlockingLeaderSTNewEnv, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents, 2))
        self.n_leaders = args.getint('n_leaders')
        self.mask[0:self.n_leaders] = 0
        

    def step(self, u):
        step_helper(self, u)

        self.curr_step += 1
        self.compute_helpers()

        # update history queue
        self.x_queue.append(self.x)
        self.x_queue.popleft()
        self.state_values_queue.append(torch.clone(self.state_values))
        self.state_values_queue.popleft()
        self.state_network_queue.append(torch.clone(self.state_network))
        self.state_network_queue.popleft()


        # saved in history for plot
        updated_state = np.copy(self.x)
        self.history += [updated_state]
        self.u_history += [self.u]

     
        end_eposode = False
        return (self.state_values, self.state_network), self.instant_cost(), end_eposode, {}
    
    
    def reset(self):
        super(FlockingLeaderSTNewEnv, self).reset()
        v_max = np.min([self.max_velocity, 5])
        self.x[:, 2:4] = np.zeros((self.n_agents, 2))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * self.np_random.uniform(low=-v_max,
                                                                                         high=v_max, size=(1, 2))
        self.x_init=np.copy(self.x)
        return (self.x_features, self.state_network)

    def plot(self, j=0, fname='', dir='plots'):
        plot_details(self, j, fname, dir, plot_leaders=True)

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        super(FlockingLeaderSTNewEnv, self).render(mode)

        X = self.x[0:self.n_leaders, 0]
        Y = self.x[0:self.n_leaders, 1]
        U = self.x[0:self.n_leaders, 2]
        V = self.x[0:self.n_leaders, 3]

        if self.quiver == None:
            self.quiver = self.ax.quiver(X, Y, U, V, color='r')
        else:
            self.quiver.set_offsets(self.x[0:self.n_leaders, 0:2])
            self.quiver.set_UVC(U, V)

#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
        if mode == 'human':
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return self.fig.canvas.draw()
        else:
            # test
            buf = io.BytesIO()
            self.fig.savefig(buf)
            buf.seek(0)
            im = np.asarray(Image.open(buf))
            # buf.close()
            return im
