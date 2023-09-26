import gym
import io
from PIL import Image
from gym import spaces, error, utils
from gym.utils import seeding
from envs.flocking.utils import calc_metrics, calc_reward, constrant_initialization, load_clip_settings, plot_details, potential_grad, reset_history, step_helper
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class FlockingRelativeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 200}
    def __init__(self):

        # config_file = path.join(path.dirname(__file__), "params_flock.cfg")
        # config = configparser.ConfigParser()
        # config.read(config_file)
        # config = config['flock']

        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not
        self.centralized = True

        # number states per agent
        self.nx_system = 4
        # numer of observations per agent
        self.n_features = 6
        # number of actions per agent
        self.nu = 2 

        # default problem parameters
        self.n_agents = 100  # int(config['network_size'])
        self.comm_radius = 0.9  # float(config['comm_radius'])
        self.dt = 0.01  # #float(config['system_dt'])
        self.v_max = 5.0  #  float(config['max_vel_init'])
        self.r_max = 1.0 #10.0  #  float(config['max_rad_init'])
        #self.std_dev = 0.1  #  float(config['std_dev']) * self.dt

        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
        self.v_bias = self.v_max 

        # intitialize state matrices
        self.x = None
        self.u = None
        self.mean_vel = None
        self.init_vel = None

        self.max_accel = 1
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.action_scalar = 1.0

        # used to track the performance
        self.history = []
        self.u_history = []
        self.min_history = []
        self.max_history = []
        self.curr_step=0

        self.seed()

    def params_from_cfg(self, args):

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.gt_centralized = args.getboolean('gt_centralized')
        self.centralized = args.getboolean('centralized')
        self.comm_radius = args.getfloat('comm_radius')
        self.comm_radius2 = self.comm_radius * self.comm_radius

        self.n_agents = args.getint('n_agents')
        self.NUMBER_OF_AGENTS = self.n_agents

        self.dt = args.getfloat('dt')
        self.len = args.getint('len')  # history length

        # config similar to the Olfati-Saber's paper
        self.DISTANCE = self.comm_radius/1.2  # desire distance
        # r, interaction range, distance less than r will be neighbor; cant be larger than distance*1.414
        self.RANGE = self.comm_radius


        # controlls of the ground truth generators
        # control of neighbor distance
        self.C1_alpha = args.getfloat('c_alpha')
        self.C2_alpha = np.sqrt(self.C1_alpha)  # control of velocity
        # control for desired location


        self.max_accel = args.getfloat('max_accel')
        self.max_state_value = args.getfloat(
            'max_state_value')  # max state value
        self.max_velocity = args.getfloat('max_velocity')
        load_clip_settings(self,args)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        step_helper(self, u)

        self.curr_step+=1
        self.compute_helpers()

         # saved in history for plot
        updated_state = np.copy(self.x)
        self.history += [updated_state]
        self.u_history += [self.u]

        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def compute_helpers(self):

        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))
        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        self.max_history += [np.sqrt(np.max(self.r2))]
        np.fill_diagonal(self.r2, np.Inf)
        self.min_history += [np.sqrt(np.min(self.r2))]

        self.adj_mat = (self.r2 < self.comm_radius2).astype(float)

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors 

        self.x_features = np.dstack((self.diff[:, :, 2]/self.comm_radius, np.divide(self.diff[:, :, 0], np.multiply(self.r2, self.r2)*self.comm_radius), np.divide(self.diff[:, :, 0], self.r2),
                          self.diff[:, :, 3]/self.comm_radius, np.divide(self.diff[:, :, 1], np.multiply(self.r2, self.r2)*self.comm_radius), np.divide(self.diff[:, :, 1], self.r2)))


        self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat
        

    def get_stats(self):

       return calc_metrics(self)
    

    def instant_cost(self):  # sum of differences in velocities
        return calc_reward(self)


    def reset(self):
        reset_history(self)
        constrant_initialization(self)
        #self.a_net = self.get_connectivity(self.x)
        self.compute_helpers()
        reset_history(self)
        self.x_init=np.copy(self.x)
        return (self.state_values, self.state_network)

    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """

        if centralized is None:
            centralized = self.gt_centralized

        # TODO use the helper quantities here more? 
        # potentials = np.dstack((self.diff, self.potential_grad(self.diff[:, :, 0], self.r2), self.potential_grad(self.diff[:, :, 1], self.r2)))
         # normalize based on comm_radius
        diff = self.diff/self.comm_radius
        r2 = self.r2/self.comm_radius2
        r2=r2
        # TODO use the helper quantities here more?
        potentials = np.dstack((diff, potential_grad(
            diff[:, :, 0], r2), potential_grad(diff[:, :, 1], r2)))
        potentials = np.nan_to_num(potentials, nan=0.0)  # fill nan with 0
        
        
        if not centralized:
            potentials = potentials * self.adj_mat.reshape(self.n_agents, self.n_agents, 1) 

        p_sum = np.sum(potentials, axis=1).reshape((self.n_agents, self.nx_system + 2))
        controls =  np.hstack(((-  p_sum[:, 4] - p_sum[:, 2]).reshape((-1, 1)), (- p_sum[:, 3] - p_sum[:, 5]).reshape(-1, 1)))
        # controls = np.clip(controls, -10, 10)
        controls = np.clip(controls*self.comm_radius, -
                           self.max_accel, self.max_accel)
        controls = controls 
        return controls


    def plot(self, j=0, fname='',dir='plots'):
        plot_details(self,j,fname,dir)


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1],
                                  'bo', markersize=2)  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            # if self.quiver is None:

            self.quiver = self.ax.quiver(self.x[:, 0], self.x[:, 1], self.x[:, 2], self.x[:, 3],scale=10, scale_units='inches')

            plt.ylim(np.min(self.x[:, 1]) - 5, np.max(self.x[:, 1]) + 5)
            plt.xlim(np.min(self.x[:, 0]) - 5, np.max(self.x[:, 0]) + 5)
            plt.grid(which='both')

            a = gca()
     
            plt.title('GNN Controller {} agents'.format(self.n_agents))
            self.fig = fig
            self.line1 = line1


        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        plt.ylim(np.min(self.x[:, 1]) - 5, np.max(self.x[:, 1]) + 5)
        plt.xlim(np.min(self.x[:, 0]) - 5, np.max(self.x[:, 0]) + 5)
        a = gca()


        self.quiver.set_offsets(self.x[:, 0:2])
        self.quiver.set_UVC(self.x[:, 2], self.x[:, 3])


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
        # return self.fig.canvas.draw()
    


    def close(self):
        pass
 