from envs.flocking.utils import plot_details, step_helper
import numpy as np
from envs.flocking.flocking_relative import FlockingRelativeEnv
import io
from PIL import Image


class FlockingLeaderEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv, self).__init__()
        self.quiver = None

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv, self).params_from_cfg(args)
        self.n_leaders = args.getint('n_leaders')
        self.mask = np.ones((self.n_agents, 2))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        step_helper(self, u)

        self.curr_step += 1
        self.compute_helpers()

         # saved in history for plot
        updated_state = np.copy(self.x)
        self.history += [updated_state]
        self.u_history += [self.u]

        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingLeaderEnv, self).reset()
        self.quiver = None
        v_max = np.min([self.max_velocity, 5])
        # v_max = self.max_velocity
        self.x[:, 2:4] = np.zeros((self.n_agents, 2))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * self.np_random.uniform(low=-v_max,
                                                                                         high=v_max, size=(1, 2))
        self.x_init=np.copy(self.x)
        return (self.state_values, self.state_network)

    def plot(self, j=0, fname='',dir='plots'):
        plot_details(self,j,fname,dir,plot_leaders=True)
        
    def render(self, mode='human'):
        super(FlockingLeaderEnv, self).render(mode)

        X = self.x[0:self.n_leaders, 0]
        Y = self.x[0:self.n_leaders, 1]
        U = self.x[0:self.n_leaders, 2]
        V = self.x[0:self.n_leaders, 3]

        if self.quiver == None:
            self.quiver = self.ax.quiver(X, Y, U, V, color='r')
        else:
            self.quiver.set_offsets(self.x[0:self.n_leaders, 0:2])
            self.quiver.set_UVC(U, V)

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
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
