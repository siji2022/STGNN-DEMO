from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from envs.flocking.saber_utils import get_adjacency_matrix
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# N - number of drones
# dist - dist between drones on circumference, 0.5 < 0.75 keeps things interesting


def circle_helper(N, dist):
    r = dist * N / 2 / np.pi
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).reshape((N, 1))
    # angles2 = np.pi - angles
    return r * np.hstack((np.cos(angles), np.sin(angles))), -0.5 * np.hstack((np.cos(angles), -0.5 * np.sin(angles)))


def circle(N):
    if N <= 20:
        return circle_helper(N, 0.5)
    else:
        smalln = int(N * 2.0 / 5.0)
        circle1, v1 = circle_helper(smalln, 0.5)
        circle2, v2 = circle_helper(N - smalln, 0.5)
        return np.vstack((circle1, circle2)), np.vstack((v1, v2))


def grid(N, side=5):
    side2 = int(N / side)
    xs = np.arange(0, side) - side / 2.0
    ys = np.arange(0, side2) - side2 / 2.0
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape((N, 1))
    ys = ys.reshape((N, 1))
    return 0.8 * np.hstack((xs, ys))


def twoflocks(N, delta=6, side=None):

    half_n = int(N / 2)
    if side is None:
        grid1 = grid(half_n)
    else:
        grid1 = grid(half_n, side)

    grid2 = grid1.copy() + np.array([0, delta / 2]).reshape((1, 2))
    grid1 = grid1 + np.array([0, -delta / 2]).reshape((1, 2))

    vels1 = np.tile(np.array([0., delta]).reshape((1, 2)), (half_n, 1))
    vels2 = np.tile(np.array([0., -delta]).reshape((1, 2)), (half_n, 1))

    grids = np.vstack((grid1, grid2))
    vels = np.vstack((vels1, vels2))

    return grids, vels


def twoflocks_old(N):
    half_n = int(N / 2)
    grid1 = grid(half_n)
    delta = 6
    grid2 = grid1.copy() + np.array([0, delta / 2]).reshape((1, 2))
    grid1 = grid1 + np.array([0, -delta / 2]).reshape((1, 2))

    vels1 = np.tile(np.array([-1.0, delta]).reshape((1, 2)), (half_n, 1))
    vels2 = np.tile(np.array([1.0, -delta]).reshape((1, 2)), (half_n, 1))

    grids = np.vstack((grid1, grid2))
    velss = 0.1 * np.vstack((vels1, vels2))

    return grids, velss


def parse_settings(fname):
    names = []
    homes = []
    for line in open(fname):
        for n in re.findall(r'\"(.+?)\": {', line):
            if n != 'Vehicles':
                names.append(n)
        p = re.findall(
            r'"X": ([-+]?\d*\.*\d+), "Y": ([-+]?\d*\.*\d+), "Z": ([-+]?\d*\.*\d+)', line)
        if p:
            homes.append(np.array([float(p[0][0]), float(
                p[0][1]), float(p[0][2])]).reshape((1, 3)))
    return names, np.concatenate(homes, axis=0)


def load_clip_settings(self, args):
    self.comm_radius = args.getfloat('comm_radius')
    self.comm_radius2 = self.comm_radius * self.comm_radius
    self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
    self.max_accel = args.getfloat('max_accel')
    self.n_agents = args.getint('n_agents')
    self.r_max = self.r_max * np.sqrt(self.n_agents)

    self.v_max = args.getfloat('v_max')
    self.v_bias = self.v_max
    self.dt = args.getfloat('dt')

    self.max_accel = args.getfloat('max_accel')
    self.max_state_value = args.getfloat(
        'max_state_value')  # max state value
    self.max_velocity = args.getfloat('max_velocity')


def reset_history(self):
    self.fig = None
    self.quiver = None
    self.curr_step = 0
    self.history = []
    self.u_history = []
    self.min_history = []
    self.max_history = []


def constrant_initialization(self):
    x = np.zeros((self.n_agents, self.nx_system))
    degree = 0
    min_dist = 0
    # min_dist_thresh = 0.1  # 0.25
    min_dist_thresh = 0.5  # 0.25
    v_bias = np.min([self.max_velocity, 10])
    v_max = np.min([self.max_velocity, 10])
    # generate an initial configuration with all agents connected,
    # and minimum distance between agents > min_dist_thresh
    while degree < 2 or degree_leader < 0 or min_dist < min_dist_thresh:

        # randomly initialize the location and velocity of all agents
        area = 1+self.n_agents/100
        length = np.sqrt(self.np_random.uniform(
            0, area*self.comm_radius*np.sqrt(self.n_agents), size=(self.n_agents,)))
        angle = np.pi * self.np_random.uniform(0, 2, size=(self.n_agents,))
        x[:, 0] = length * np.cos(angle)
        x[:, 1] = length * np.sin(angle)

        bias = self.np_random.uniform(
            low=-v_bias, high=v_bias, size=(2,))
        x[:, 2] = self.np_random.uniform(
            low=-v_max, high=v_max, size=(self.n_agents,)) + bias[0]
        x[:, 3] = self.np_random.uniform(
            low=-v_max, high=v_max, size=(self.n_agents,)) + bias[1]

        # compute distances between agents
        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        degree_leader = 2
        if hasattr(self, 'n_leaders'):
            a_leader_net = np.sum(np.square(np.transpose(x_loc[self.n_leaders:], (0, 2, 1)) -
                                            np.transpose(x_loc[:self.n_leaders], (2, 0, 1))), axis=2)
            np.fill_diagonal(a_leader_net, np.Inf)
            min_dist = np.sqrt(np.min(np.min(a_leader_net)))
            a_leader_net = a_leader_net < self.comm_radius2
            degree_leader = np.min(np.sum(a_leader_net.astype(int), axis=1))

        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                       np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))
     # keep good initialization
    # print('initialization successfully!')
    self.mean_vel = np.mean(x[:, 2:4], axis=0)
    self.init_vel = x[:, 2:4]
    self.x = x


def potential_grad(pos_diff, r2):
    """
    Computes the gradient of the potential function for flocking proposed in Turner 2003.
    Args:
        pos_diff (): difference in a component of position among all agents
        r2 (): distance squared between agents

    Returns: corresponding component of the gradient of the potential

    """
    grad = -1.0 * np.divide(pos_diff, np.multiply(r2, r2)
                            ) + 1.0 * np.divide(pos_diff, r2)
    grad[r2 > 1.0] = 0
    return grad


def potential(self, r2):
    p = np.reciprocal(r2) + np.log(r2)
    p[r2 > self.comm_radius2] = self.vr
    np.fill_diagonal(p, 0)
    return np.sum(np.sum(p))


def calc_reward(self):  # sum of differences in velocities;
    curr_v_variance = -1.0 * np.sum((np.var(self.x[:, 2:4], axis=0)))/10
    # np.fill_diagonal(self.r2, 0)
    curr_d_variance = -1.0 * \
        np.sum((np.var(np.min(self.r2/self.comm_radius2, axis=1), axis=0)))
    # +1 for each success steps.
    return curr_v_variance+curr_d_variance + 1


def calc_metrics(self):
    stats = {}
    # the average distance to the following taget in the whole trajectory
    history = np.array(self.history)
    if hasattr(self, 'n_leaders'):
        distance_target_all = 0
        velocity_variance = []
        for x in history[200:]:
            velocity = np.var(x[self.n_leaders:, 2]-x[0, 2]) + \
                np.var(x[self.n_leaders:,  3]-x[0, 3])
            velocity_variance.append(velocity)
            for leader_i in range(self.n_leaders):
                distance_target = np.mean(np.sqrt(
                    (x[:, 0]-x[leader_i, 0])**2+(x[:, 1]-x[leader_i, 1])**2))
                distance_target_all += distance_target
        distance_target_all = distance_target_all/len(history[200:])
        distance_target_all = distance_target_all/self.n_leaders
        stats['distance_target'] = distance_target_all
        stats['velocity_var_from_leader'] = np.mean(velocity_variance)
    elif hasattr(self, 'leader_traj'):
        target = np.array(self.leader_traj)
        start = 0
        span = 3000
        distance_target_all = 0
        while start < len(history):
            distance_target_slice = np.sum(np.sqrt(
                (history[start:span, :, 0:1]-target[:, 0])**2+(history[start:span, :, 1:2]-target[:, 1])**2))
            start += span
            distance_target_all += distance_target_slice
        stats['distance_target'] = distance_target_all/len(history)

    collision_th = 0.5
    stats['collision_rate'] = np.sum(np.array(self.min_history[200:])[
                                     :] < collision_th)/len(self.min_history[:])
    stats['steps_succeed'] = len(self.history)
    start = 200
    velocity_variance = []
    for x in self.history[start:]:
        velocity = np.var(x[:, 2])+np.var(x[:,  3])
        velocity_variance.append(velocity)

    stats['velocity_alignment'] = np.mean(velocity_variance)
    stats['velocity_var_last'] = velocity_variance[-1]
    # stats['velocity_alignment'] = np.var(history[200:,2:3])+np.var(history[200:,3:4])

    # stats['vel_diffs'] = np.sqrt(np.sum(
    #     np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2)))/self.n_agents
    # # stats['vel_diffs'] = np.sqrt(np.sum(np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2), axis=1))

    stats['min_dists'] = np.min(np.sqrt(self.r2))
    stats['avg_min_dists'] = np.mean(self.min_history[start:])
    stats['avg_max_dists'] = np.mean(self.max_history[start:])
    try:
        kmeans1_v = KMeans(n_clusters=1, max_iter=1000).fit(self.x[:, 2:])
        kmeans2_v = KMeans(n_clusters=5, max_iter=1000).fit(self.x[:, 2:])
        # kmeans1_p = KMeans(n_clusters=1, max_iter=1000).fit(self.x[:, :2])
        # kmeans2_p = KMeans(n_clusters=5, max_iter=1000).fit(self.x[:, :2])

        print(f'v1:{kmeans1_v.inertia_}, v2:{kmeans2_v.inertia_}')
        if (velocity_variance[-1] > 0.1 and kmeans1_v.inertia_ > 10*kmeans2_v.inertia_):
            stats['converge'] = 0
        else:
            stats['converge'] = 1
    except:
        stats['converge'] = 1
    # find velocity converge time
    # v_converge_time = 10

    # for x in self.history[10:]:
    #     v_converge_time += 1
    #     variance = np.sqrt(
    #         (np.max(x[:, 2:3])-np.min(x[:, 3:4]))**2+(np.max(x[:, 3:4])-np.min(x[:, 3:4]))**2)
    #     # variance = np.max(np.power(x[:, 2:4] - np.mean(x[:, 2:4], axis=0), 2))
    #     # if the max of variance of all agents is less than 5 then the velocity converged.
    #     if variance < 1.0:
    #         break
    # stats['velocity_converge'] = v_converge_time

    # adjacency_matrix = get_adjacency_matrix(self.x, self.comm_radius)
    # adjacency_matrix = np.sum(adjacency_matrix, axis=1)
    # adjacency_matrix_avg = np.mean(adjacency_matrix)
    # stats['neighborhood_size'] = adjacency_matrix_avg

    dist = np.array([np.linalg.norm(self.x[i, :2]-self.x[:, :2], axis=-1)
                    for i in range(self.n_agents)])
    dist_var = np.mean(np.var(dist, axis=0))
    stats['distance_var'] = dist_var

    return stats


def step_helper(self, u):
    assert u.shape == (self.n_agents, self.nu)
    u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
    self.u = u

    # x position
 # for leader situation
    if hasattr(self, 'mask'):
        self.x[:, :2] += self.x[:, 2:]*self.dt
        self.x[:, :2] += self.u*self.dt*self.dt*0.5*self.mask
        self.x[:, 2:] += self.u*self.dt*self.mask
    else:
        self.x[:, :2] += self.x[:, 2:]*self.dt
        self.x[:, :2] += self.u*self.dt*self.dt*0.5
        self.x[:, 2:] += self.u*self.dt

    self.x[:, 2:] = np.clip(
        self.x[:, 2:], -self.max_velocity, self.max_velocity)


def plot_details(self, j=0, fname='', dir='plots', plot_leaders=False):
    history_array = np.array(self.history)
    u_history_array = np.array(self.u_history)
    min_history_array = np.array(self.min_history)
    state = self.x
    plt.clf()
    for i in range(self.n_agents):
        plt.plot(history_array[:, i, 0], history_array[:, i, 1], alpha=0.8)
        # plt.plot(self.Destination_x, self.Destination_y, 'rx', markersize=3)
        if plot_leaders and i < self.n_leaders:
            plt.plot(state[i, 0], state[i, 1], 'ro', markersize=5)
        else:
            plt.plot(state[i, 0], state[i, 1], 'bo', markersize=1)
    plt.quiver(state[:, 0], state[:, 1], state[:, 2],
               state[:, 3], linewidths=0.1, edgecolors='k')
    # for i in range(self.NUMBER_OF_OBS):
    #     phis = np.arange(0, np.pi*2, 0.01)
    #     plt.plot(*xy(self.RK[i], phis, self.yk[i]), c='r', ls='-')
    plt.savefig(f'./{dir}/{fname}_{self.n_agents}_{j}', dpi=150)
    plt.close()

    # plt.clf()
    # for i in range(self.n_agents):
    #     plt.plot(history_array[:, i, 2])
    # plt.savefig(f'./{dir}/{j}_test_vx_{fname}_{self.n_agents}', dpi=150)
    # plt.close()

    # plt.clf()
    # for i in range(self.n_agents):
    #     plt.plot(history_array[:, i, 2])
    # plt.savefig(f'./{dir}/{j}_test_vy_{fname}_{self.n_agents}', dpi=150)
    # plt.close()

#     plt.clf()
#     for i in range(self.n_agents):
#         plt.plot(u_history_array[:, i, 0])
#     plt.savefig(f'./{dir}/{fname}_{self.n_agents}_{j}_test_action_x', dpi=150)
#     plt.close()
# #### only plot first 5, exlucde leaders
#     params={'legend.fontsize':16, 'legend.handlelength':2,
#        'figure.figsize':(8,4),
#        'xtick.labelsize' : 16,
# 'ytick.labelsize' : 16}
#     plt.rcParams.update(params)
#     for i in range(5):
#         plt.plot(u_history_array[:, i+2, 0],label=f'robot_{i}')
#     plt.legend(loc='right')
#     plt.savefig(f'./{dir}/{fname}_{self.n_agents}_{j}_test_action_x', dpi=150)
#     plt.close()

    # plt.clf()
    # for i in range(self.n_agents):
    #     plt.plot(u_history_array[:, i, 1])
    # plt.savefig(f'./{dir}/{j}_{fname}_{self.n_agents}_test_action_y', dpi=150)
    # plt.close()
    # plt.clf()
    # plt.plot(min_history_array)
    # plt.savefig(f'./{dir}/{j}_{fname}_{self.n_agents}_test_min_dist', dpi=150)


def plot_details1(self, j=0, fname='', dir='plots', plot_leaders=False):
    history_array = np.array(self.history)
    u_history_array = np.array(self.u_history)
    min_history_array = np.array(self.min_history)

    state = self.x
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(self.n_agents):
        plt.plot(history_array[:, i, 0], history_array[:,
                 i, 1], linestyle='-.', color='grey', alpha=0.3)
        # t1 = plt.plot(history_array[0, i, 0], history_array[0,
        #                                                     i, 1], 'go', markersize=3)  # start
        # plt.quiver(history_array[0, i, 0], history_array[0, i, 1], history_array[0,
        #            i, 2], history_array[0, i, 3], linewidths=0.01, edgecolors='k')

        if plot_leaders and i < self.n_leaders:
            plt.plot(state[i, 0], state[i, 1], 'ro', markersize=5)
            plt.plot(self.x_init[i, 0],
                     self.x_init[i, 1], 'ro', markersize=5)
        else:
            plt.plot(state[i, 0], state[i, 1], 'bo', markersize=3)
            plt.plot(self.x_init[i, 0],
                     self.x_init[i, 1], 'go', markersize=3)

    plt.quiver(state[:, 0], state[:, 1], state[:, 2],
               state[:, 3], linewidths=0.01, edgecolors='k')
    if plot_leaders:
        plt.quiver(self.x_init[:self.n_leaders, 0], self.x_init[:self.n_leaders, 1], self.x_init[:self.n_leaders, 2],
                   self.x_init[:self.n_leaders, 3], linewidths=0.01, edgecolors='k')
    else:
        plt.quiver(self.x_init[:, 0], self.x_init[:, 1], self.x_init[:, 2],
                   self.x_init[:, 3], linewidths=0.01, edgecolors='k')
    # t1 = plt.plot(self.x_init[: 0], self.x_init[:,1], 'go', markersize=3)  # start

    
    start_legend = mpatches.Patch(color='green', label='step = 0')
    end_legend = mpatches.Patch(color='blue', label=f'step = {self.curr_step}')
    traj_legend = Line2D([0], [0], label='Trajectory',
                         color='grey', linestyle='-.')
    if plot_leaders:
        leader_legend = mpatches.Patch(color='red', label='leader')
        plt.legend(handles=[start_legend, end_legend, traj_legend,leader_legend])
        
    else:
        plt.legend(handles=[start_legend, end_legend, traj_legend])
    # plt.legend([t1,t2],['start','end'],ncol=3)
    plt.xlim(np.min(history_array[:, :, 0])-5,
             np.max(history_array[:, :, 0]) + 5)
    plt.ylim(np.min(history_array[:, :, 1])-5,
             np.max(history_array[:, :, 1]) + 5)
    plt.savefig(f'./{dir}/{fname}_{self.n_agents}_{j}_{self.curr_step}',
                dpi=400, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()
