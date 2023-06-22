import pickle
import random
from collections import deque

import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GCNConv

EPSILON = 0.1


def save_model(model, model_name=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    path = "models/"
    if model_name is None:
        path += 'my_gnn'
    else:
        path += model_name
    print('Saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def load_model(model_name, model=None):
    # print('Loading model from {}'.format(actor_path))
    if model_name is not None:
        path = f"models/{model_name}"
        with open(path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location='cpu'))
        return model


def getEdge2DfromMatrix(adj_mat):
    edges = np.array([[]])
    for i, v in enumerate(adj_mat):
        for j, v in enumerate(adj_mat):
            if adj_mat[i, j] > 0.0 or i == j:
                # if  i == j:
                if edges.shape[1] == 0:
                    edges = np.array([[i, j]])
                else:
                    edges = np.append(edges, [[i, j]], axis=0)
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edges


def getWeight2DfromMatrix(adj_mat):
    edges = np.array([])
    for i, v in enumerate(adj_mat):
        for j, v in enumerate(adj_mat):
            if adj_mat[i, j] > 0.0 or i == j:
                if i == j:
                    w = 1
                else:
                    w = adj_mat[i, j]
                if len(edges) == 0:
                    edges = np.array([w])
                else:
                    edges = np.append(edges, [w], axis=0)
    edges = torch.tensor(edges, dtype=torch.float).t().contiguous()
    return edges


def get_adjacency_weight(nodes, dim=2):
    weights = np.array([np.linalg.norm(nodes[i, :dim] - nodes[:, :dim], axis=-1) for i in range(len(nodes))])
    sigma_norm_val = np.sqrt(EPSILON + weights ** 2)
    new_w = np.divide(1, sigma_norm_val)
    new_w = np.exp(new_w) - 1
    return new_w


def get_adjacency_weight_v(nodes, dim=2):
    weights = np.array([np.linalg.norm(nodes[i, :dim] - nodes[:, :dim], axis=-1) for i in range(len(nodes))])
    sigma_norm_val = np.sqrt(EPSILON + weights ** 2)
    new_w = np.divide(1, sigma_norm_val)
    new_w = np.exp(new_w) - 1
    return new_w


def get_adjacency_distance(nodes, dim=2):
    distance = np.array([np.linalg.norm(nodes[i, :dim] - nodes[:, :dim], axis=-1) for i in range(len(nodes))])
    return distance


## true or false array
def get_adjacency_matrix(nodes, r, dim=2):
    return np.array([np.linalg.norm(nodes[i, :dim] - nodes[:, :dim], axis=-1) <= r for i in range(len(nodes))])


## where the new index of node i is , in this new neighborhood
def get_center_index(n_nodes, neighbor_idxs, i):
    count = 0
    for it in range(n_nodes):
        if it < i and neighbor_idxs[it]:
            count += 1
    return count


def gnn_edge_weight_help_global(adj_matrx_true_false, state, dim=2):
    state = np.copy(state)
    # the states if no actions given
    # state[:, 0] = state[:, 0] + state[:, 2] * 0.01
    # state[:, 1] = state[:, 1] + state[:, 3] * 0.01

    neighbors_vel = state[:, dim:dim + dim]  # pick the velocity on x and y
    neighbors_loc = np.copy(state[:, :dim])  # locations

    adj_weight = get_adjacency_weight(neighbors_loc, dim)
    edge = getEdge2DfromMatrix(adj_weight)
    weight = getWeight2DfromMatrix(adj_weight)

    adj_weight_v_ = get_adjacency_weight_v(neighbors_vel, dim)
    edge_v = getEdge2DfromMatrix(adj_weight_v_)
    weight_v = getWeight2DfromMatrix(adj_weight_v_)
    # move loc to center by minus the average loc
    for dim_i in range(dim):
        neighbors_loc[:, dim_i] = neighbors_loc[:, dim_i] - np.average(neighbors_loc[:, dim_i], axis=0)

    distance = get_adjacency_distance(neighbors_loc, dim)
    neighbors = np.hstack((neighbors_loc, neighbors_vel))

    # don't use self distance( of course 0) as min distance to the neighbor
    if distance.shape[0] > 1:
        np.fill_diagonal(distance, np.inf)
    neighbors = np.append(neighbors, np.min(distance, axis=1).reshape(-1, 1), axis=1)

    return edge, weight, neighbors, edge_v, weight_v


def gnn_edge_weight_help_global_i(adj_matrx_true_false, i, state):
    neighbor_idxs = adj_matrx_true_false[i]  # true/false array for this agent
    new_i = get_center_index(state.shape[0], neighbor_idxs, i)
    return new_i


def gnn_edge_weight_help(orig_i, r, state_queue, dim=2):
    neighbors = []
    neighbor_idxs = None
    neighbor_idxs_prev = None

    for i, e in enumerate(state_queue):
        if len(neighbors) < e.shape[0]:
            adj_matrx_true_false = get_adjacency_matrix(e, r, dim)
            #  todo assume virtual target is visble all the time
            adj_matrx_true_false[-1] = True
            if neighbor_idxs is None:
                neighbor_idxs = adj_matrx_true_false[orig_i]
                neighbor_idxs_prev = neighbor_idxs
                neighbors = e[neighbor_idxs]
            else:
                neighbors = np.append(neighbors, e[np.logical_and(adj_matrx_true_false[orig_i],
                                                                  np.logical_xor(neighbor_idxs_prev,
                                                                                 adj_matrx_true_false[orig_i]))],
                                      axis=0)
                neighbor_idxs_prev = np.logical_or(adj_matrx_true_false[orig_i], neighbor_idxs_prev)
            r += r

    edge, weight, neighbors, edge_v, weight_v = gnn_edge_weight_help_global(None, neighbors, dim)
    new_i = get_center_index(state_queue[0].shape[0], neighbor_idxs, orig_i)
    return edge, weight, neighbors, edge_v, weight_v, new_i


def set_seed(seed, env):
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_state_queue(k, state):
    states_queue = deque()  # right is old state, left is new stat
    while len(states_queue) < k:
        states_queue.append(state)
    return states_queue


def batch_helper_init(n_model_features):
    ptr = 0
    batch_idx = []
    batch_edge = torch.tensor([[], []], dtype=torch.long)
    batch_weight = torch.tensor([])
    batch_neighbors = torch.empty(0, n_model_features)
    batch_edge_v = torch.tensor([[], []], dtype=torch.long)
    batch_weight_v = torch.tensor([])
    batch_new_i = []
    start = 0
    return ptr, batch_idx, batch_edge, batch_weight, batch_neighbors, batch_edge_v, batch_weight_v, batch_new_i, start


def batch_add(edge, weight, neighbors, edge_v, weight_v, new_i, ptr, batch_idx, batch_edge, batch_weight,
              batch_neighbors, batch_edge_v, batch_weight_v, batch_new_i, start):
    edge = edge + start
    edge_v = edge_v + start
    batch_idx = batch_idx + [ptr for i in range(neighbors.shape[0])]
    batch_edge = torch.concat((batch_edge, edge), 1)
    batch_weight = torch.concat((batch_weight, weight))
    batch_neighbors = torch.concat((batch_neighbors, torch.tensor(neighbors, dtype=torch.float)), 0)
    batch_edge_v = torch.concat((batch_edge_v, edge_v), 1)
    batch_weight_v = torch.concat((batch_weight_v, weight_v))
    new_i += start
    batch_new_i = batch_new_i + [new_i]
    ptr += 1
    start += neighbors.shape[0]
    return ptr, batch_idx, batch_edge, batch_weight, batch_neighbors, batch_edge_v, batch_weight_v, batch_new_i, start


def load_training_data(n_agents, r, batch_size, it):
    all_data = pickle.load(open(f'./batchdata/train_{n_agents}_{r}_{batch_size}_{it}.pkl', 'rb'))
    batch_neighbors, batch_edge, batch_weight, batch_edge_v, batch_weight_v, batch_new_i, batch_idx, action = \
        all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5], all_data[6], all_data[7]
    return batch_neighbors, batch_edge, batch_weight, batch_edge_v, batch_weight_v, batch_new_i, batch_idx, action


"""
calcualte the performance metric. use the normalized MAE
"""


def eval_metric(pred, target, reduction='mean'):
    mae = torch.abs((pred - target))
    nmae = torch.abs(torch.divide(mae, target))
    if reduction == 'mean':
        mae = torch.mean(mae)
        nmae = torch.mean(nmae)
    else:
        mae = torch.sum(mae)
        nmae = torch.sum(nmae)
    return mae, nmae

# a,b =eval_metric(torch.tensor([[1.,2,-3],[1,2,3]]),torch.tensor([[1.,2,3],[1,2,3]]),'sum')
# print(a)
# print(b)
