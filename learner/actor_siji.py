import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.utils as utils

# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class Actor(nn.Module):

    def __init__(self, device, len, hidden_layers, enable_alpha=True, enable_obs=True, enable_gamma=True):
        # assume history size is 3; history result concatenate and pass to next layer
        super(Actor, self).__init__()
        self.device = device
        self.len = len
        self.hidden_layers = hidden_layers
        self.enable_alpha = enable_alpha
        self.enable_obs = enable_obs
        self.enable_gamma = enable_gamma

        layers = []
        layers_obs = []
        layers_gamma = []
        
        layers += [
            geom_nn.SAGEConv(6, hidden_layers, bias=True, normalize=True),
            # geom_nn.SAGEConv(hidden_layers, hidden_layers,
            #                     bias=True, normalize=True),
        ]
        layers_obs += [
            geom_nn.SAGEConv(6, hidden_layers, bias=True, normalize=True),
        ]
        layers_gamma += [
            geom_nn.SAGEConv(2, hidden_layers, bias=True, normalize=True),
        ]
        counter = 0
        if self.enable_alpha:
            self.layers = nn.ModuleList(layers)
            self.lstm_alpha = nn.LSTM(hidden_layers, hidden_layers*2)
            self.linear_alpha = nn.Linear((hidden_layers)*2, hidden_layers)
            counter += 1
        if self.enable_obs:
            self.layers_obs = nn.ModuleList(layers_obs)
            self.lstm_obs = nn.LSTM(hidden_layers, hidden_layers*2)
            self.linear_obs = nn.Linear(hidden_layers*2, hidden_layers)
            counter += 1
        if self.enable_gamma:
            self.layers_gamma = nn.ModuleList(layers_gamma)
            self.lstm_gamma = nn.LSTM(hidden_layers, hidden_layers*2)
            self.linear_gamma = nn.Linear(hidden_layers*2, hidden_layers)
            counter += 1

        self.linear1 = nn.Linear(hidden_layers*counter, 2)

    def forward(self, self_loop, x_queue, a_queue, obs_queue, obs_a_queue, u_gamma_queue):
        x = x_queue[0]
        # self_loop = self_loop.to(self.device)
        x_inputs = []
        obs_inputs = []
        gamma_inputs = []
        for ii in range(self.len):
            i = self.len-ii-1  # reverse order
            x_delayed = x_queue[i].to(self.device)  # 100,100,6
            a_plus = a_queue[i].to(self.device)

            if self.enable_alpha:
                x_i = F.relu(self.layers[0](x_delayed, a_plus))  # 100,100,32
                # x_i = F.relu(self.layers[1](x_i, a_plus))
                # .permute( 1, 0, 2)
                x_i = geom_nn.global_max_pool(
                    x_i, None).reshape(-1, self.hidden_layers)  # 100,32
                # 3,100,32
                x_inputs += [x_i.reshape(1, x.shape[0], self.hidden_layers)]
            if self.enable_obs:
                obs = obs_queue[i].to(self.device)  # 100,6
                # obs_adj = obs_a_queue[i].to(self.device)
                obs_i = F.relu(self.layers_obs[0](obs, a_plus))
                obs_inputs += [obs_i.reshape(1,
                                             x.shape[0], self.hidden_layers)]
            if self.enable_gamma:
                x_gamma = u_gamma_queue[i].to(self.device)
                gamma_i = F.relu(self.layers_gamma[0](x_gamma, a_plus))
                gamma_inputs += [gamma_i.reshape(1,
                                                 x.shape[0], self.hidden_layers)]
        if self.enable_alpha:
            x_inputs = torch.cat(x_inputs)  # 3,100,32
            x_alpha, (h_a, c_a) = self.lstm_alpha(x_inputs)
            # x_alpha = torch.index_select(x_alpha, 0, torch.tensor([0]).to(self.device)).reshape(x.shape[0], -1)
            x_alpha = torch.index_select(x_alpha, 0, torch.tensor(
                [self.len-1]).to(self.device)).reshape(x.shape[0], -1)

            x_alpha = F.dropout(F.relu(self.linear_alpha(
                x_alpha)), p=0.1, training=self.training)
        if self.enable_obs:
            obs_inputs = torch.cat(obs_inputs)
            x_obs, hidden_obs = self.lstm_obs(obs_inputs)  # 3,100,6
            x_obs = torch.index_select(x_obs, 0, torch.tensor(
                [self.len-1]).to(self.device)).reshape(x.shape[0], -1)
            # x_obs = x_obs.permute(1, 0, 2).reshape(x.shape[0], -1)
            x_obs = F.dropout(F.relu(self.linear_obs(x_obs)),
                              p=0.1, training=self.training)

        if self.enable_gamma:
            gamma_inputs = torch.cat( gamma_inputs)
            x_gamma, hidden = self.lstm_gamma(gamma_inputs)
            x_gamma = torch.index_select(x_gamma, 0, torch.tensor(
                [self.len-1]).to(self.device)).reshape(x.shape[0], -1)
            # x_gamma = x_gamma.permute(1, 0, 2).reshape(x.shape[0], -1)
            x_gamma = F.dropout(F.relu(self.linear_obs(
                x_gamma)), p=0.1, training=self.training)

        x = x_alpha  # alpha object always needed for flocking
        if self.enable_obs:
            x = torch.cat([x, x_obs], dim=1)
        if self.enable_gamma:  # alpha is a must
            x = torch.cat([x, x_gamma], dim=1)

        x = self.linear1(x)

        return x
