import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.utils as utils
import torch_scatter

# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class Actor(nn.Module):

    def __init__(self, device, len, hidden_layers, k=3, enable_alpha=True, enable_obs=True, enable_gamma=True, central=False):
        # assume history size is 3; history result concatenate and pass to next layer
        super(Actor, self).__init__()
        self.device = device
        self.len = len
        self.k = k
        self.hidden_layers = hidden_layers
        self.enable_alpha = enable_alpha
        self.enable_obs = enable_obs
        self.enable_gamma = enable_gamma
        self.central = central
        layers_alpha = []

        if central:
            layers_alpha += [
                geom_nn.SAGEConv(6, hidden_layers, bias=True, normalize=True),
                geom_nn.SAGEConv(hidden_layers, hidden_layers,
                                 bias=True, normalize=True),
                geom_nn.SAGEConv(hidden_layers, hidden_layers,
                                 bias=True, normalize=True),
                geom_nn.SAGEConv(hidden_layers, hidden_layers,
                                 bias=True, normalize=True),

            ]
        else:
            layers_alpha += [
                geom_nn.SAGEConv(6, hidden_layers, bias=True,
                                normalize=True),

            ]
        # for K spaning
        layers_linear_mapping = [
            nn.LSTM(hidden_layers, hidden_layers),
        ]

        counter = 0
        if self.enable_obs and self.enable_gamma:
            counter = 8

        self.layers_alpha = nn.ModuleList(layers_alpha)

        # for L aggregation
        self.lstm = nn.LSTM(hidden_layers+counter, hidden_layers+counter, 1)
        self.layers_linear_mapping = nn.ModuleList(layers_linear_mapping)
        self.linear = nn.Linear(
            (hidden_layers)+counter, hidden_layers+counter)
        self.linear1 = nn.Linear(hidden_layers+counter, 2)

    def forward(self, self_loop, x_queue, a_queue, obs_queue, obs_a_queue, u_gamma_queue):
        x = x_queue[0]
        self_loop = self_loop.to(self.device)
        x_inputs = []
        obs_inputs = []
        gamma_inputs = []
        counter = 1
        delayed_states = torch.zeros(
            (self.k, x.shape[0],  self.hidden_layers)).to(self.device)

        for ii in range(self.len+self.k-1):
            i = ii
            x_i = x_queue[i].to(self.device)  # 100,100,6
            a_i = a_queue[i].to(torch.float).to(self.device)
         

            a_i_orig = a_i/a_i.sum(dim=1)  # used for delayed adj matmul
            a_i_orig[a_i_orig != a_i_orig] = 0  # set NaN to 0
            a_i = utils.dense_to_sparse(a_i)[0]  # used for GCN

            # for delayed state spatial expansion
            temp_delayed_state = torch.zeros(
                (self.k, x.shape[0], self.hidden_layers)).to(self.device)

            delayed_memory_lenth = self.k-1 if self.k-1 < i else i

            if delayed_memory_lenth > 0:
                dense_a_i = a_i_orig
                merged_delay_states = torch.matmul(
                    dense_a_i, delayed_states[-delayed_memory_lenth:, :])

                temp_delayed_state[-delayed_memory_lenth -
                                   1:-1] = merged_delay_states

           

            if self.central:
                 # curr stage aggregation
                x_i = F.relu(self.layers_alpha[0](x_i, a_i))  # 100,100,32
                # central can have multiple layers of gnn without information leaker to N-hop since it's central
                x_i = F.relu(self.layers_alpha[1](x_i, a_i))  # 100,32
                x_i = F.relu(self.layers_alpha[2](x_i, a_i))  # 100,32
                x_i = F.relu(self.layers_alpha[3](x_i, a_i))  # 100,32
                
            else:
                 # curr stage aggregation
                x_i = F.relu(self.layers_alpha[0](x_i, a_i))  # 100,100,32
      

            temp_delayed_state[self.k-1, :] = x_i
            delayed_states = temp_delayed_state

            if i >= self.k-1:  # now have spatial info
                x_i = temp_delayed_state
                if self.k > 1:
                    x_i, (h, c) = self.layers_linear_mapping[0](x_i)  # k,100,32

                x_i = x_i[-1, :]  # delayed state
                counter=0
                if self.enable_obs and self.enable_gamma:
                    counter = 8
                    x_obs = obs_queue[i].to(self.device)  # 100,6
                    x_gamma = u_gamma_queue[i].to(self.device)
                    x_i = torch.cat((x_i, x_obs, x_gamma), axis=1)  # 3,100,32

                    # x_i = F.relu(self.layers_all[0](x_i, a_i))  # 100,32

                x_inputs += [x_i.reshape(1, x.shape[0],
                                         self.hidden_layers+counter)]

        x_inputs = torch.cat(x_inputs)  # 3,100,32
        if self.len > 1:
            x_inputs, (h_a, c_a) = self.lstm(x_inputs)
            
        x_inputs = x_inputs[-1, :]

        x = F.dropout(F.relu(self.linear(x_inputs)),
                      p=0, training=self.training)
        x = self.linear1(x)

        return x
