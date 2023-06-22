import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, global_mean_pool, global_max_pool


class GNN(nn.Module):
    def __init__(self, device, n_agents, input_features, action_space):
        super(GNN, self).__init__()
        self.device=device
        self.n_agents = n_agents
        scale = 46
        input_features = input_features-2
        self.graph1 = GraphConv(input_features, input_features * scale, aggr='mean')
        # self.graph2 = GraphConv(input_features * scale, input_features * scale, aggr='mean', add_self_loops=False)
        self.linear2 = nn.Linear(2, 16)
        self.linear2_1 = nn.Linear(16, 16)
        self.linear1 = nn.Linear((input_features) * scale + 16, (input_features) * scale + 4)
        # self.linear3 = nn.Linear((input_features + 1) * scale, 128)
        self.linear4 = nn.Linear((input_features) * scale + 4, action_space)

    def forward(self, data):
        device=self.device
        x = data[0][:, :4].to(device)
        edge = data[1].to(device)
        weight = data[2].to(device)
        center_idx = data[3]
        min_distance = data[0][center_idx, 4:]
        min_distance = min_distance.reshape(1, -1).to(device)
        # min_distance = torch.index_select(x, 0, torch.tensor(center_idx))
        # min_distance = torch.index_select(min_distance, 1, torch.tensor(5))
        x = F.relu(self.graph1(x, edge, weight))
        # x = F.relu(self.graph2(x, edge, weight))
        x1 = global_max_pool(x, None)
        # print(x.shape)
        x2 = torch.index_select(x, 0, torch.tensor([center_idx]).to(device))
        # x1 = torch.flatten(x1)
        # x2 = torch.flatten(x2)

        x = torch.sub(x2, x1)
        min_distance = F.relu(self.linear2(min_distance))
        min_distance = F.relu(self.linear2_1(min_distance))
        x = torch.concat((x, min_distance), axis=1)
        x = F.relu(self.linear1(x))
        # x = self.linear3(x)
        x = self.linear4(x)
        return x
