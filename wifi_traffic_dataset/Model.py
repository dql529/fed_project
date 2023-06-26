import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Dataset import n_nodes
import pandas as pd

# 加载服务器的训练和测试数据
server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")

# 对于每个节点，加载训练和测试数据
node_train = []
node_test = []
for i in range(1, n_nodes + 1):
    node_train.append(torch.load(f"data_object/node_train_{i}.pt"))
    node_test.append(torch.load(f"data_object/node_test_{i}.pt"))


# Define a simple GNN with GCN layers
class Net18(torch.nn.Module):
    def __init__(self, num_output_features):
        super(Net18, self).__init__()
        self.conv1 = GCNConv(server_train.num_node_features, 16)
        self.conv2 = GCNConv(16, num_output_features)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
