import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
df_train = pd.read_csv('C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\train.csv', sep=' ')
df_test = pd.read_csv('C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\test.csv', sep=' ')

train_features = df_train.iloc[:,:18]
train_labels = df_train.iloc[:,18]
test_features = df_test.iloc[:,:18]
test_labels = df_test.iloc[:,18]

# local_data = torch.tensor(train_features.values.reshape(-1, 18), dtype=torch.float32)
# local_targets = torch.tensor(train_labels.values.reshape(-1,1), dtype=torch.float32)


# Define the adjacency matrix for the feature computational dependencies
adjacency_matrix = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
], dtype=torch.float)
# Convert the adjacency matrix to edge index format
edges = adjacency_matrix.nonzero(as_tuple=False).t()
data = Data(x=torch.tensor(train_features.values.reshape(-1, 18), dtype=torch.float32), edge_index=edges, y=torch.tensor(train_labels.values.reshape(-1,1), dtype=torch.float32))
data_test = Data(x=torch.tensor(test_features.values.reshape(-1, 18), dtype=torch.float32), edge_index=edges, y=torch.tensor(test_labels.values.reshape(-1,1), dtype=torch.float32))
#data_test = Data(x=torch.tensor(test_features.values.reshape(-1, 18), dtype=torch.float32), edge_index=edges, y=torch.tensor(test_labels.values.reshape(-1,1), dtype=torch.float32))

# Define a simple GNN with GCN layers
class Net18(torch.nn.Module):
    def __init__(self):
        super(Net18, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 1) #Binary classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

print(train_features)

