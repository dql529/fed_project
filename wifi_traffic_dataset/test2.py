import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)

# Assume we have 10 nodes, each with 5 features
num_nodes = 10
num_node_features = 5
num_classes = 2

# Randomly initialize node feature matrix
x = torch.randn(num_nodes, num_node_features)

# Define edges
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
], dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)

# Initialize the GCN
model = GCN(num_node_features, num_classes)

# Forward pass
out = model(data)
print(out)
