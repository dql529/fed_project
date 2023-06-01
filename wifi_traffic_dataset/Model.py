import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size =output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device  # 获取输入数据的设备
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 将隐藏状态移至相同的设备
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 将隐藏状态移至相同的设备

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out




# Define the edges and nodes of the hypergraph
edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
nodes = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# Create a PyTorch Geometric data object
data = Data(x=nodes, edge_index=edges)

# Define a simple GNN with GCN layers
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)

# Initialize and train the GNN (this is just a placeholder example, in practice you would use a proper training loop)
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
