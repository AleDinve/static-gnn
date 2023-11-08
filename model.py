from torch_geometric.nn import global_add_pool, GINConv
import torch
from torch.nn import Linear, Sequential,  BatchNorm1d, Tanh
from torch_geometric.nn import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

class GIN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GINConv(Sequential(Linear(input_size, hidden_size),
                       BatchNorm1d(hidden_size),Tanh(),
                       Linear(hidden_size, hidden_size), Tanh()),eps=0.3).to(device))
            input_size = hidden_size
        self.linear = Linear(hidden_size, output_size).to(device)
        self.mlp = MLP([hidden_size, 2*hidden_size, output_size]).to(device)
    
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.linear(x)
        return torch.sigmoid(x)
        
                        
