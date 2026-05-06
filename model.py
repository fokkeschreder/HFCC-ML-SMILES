import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EPRPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3):
        super(EPRPredictor, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        out = self.out(x)
        return out.squeeze(-1)
