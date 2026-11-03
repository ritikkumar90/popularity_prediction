import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HGTPopularityPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # Lazy initialization (-1) adapts to whatever feature dimensions you feed it
        self.lin_dict = torch.nn.ModuleDict({
            nt: Linear(-1, hidden_channels) for nt in node_types
        })
        
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Linear head (We will clamp non-negative values during evaluation, not during training)
        self.out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Input projection
        x_dict = {k: F.gelu(self.lin_dict[k](x)) for k, x in x_dict.items()}
        
        # HGT Convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
            
        return self.out(x_dict['photo']).squeeze(-1)