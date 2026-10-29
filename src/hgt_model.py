import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HGTPopularityPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata):
        super().__init__()
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            # We used feat_dim = 16 in the data loader, so in_channels = 16
            self.lin_dict[node_type] = Linear(16, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # FIX: Removed the invalid `group='sum'` parameter
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
            
        # Final linear layer to predict a single popularity score (scalar) for the photo
        self.out_layer = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Project initial features to hidden dimensions
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x_dict_out[node_type] = F.gelu(self.lin_dict[node_type](x))
            
        # 2. Pass through HGT layers
        for conv in self.convs:
            x_dict_out = conv(x_dict_out, edge_index_dict)
            
        # 3. Predict popularity specifically for the 'photo' nodes
        photo_embeddings = x_dict_out['photo']
        popularity_prediction = F.relu(self.out_layer(photo_embeddings))
        
        return popularity_prediction.squeeze()
