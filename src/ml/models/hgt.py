import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class PopularityHGT(nn.Module):
    def __init__(self, metadata, hidden_channels: int = 64, num_heads: int = 4, num_layers: int = 2):
        """
        Heterogeneous Graph Transformer for Post Popularity Prediction.
        
        Args:
            metadata: Tuple of (node_types, edge_types) from the HeteroData object.
            hidden_channels: Dimension of the hidden node embeddings.
            num_heads: Number of attention heads for the HGT layers.
            num_layers: Number of HGT message passing layers.
        """
        super().__init__()
        self.num_layers = num_layers
        
        # 1. Node Feature Projections & Initial Normalization
        # Projects differently sized node inputs into the same hidden dimension.
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            self.norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            
        # 2. HGT Message Passing Layers
        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads
            ))
            # Independent LayerNorm for each node type per layer
            layer_norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels) for node_type in metadata[0]
            })
            self.conv_norms.append(layer_norm_dict)
            
        # 3. Final Regression Head (MLP applied only to 'post' nodes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        # Step 1: Project raw features and normalize
        out_dict = {}
        for node_type, x in x_dict.items():
            x_proj = self.lin_dict[node_type](x)
            x_norm = self.norm_dict[node_type](x_proj)
            out_dict[node_type] = F.gelu(x_norm)
            
        # Step 2: Pass through HGT layers with Residual Connections
        for i in range(self.num_layers):
            conv = self.convs[i]
            norm_dict = self.conv_norms[i]
            
            # Cache previous state for the residual connection
            identity_dict = {k: v for k, v in out_dict.items()}
            
            # Message passing
            out_dict = conv(out_dict, edge_index_dict)
            
            # Apply residual, normalization, and activation
            for node_type, x in out_dict.items():
                x_res = x + identity_dict[node_type]  # Residual connection
                x_norm = norm_dict[node_type](x_res)  # LayerNorm
                out_dict[node_type] = F.gelu(x_norm)  # Activation
            
        # Step 3: Extract the root post embeddings and predict
        # Since batches merge subgraphs, out_dict['post'] will have shape (Batch_Size, Hidden_Channels)
        post_embeddings = out_dict['post']
        
        # Squeeze output to shape (Batch_Size,) to match target labels safely
        predictions = self.mlp(post_embeddings).squeeze(-1)
        
        return predictions

if __name__ == "__main__":
    # Quick sanity check for the architecture
    from torch_geometric.data import HeteroData
    
    # Mock data to match our dataset
    data = HeteroData()
    data['user'].x = torch.randn(10, 6)
    data['post'].x = torch.randn(1, 8)
    data['user', 'follows', 'user'].edge_index = torch.randint(0, 10, (2, 15))
    data['user', 'interacts', 'post'].edge_index = torch.stack([torch.randint(0, 10, (5,)), torch.zeros(5, dtype=torch.long)])
    data['post', 'rev_interacts', 'user'].edge_index = torch.stack([torch.zeros(5, dtype=torch.long), torch.randint(0, 10, (5,))])
    data['user', 'followed_by', 'user'].edge_index = torch.randint(0, 10, (2, 15))
    
    model = PopularityHGT(metadata=data.metadata(), hidden_channels=32, num_heads=2, num_layers=2)
    
    # Forward pass
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        print("Mock output shape:", out.shape) # Should be torch.Size([1])