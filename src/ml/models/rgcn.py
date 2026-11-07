import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

# --- Default Configurations ---
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2

class PopularityRGCN(nn.Module):
    def __init__(self, metadata, hidden_channels: int = DEFAULT_HIDDEN_CHANNELS, num_layers: int = DEFAULT_NUM_LAYERS):
        """
        Relational Graph Convolutional Network (R-GCN) Baseline for Heterogeneous Data.
        
        Args:
            metadata: Tuple of (node_types, edge_types) from the HeteroData object.
            hidden_channels: Dimension of the hidden node embeddings.
            num_layers: Number of relational message passing layers.
        """
        super().__init__()
        
        # 1. Node Feature Projections
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            
        # 2. Relational Message Passing Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                # Assign a distinct SAGEConv operator to each relation (edge type)
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(-1, -1), 
                    out_channels=hidden_channels
                )
            
            # HeteroConv aggregates the outputs from the distinct relation convolutions
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
        # 3. Final Regression Head (Applied only to 'post' nodes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        # Step 1: Project raw features
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = F.gelu(self.lin_dict[node_type](x))
            
        # Step 2: Pass through R-GCN layers
        for conv in self.convs:
            out_dict = conv(out_dict, edge_index_dict)
            out_dict = {node_type: F.gelu(x) for node_type, x in out_dict.items()}
            
        # Step 3: Extract the root post embeddings and predict
        post_embeddings = out_dict['post']
        predictions = self.mlp(post_embeddings).squeeze(-1)
        
        return predictions

if __name__ == "__main__":
    from torch_geometric.data import HeteroData
    
    parser = argparse.ArgumentParser(description="Test R-GCN Model Initialization")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of GNN layers")
    args = parser.parse_args()

    # Mock data validation
    data = HeteroData()
    data['user'].x = torch.randn(10, 6)
    data['post'].x = torch.randn(1, 8)
    data['user', 'follows', 'user'].edge_index = torch.randint(0, 10, (2, 15))
    data['user', 'interacts', 'post'].edge_index = torch.stack([torch.randint(0, 10, (5,)), torch.zeros(5, dtype=torch.long)])
    data['post', 'rev_interacts', 'user'].edge_index = torch.stack([torch.zeros(5, dtype=torch.long), torch.randint(0, 10, (5,))])
    data['user', 'followed_by', 'user'].edge_index = torch.randint(0, 10, (2, 15))
    
    model = PopularityRGCN(
        metadata=data.metadata(), 
        hidden_channels=args.hidden_channels, 
        num_layers=args.num_layers
    )
    
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        print(f"Model initialized with hidden_channels={args.hidden_channels}, num_layers={args.num_layers}")
        print("Mock R-GCN output shape:", out.shape)
        print("Mock R-GCN output values:", out)