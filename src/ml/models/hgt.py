"""Heterogeneous Graph Transformer (HGT) for post popularity prediction.

No config files: every hyperparameter is a keyword argument with a default from the
block below (or pass --flags on the command line via train.py).

Node / edge types must match the HeteroData objects built in dataset.py:
  nodes:  user (USER_FEATURE_DIM features), post (POST_FEATURE_DIM features)
  edges:  (user, follows, user), (user, interacts, post),
          (post, rev_interacts, user), (user, followed_by, user)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

# ============================================================================ #
# HYPERPARAMETERS / DEFAULTS  (train.py overrides these via its CLI flags)
# ============================================================================ #
HIDDEN_CHANNELS = 64     # hidden node embedding size
NUM_HEADS = 4            # attention heads per HGT layer
NUM_LAYERS = 2           # message passing depth
DROPOUT = 0.2            # dropout in the final regression head


class PopularityHGT(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        """
        Args:
            metadata: tuple (node_types, edge_types) from a HeteroData object.
            hidden_channels: dimension of the hidden node embeddings.
            num_heads: attention heads per HGT layer (hidden_channels must be divisible by it).
            num_layers: number of HGT message passing layers.
            dropout: dropout rate in the final MLP head.
        """
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})")
        self.num_layers = num_layers

        # 1. Node feature projections & initial normalization
        # Projects differently sized node inputs into the same hidden dimension.
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            self.norm_dict[node_type] = nn.LayerNorm(hidden_channels)

        # 2. HGT message passing layers
        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
            ))
            # Independent LayerNorm for each node type per layer
            self.conv_norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels) for node_type in metadata[0]
            }))

        # 3. Final regression head (MLP applied only to 'post' nodes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # Step 1: project raw features and normalize
        out_dict = {}
        for node_type, x in x_dict.items():
            x_proj = self.lin_dict[node_type](x)
            x_norm = self.norm_dict[node_type](x_proj)
            out_dict[node_type] = F.gelu(x_norm)

        # Step 2: pass through HGT layers with residual connections
        for i in range(self.num_layers):
            conv = self.convs[i]
            norm_dict = self.conv_norms[i]

            # Cache previous state for the residual connection
            identity_dict = {k: v for k, v in out_dict.items()}

            # Message passing
            out_dict = conv(out_dict, edge_index_dict)

            # Apply residual, normalization, and activation
            for node_type, x in out_dict.items():
                x_res = x + identity_dict[node_type]  # residual connection
                x_norm = norm_dict[node_type](x_res)  # LayerNorm
                out_dict[node_type] = F.gelu(x_norm)  # activation

        # Step 3: extract the root post embeddings and predict.
        # Batched subgraphs merge into one graph, so out_dict['post'] has shape
        # (Batch_Size, Hidden_Channels).
        post_embeddings = out_dict["post"]

        # Squeeze to (Batch_Size,) to match the target labels
        predictions = self.mlp(post_embeddings).squeeze(-1)

        return predictions


if __name__ == "__main__":
    # Quick sanity check for the architecture (no data needed)
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["user"].x = torch.randn(10, 6)
    data["post"].x = torch.randn(1, 8)
    data["user", "follows", "user"].edge_index = torch.randint(0, 10, (2, 15))
    data["user", "interacts", "post"].edge_index = torch.stack(
        [torch.randint(0, 10, (5,)), torch.zeros(5, dtype=torch.long)]
    )
    data["post", "rev_interacts", "user"].edge_index = torch.stack(
        [torch.zeros(5, dtype=torch.long), torch.randint(0, 10, (5,))]
    )
    data["user", "followed_by", "user"].edge_index = torch.randint(0, 10, (2, 15))

    model = PopularityHGT(metadata=data.metadata(), hidden_channels=32, num_heads=2, num_layers=2)

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        print("Mock output shape:", out.shape)  # should be torch.Size([1])