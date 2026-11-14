"""
Robust heterogeneous R-GCN for popularity regression (Flickr / Digg / Higgs).

Design goals
------------
* Works with any node/edge types in the shared schema, including relations or node
  types that are missing from some batches.
* Does not lose cascade-size information: mean aggregation hides "how many users
  interacted", so log-degree features and per-graph pooled counts are added.
* Stable on small datasets (Digg): LayerNorm, residual connections, dropout, DropEdge.
* Exposes the penultimate embedding so it can be stacked with XGBoost.
"""
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import degree, dropout_edge

EdgeType = Tuple[str, str, str]

# --- Default Configurations ---
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_EDGE_DROPOUT = 0.1
DEFAULT_TARGET_NODE = "post"


def infer_schema(batches: Iterable) -> dict:
    """
    Build the model schema as the UNION over several batches.

    Reading metadata from a single batch is fragile: a relation that is absent from
    that batch would be silently ignored for the whole run.
    """
    node_types: List[str] = []
    edge_types: List[EdgeType] = []
    in_channels: Dict[str, int] = {}
    for batch in batches:
        for nt in batch.node_types:
            if nt not in node_types:
                node_types.append(nt)
            x = getattr(batch[nt], "x", None)
            if x is not None and nt not in in_channels:
                in_channels[nt] = int(x.size(-1))
        for et in batch.edge_types:
            et = tuple(et)
            if et not in edge_types:
                edge_types.append(et)
    if not node_types:
        raise ValueError("infer_schema received no batches.")
    return {"node_types": node_types, "edge_types": edge_types, "in_channels": in_channels}


class PopularityRGCN(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],
        in_channels: Dict[str, int],
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        edge_dropout: float = DEFAULT_EDGE_DROPOUT,
        use_degree_features: bool = True,
        use_pooling: bool = True,
        target_node: str = DEFAULT_TARGET_NODE,
    ):
        """
        Args:
            metadata: (node_types, edge_types), e.g. from `infer_schema`.
            in_channels: raw feature dimension per node type (no lazy init needed).
            hidden_channels: hidden embedding size.
            num_layers: number of relational message-passing layers.
            dropout: dropout used in encoders, layers and the head.
            edge_dropout: DropEdge probability applied during training only.
            use_degree_features: append log1p(in-degree per relation) to node features.
            use_pooling: add per-graph mean/max pooling and log node counts of all
                non-target node types (e.g. users) to the readout.
            target_node: node type holding the prediction target (one per graph).
        """
        super().__init__()
        node_types = list(metadata[0])
        edge_types = [tuple(et) for et in metadata[1]]

        if target_node not in node_types:
            raise ValueError(f"target_node '{target_node}' not in node types {node_types}")
        missing = [nt for nt in node_types if nt not in in_channels]
        if missing:
            raise ValueError(f"Missing in_channels for node types: {missing}")

        self.node_types = node_types
        self.edge_types = edge_types
        self.target_node = target_node
        self.hidden_channels = hidden_channels
        self.edge_dropout = edge_dropout
        self.use_degree_features = use_degree_features
        self.incoming = {nt: [et for et in edge_types if et[2] == nt] for nt in node_types}
        self.pool_types = sorted(nt for nt in node_types if nt != target_node) if use_pooling else []

        # 1. Per-node-type input encoders (own scale per type and per dataset)
        self.encoders = nn.ModuleDict()
        for nt in node_types:
            d_in = in_channels[nt] + (len(self.incoming[nt]) if use_degree_features else 0)
            self.encoders[nt] = nn.Sequential(
                nn.Linear(d_in, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # 2. Relational message passing: one operator per relation, summed across
        #    relations (mean within a relation + sum over relations = R-GCN style).
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {et: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for et in edge_types},
                    aggr="sum",
                )
            )
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types}))
        self.dropout = nn.Dropout(dropout)

        # 3. Readout + regression head
        readout_dim = hidden_channels + len(self.pool_types) * (2 * hidden_channels + 1)
        self.head_hidden = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_channels),
            nn.GELU(),
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head_out = nn.Linear(hidden_channels, 1)

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def init_output_bias(self, value: float) -> None:
        """Start from predicting the training mean of the (log1p) target."""
        self.head_out.bias.fill_(float(value))

    def _encode(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[EdgeType, Tensor]) -> Dict[str, Tensor]:
        h = {}
        for nt, enc in self.encoders.items():
            x = torch.nan_to_num(x_dict[nt].float())
            if self.use_degree_features:
                cols = [x]
                for et in self.incoming[nt]:
                    ei = edge_index_dict.get(et)
                    if ei is None or ei.numel() == 0:
                        d = x.new_zeros(x.size(0))
                    else:
                        d = degree(ei[1], num_nodes=x.size(0), dtype=x.dtype)
                    cols.append(torch.log1p(d).unsqueeze(-1))
                x = torch.cat(cols, dim=-1)
            h[nt] = enc(x)
        return h

    def _drop_edges(self, edge_index_dict: Dict[EdgeType, Tensor]) -> Dict[EdgeType, Tensor]:
        if not self.training or self.edge_dropout <= 0:
            return edge_index_dict
        return {
            et: dropout_edge(ei, p=self.edge_dropout, training=True)[0]
            for et, ei in edge_index_dict.items()
        }

    def _readout(self, h: Dict[str, Tensor], batch_dict: Optional[Dict[str, Tensor]]) -> Tensor:
        root = h[self.target_node]
        n_graphs = root.size(0)
        if batch_dict is None and n_graphs > 1 and self.pool_types:
            raise ValueError("batch_dict is required to pool when a batch holds more than one graph.")
        parts = [root]
        for nt in self.pool_types:
            hn = h[nt]
            if batch_dict is not None and nt in batch_dict:
                vec = batch_dict[nt]
            else:
                vec = torch.zeros(hn.size(0), dtype=torch.long, device=hn.device)
            parts.append(global_mean_pool(hn, vec, size=n_graphs))
            parts.append(global_max_pool(hn, vec, size=n_graphs))
            counts = torch.bincount(vec, minlength=n_graphs).to(hn.dtype)
            parts.append(torch.log1p(counts).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        return_embedding: bool = False,
    ):
        # Degree features use the full graph; DropEdge only affects message passing.
        h = self._encode(x_dict, edge_index_dict)
        mp_edges = self._drop_edges(edge_index_dict)

        for conv, norms in zip(self.convs, self.norms):
            out = conv(h, mp_edges)
            # Node types that received no messages keep their state (residual path),
            # instead of disappearing from the dict as in plain HeteroConv.
            h = {
                nt: (x + self.dropout(F.gelu(norms[nt](out[nt]))) if nt in out else x)
                for nt, x in h.items()
            }

        emb = self.head_hidden(self._readout(h, batch_dict))
        pred = self.head_out(self.head_dropout(emb)).squeeze(-1)
        return (pred, emb) if return_embedding else pred


if __name__ == "__main__":
    from torch_geometric.data import Batch, HeteroData

    parser = argparse.ArgumentParser(description="Smoke test for the R-GCN model")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    args = parser.parse_args()

    def make_graph(n_users: int, n_inter: int, with_follows: bool = True) -> HeteroData:
        d = HeteroData()
        d["user"].x = torch.randn(n_users, 6)
        d["post"].x = torch.randn(1, 8)
        d["post"].y = torch.rand(1) * 5
        src = torch.randint(0, n_users, (n_inter,))
        d["user", "interacts", "post"].edge_index = torch.stack([src, torch.zeros(n_inter, dtype=torch.long)])
        d["post", "rev_interacts", "user"].edge_index = torch.stack([torch.zeros(n_inter, dtype=torch.long), src])
        if with_follows:
            d["user", "follows", "user"].edge_index = torch.randint(0, n_users, (2, 3 * n_users))
        return d

    # Second graph has no 'follows' edges: the union schema must still work.
    batch = Batch.from_data_list([make_graph(10, 5), make_graph(4, 2, with_follows=False), make_graph(7, 3)])
    schema = infer_schema([batch])
    model = PopularityRGCN(
        (schema["node_types"], schema["edge_types"]),
        schema["in_channels"],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    )
    model.eval()
    with torch.no_grad():
        pred, emb = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, return_embedding=True)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print("pred shape:", tuple(pred.shape), "| embedding shape:", tuple(emb.shape))
    print("pred:", pred)