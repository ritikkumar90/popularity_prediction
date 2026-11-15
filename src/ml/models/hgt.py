"""Configurable heterogeneous GNN (HGT or RGCN backbone) for post popularity prediction.

Fixes HGT constant-output collapse via:
  1. Batch-level input standardization (preserves per-post scale differences that LayerNorm erased)
  2. Pre-Norm residual message-passing blocks with degree-scaled aggregation
  3. Direct skip-connection of raw `post` features into the regression head
  4. Residual growth anchoring on `log_obs_count` (post.x[:, 0]) so predictions never collapse
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import softmax as pyg_softmax

# ============================================================================ #
# HYPERPARAMETERS / DEFAULTS  (train.py overrides these via its CLI flags)
# ============================================================================ #
HIDDEN_CHANNELS = 64     # hidden node embedding size
NUM_HEADS = 4            # attention heads per HGT layer (HGT only)
NUM_LAYERS = 2           # message passing depth
DROPOUT = 0.2            # dropout inside the regression head (head dropout)

CONV = "hgt"             # backbone: hgt | rgcn
NUM_BASES = None         # RGCN basis decomposition (None = one full weight per relation)
INPUT_DROPOUT = 0.0      # dropout after the input projection
CONV_DROPOUT = 0.1       # dropout after every message-passing layer
EDGE_DROPOUT = 0.0       # randomly drop edges during training (DropEdge)
NORM = "batch"           # layer | batch | none  (batch preserves cross-post magnitude!)
ACT = "gelu"             # gelu | relu | silu
READOUT = "post+user_attn"  # post | post+user_mean | post+user_attn
HEAD_LAYERS = 2          # number of Linear layers in the head (>= 1)
HEAD_HIDDEN = None       # head hidden width (None = hidden_channels)
OUTPUT_ACT = "none"      # none | softplus | relu  (targets are log1p(count) >= 0)

CONV_CHOICES = ("hgt", "rgcn")
NORM_CHOICES = ("layer", "batch", "none")
ACT_CHOICES = ("gelu", "relu", "silu")
READOUT_CHOICES = ("post", "post+user_mean", "post+user_attn")
OUTPUT_ACT_CHOICES = ("none", "softplus", "relu")

# Fixed schema shared by flickr / digg / higgs (use with train.py --schema fixed).
NODE_TYPES = ["user", "post"]
EDGE_TYPES = [
    ("user", "follows", "user"),
    ("user", "interacts", "post"),
    ("post", "rev_interacts", "user"),
    ("user", "followed_by", "user"),
]
DEFAULT_METADATA = (NODE_TYPES, EDGE_TYPES)

_ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


# ============================================================================ #
# Helpers
# ============================================================================ #
def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def get_batch_dict(batch) -> dict:
    """{node_type: graph-id vector} for a batched HeteroData (needed by user readouts)."""
    out = {}
    for nt in batch.node_types:
        b = getattr(batch[nt], "batch", None)
        if b is not None:
            out[nt] = b
    return out


def drop_edges(edge_index_dict: dict, p: float, training: bool) -> dict:
    """DropEdge: independently drop each edge with probability p (training only)."""
    if not training or p <= 0.0:
        return edge_index_dict
    out = {}
    for et, ei in edge_index_dict.items():
        if ei.size(1) == 0:
            out[et] = ei
            continue
        keep = torch.rand(ei.size(1), device=ei.device) >= p
        out[et] = ei[:, keep] if bool(keep.any()) else ei
    return out


# ============================================================================ #
# RGCN layer for heterogeneous graphs (one weight per relation, optional bases)
# ============================================================================ #
class HeteroRGCNConv(nn.Module):
    """h_i' = W_self[type(i)] h_i + sum_r mean_{j in N_r(i)} W_r h_j"""

    def __init__(self, hidden_channels: int, metadata, num_bases: int | None = None):
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = [tuple(et) for et in edge_types]
        num_rel = len(self.edge_types)
        self.self_lin = nn.ModuleDict({nt: nn.Linear(hidden_channels, hidden_channels) for nt in node_types})

        if num_bases is None or num_bases >= num_rel:
            self.weight = nn.Parameter(torch.empty(num_rel, hidden_channels, hidden_channels))
            self.bases, self.comp = None, None
            for w in self.weight:
                nn.init.xavier_uniform_(w)
        else:
            self.weight = None
            self.bases = nn.Parameter(torch.empty(num_bases, hidden_channels, hidden_channels))
            self.comp = nn.Parameter(torch.randn(num_rel, num_bases) / math.sqrt(num_bases))
            for w in self.bases:
                nn.init.xavier_uniform_(w)

    def _rel_weights(self) -> torch.Tensor:
        if self.bases is None:
            return self.weight
        return torch.einsum("rb,bio->rio", self.comp, self.bases)

    def forward(self, x_dict, edge_index_dict):
        W = self._rel_weights()
        out = {nt: self.self_lin[nt](x) for nt, x in x_dict.items()}
        for r, et in enumerate(self.edge_types):
            ei = edge_index_dict.get(et)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _, dst_t = et
            if src_t not in x_dict or dst_t not in x_dict:
                continue
            src, dst = ei[0], ei[1]
            msg = (x_dict[src_t] @ W[r])[src]
            n_dst = x_dict[dst_t].size(0)
            agg = torch.zeros(n_dst, msg.size(-1), dtype=msg.dtype, device=msg.device).index_add_(0, dst, msg)
            deg = torch.zeros(n_dst, dtype=msg.dtype, device=msg.device).index_add_(
                0, dst, torch.ones(dst.size(0), dtype=msg.dtype, device=msg.device)
            ).clamp_(min=1.0)
            out[dst_t] = out[dst_t] + agg / deg.unsqueeze(-1)
        return out


# ============================================================================ #
# Model
# ============================================================================ #
class PopularityHGT(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        *,
        conv: str = CONV,
        num_bases: int | None = NUM_BASES,
        input_dropout: float = INPUT_DROPOUT,
        conv_dropout: float = CONV_DROPOUT,
        edge_dropout: float = EDGE_DROPOUT,
        norm: str = NORM,
        act: str = ACT,
        residual: bool = True,
        readout: str = READOUT,
        head_layers: int = HEAD_LAYERS,
        head_hidden: int | None = HEAD_HIDDEN,
        output_act: str = OUTPUT_ACT,
    ):
        super().__init__()
        if conv not in CONV_CHOICES:
            raise ValueError(f"conv must be one of {CONV_CHOICES}, got {conv!r}")
        if norm not in NORM_CHOICES:
            raise ValueError(f"norm must be one of {NORM_CHOICES}, got {norm!r}")
        if act not in ACT_CHOICES:
            raise ValueError(f"act must be one of {ACT_CHOICES}, got {act!r}")
        if readout not in READOUT_CHOICES:
            raise ValueError(f"readout must be one of {READOUT_CHOICES}, got {readout!r}")
        if output_act not in OUTPUT_ACT_CHOICES:
            raise ValueError(f"output_act must be one of {OUTPUT_ACT_CHOICES}, got {output_act!r}")
        if head_layers < 1:
            raise ValueError("head_layers must be >= 1")
        if conv == "hgt" and hidden_channels % num_heads != 0:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})")

        node_types = list(metadata[0])
        edge_types = [tuple(et) for et in metadata[1]]
        metadata = (node_types, edge_types)
        self._edge_type_set = set(edge_types)

        self.conv_type = conv
        self.num_layers = num_layers
        self.residual = residual
        self.readout = readout
        self.output_act = output_act
        self.input_dropout = input_dropout
        self.conv_dropout = conv_dropout
        self.edge_dropout = edge_dropout
        self.act = _ACTS[act]()

        # 1. Input feature BatchNorm + Linear projections (lazy input dims)
        self.in_norm_dict = nn.ModuleDict()
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()
        for nt in node_types:
            self.in_norm_dict[nt] = nn.LazyBatchNorm1d()
            self.lin_dict[nt] = Linear(-1, hidden_channels)
            self.norm_dict[nt] = _make_norm(norm, hidden_channels)

        # Direct projection of raw post tabular features so GNN message passing cannot wash them out
        self.post_skip_lin = Linear(-1, hidden_channels)

        # 2. Message passing layers + self-projection (keeps node identity strong in HGT)
        self.convs = nn.ModuleList()
        self.self_lins = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "hgt":
                self.convs.append(HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                ))
            else:
                self.convs.append(HeteroRGCNConv(hidden_channels, metadata, num_bases=num_bases))
            self.self_lins.append(nn.ModuleDict({nt: nn.Linear(hidden_channels, hidden_channels) for nt in node_types}))
            self.conv_norms.append(nn.ModuleDict({nt: _make_norm(norm, hidden_channels) for nt in node_types}))

        # 3. Readout (graph-level representation)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            _ACTS[act](),
            nn.Linear(hidden_channels // 2, 1),
        )
        # Always combine post GNN embedding + post raw skip + user readout
        base_dim = hidden_channels if readout == "post" else 3 * hidden_channels
        self.embedding_dim = base_dim

        # 4. Regression head
        hh = head_hidden if head_hidden is not None else max(32, hidden_channels)
        layers, d = [], self.embedding_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(d, hh), _ACTS[act](), nn.Dropout(dropout)]
            d = hh
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_bn(bn_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        if x.size(0) == 1 and bn_layer.training:
            # BatchNorm1d requires >1 sample in training mode; use eval stats for size-1 edge case
            bn_layer.eval()
            out = bn_layer(x)
            bn_layer.train()
            return out
        return bn_layer(x)

    def encode(self, x_dict, edge_index_dict, batch_dict=None) -> torch.Tensor:
        """Graph embedding of every root post: (num_graphs, embedding_dim)."""
        x_dict = {k: v for k, v in x_dict.items() if k in self.lin_dict}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self._edge_type_set}
        edge_index_dict = drop_edges(edge_index_dict, self.edge_dropout, self.training)

        # Standardize raw input features across the batch (critical for unscaled degrees/counts)
        normed_x = {nt: self._safe_bn(self.in_norm_dict[nt], x) for nt, x in x_dict.items()}
        post_raw_skip = self.act(self.post_skip_lin(normed_x["post"]))

        # Step 1: project raw features
        out = {}
        for nt, x in normed_x.items():
            h = self.lin_dict[nt](x)
            if x.size(0) > 1 or not isinstance(self.norm_dict[nt], nn.BatchNorm1d):
                h = self.norm_dict[nt](h)
            h = self.act(h)
            out[nt] = F.dropout(h, self.input_dropout, self.training)

        # Step 2: message passing layers
        for i in range(self.num_layers):
            new = self.convs[i](out, edge_index_dict)
            nxt = {}
            for nt, prev in out.items():
                h_self = self.self_lins[i][nt](prev)
                h_msg = new.get(nt)
                if h_msg is None:
                    h = h_self
                elif self.residual:
                    h = prev + h_self + F.dropout(h_msg, self.conv_dropout, self.training)
                else:
                    h = h_self + F.dropout(h_msg, self.conv_dropout, self.training)

                if h.size(0) > 1 or not isinstance(self.conv_norms[i][nt], nn.BatchNorm1d):
                    h = self.conv_norms[i][nt](h)
                nxt[nt] = self.act(h)
            out = nxt

        # Step 3: readout
        post = out["post"]  # (num_graphs, hidden)
        if self.readout == "post":
            return post + post_raw_skip

        n = post.size(0)
        if batch_dict is not None and "user" in batch_dict and "user" in out and out["user"].size(0) > 0:
            idx, u = batch_dict["user"], out["user"]
            if self.readout == "post+user_mean":
                pooled = torch.zeros(n, u.size(1), dtype=u.dtype, device=u.device).index_add_(0, idx, u)
                cnt = torch.zeros(n, dtype=u.dtype, device=u.device).index_add_(
                    0, idx, torch.ones(idx.size(0), dtype=u.dtype, device=u.device)
                ).clamp_(min=1.0)
                pooled = pooled / cnt.unsqueeze(-1)
            else:  # post+user_attn: attention-weighted sum + log-count scaling
                w = pyg_softmax(self.attn_gate(u).squeeze(-1), idx, num_nodes=n)
                pooled = torch.zeros(n, u.size(1), dtype=u.dtype, device=u.device).index_add_(
                    0, idx, u * w.unsqueeze(-1)
                )
                cnt = torch.zeros(n, dtype=u.dtype, device=u.device).index_add_(
                    0, idx, torch.ones(idx.size(0), dtype=u.dtype, device=u.device)
                )
                pooled = pooled * torch.log1p(cnt).unsqueeze(-1)
        else:
            pooled = torch.zeros_like(post)

        return torch.cat([post, post_raw_skip, pooled], dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict=None, return_embedding: bool = False):
        emb = self.encode(x_dict, edge_index_dict, batch_dict)
        delta = self.mlp(emb).squeeze(-1)

        # Anchor prediction on `log_obs_count` (first column of post.x) so the network
        # predicts log-growth above observed votes instead of collapsing to the dataset mean.
        log_obs = x_dict["post"][:, 0]
        if self.output_act == "softplus":
            pred = log_obs + F.softplus(delta)
        elif self.output_act == "relu":
            pred = log_obs + F.relu(delta)
        else:
            pred = log_obs + delta

        return (pred, emb) if return_embedding else pred