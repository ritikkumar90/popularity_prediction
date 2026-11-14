"""Configurable heterogeneous GNN (HGT or RGCN backbone) for post popularity prediction.

No config files: every hyperparameter is a keyword argument with a default from the
block below (train.py exposes all of them as --flags).

Node / edge types must match the HeteroData objects built in dataset.py:
  nodes:  user (USER_FEATURE_DIM features), post (POST_FEATURE_DIM features)
  edges:  (user, follows, user), (user, interacts, post),
          (post, rev_interacts, user), (user, followed_by, user)

With default arguments this is the same network as the original PopularityHGT
(parameter names are unchanged, so old hgt_best.pt checkpoints still load).
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
CONV_DROPOUT = 0.0       # dropout after every message-passing layer
EDGE_DROPOUT = 0.0       # randomly drop edges during training (DropEdge)
NORM = "layer"           # layer | batch | none
ACT = "gelu"             # gelu | relu | silu
READOUT = "post"         # post | post+user_mean | post+user_attn
HEAD_LAYERS = 2          # number of Linear layers in the head (>= 1)
HEAD_HIDDEN = None       # head hidden width (None = hidden_channels // 2)
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
    """h_i' = W_self[type(i)] h_i + sum_r mean_{j in N_r(i)} W_r h_j

    num_bases=None -> a full weight matrix per edge type.
    num_bases=B    -> W_r = sum_b a_rb V_b (shared bases): far fewer parameters,
                      which is what keeps sparse relations from overfitting small graphs.
    """

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
        """
        Args:
            metadata: (node_types, edge_types), from HeteroData.metadata() or DEFAULT_METADATA.
            hidden_channels: hidden node embedding size.
            num_heads: attention heads per HGT layer (HGT only; must divide hidden_channels).
            num_layers: number of message passing layers.
            dropout: dropout inside the MLP head (head dropout).
            conv: 'hgt' or 'rgcn'.
            num_bases: RGCN basis decomposition size (None = full per-relation weights).
            input_dropout / conv_dropout / edge_dropout: extra regularisation (0 = off).
            norm / act: normalisation ('layer'|'batch'|'none') and activation.
            residual: residual connection around each message passing layer.
            readout: 'post' | 'post+user_mean' | 'post+user_attn'. The user readouts need
                     batch_dict (see get_batch_dict) in forward().
            head_layers / head_hidden: MLP head depth and width.
            output_act: 'none' | 'softplus' | 'relu' applied to the prediction.
        """
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

        self.num_layers = num_layers
        self.residual = residual
        self.readout = readout
        self.output_act = output_act
        self.input_dropout = input_dropout
        self.conv_dropout = conv_dropout
        self.edge_dropout = edge_dropout
        self.act = _ACTS[act]()

        # 1. Node feature projections & initial normalization (lazy input dims)
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()
        for nt in node_types:
            self.lin_dict[nt] = Linear(-1, hidden_channels)
            self.norm_dict[nt] = _make_norm(norm, hidden_channels)

        # 2. Message passing layers
        self.convs = nn.ModuleList()
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
            self.conv_norms.append(nn.ModuleDict({nt: _make_norm(norm, hidden_channels) for nt in node_types}))

        # 3. Readout
        if readout == "post+user_attn":
            self.attn_gate = nn.Linear(hidden_channels, 1)
        self.embedding_dim = hidden_channels if readout == "post" else 2 * hidden_channels

        # 4. Regression head (named `mlp` so default-config checkpoints stay compatible)
        hh = head_hidden if head_hidden is not None else max(1, hidden_channels // 2)
        layers, d = [], self.embedding_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(d, hh), _ACTS[act](), nn.Dropout(dropout)]
            d = hh
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def encode(self, x_dict, edge_index_dict, batch_dict=None) -> torch.Tensor:
        """Graph embedding of every root post: (num_graphs, embedding_dim). Used for XGBoost stacking."""
        x_dict = {k: v for k, v in x_dict.items() if k in self.lin_dict}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self._edge_type_set}
        edge_index_dict = drop_edges(edge_index_dict, self.edge_dropout, self.training)

        # Step 1: project raw features, normalize, activate
        out = {}
        for nt, x in x_dict.items():
            h = self.act(self.norm_dict[nt](self.lin_dict[nt](x)))
            out[nt] = F.dropout(h, self.input_dropout, self.training)

        # Step 2: message passing layers
        for i in range(self.num_layers):
            new = self.convs[i](out, edge_index_dict)
            nxt = {}
            for nt, prev in out.items():
                h = new.get(nt)
                if h is None:                      # node type received no messages
                    h = prev
                elif self.residual:
                    h = h + prev
                h = self.act(self.conv_norms[i][nt](h))
                nxt[nt] = F.dropout(h, self.conv_dropout, self.training)
            out = nxt

        # Step 3: readout
        post = out["post"]                         # (num_graphs, hidden): one root post per graph
        if self.readout == "post":
            return post
        if batch_dict is None or "user" not in batch_dict or "user" not in out:
            raise ValueError(f"readout={self.readout!r} needs batch_dict with a 'user' entry "
                             "(pass get_batch_dict(batch))")
        idx, u, n = batch_dict["user"], out["user"], post.size(0)
        if self.readout == "post+user_mean":
            pooled = torch.zeros(n, u.size(1), dtype=u.dtype, device=u.device).index_add_(0, idx, u)
            cnt = torch.zeros(n, dtype=u.dtype, device=u.device).index_add_(
                0, idx, torch.ones(idx.size(0), dtype=u.dtype, device=u.device)).clamp_(min=1.0)
            pooled = pooled / cnt.unsqueeze(-1)
        else:                                      # post+user_attn
            w = pyg_softmax(self.attn_gate(u).squeeze(-1), idx, num_nodes=n)
            pooled = torch.zeros(n, u.size(1), dtype=u.dtype, device=u.device).index_add_(
                0, idx, u * w.unsqueeze(-1))
        return torch.cat([post, pooled], dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict=None, return_embedding: bool = False):
        emb = self.encode(x_dict, edge_index_dict, batch_dict)
        pred = self.mlp(emb).squeeze(-1)
        if self.output_act == "softplus":
            pred = F.softplus(pred)
        elif self.output_act == "relu":
            pred = F.relu(pred)
        return (pred, emb) if return_embedding else pred


if __name__ == "__main__":
    # Quick sanity check for every backbone / readout combination (no data needed)
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
    bd = {"user": torch.zeros(10, dtype=torch.long), "post": torch.zeros(1, dtype=torch.long)}

    for conv in CONV_CHOICES:
        for readout in READOUT_CHOICES:
            model = PopularityHGT(data.metadata(), hidden_channels=32, num_heads=2, num_layers=2,
                                  conv=conv, num_bases=2, readout=readout, edge_dropout=0.2,
                                  conv_dropout=0.1, output_act="softplus")
            with torch.no_grad():
                model(data.x_dict, data.edge_index_dict, bd)      # init lazy layers
            model.train()
            out, emb = model(data.x_dict, data.edge_index_dict, bd, return_embedding=True)
            out.sum().backward()
            print(f"{conv:5s} {readout:15s} -> out {tuple(out.shape)} emb {tuple(emb.shape)}")