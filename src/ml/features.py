"""Per-post feature extraction and subgraph sampling at t_obs (no config files).

For every post in split_meta.parquet this builds, using ONLY information visible at t_obs:
  * tabular features  -> tabular_features.parquet (one row per post)
  * a sampled user subgraph (nodes, edges, features) -> shard_*.pt (one dict per post)
  * a small index     -> index.parquet (post_id, shard, t_obs, max_event_time, sizes)

Usage
-----
    from src.data.features import extract_all, audit, SampleConfig
    run_dir = extract_all(".", "flickr", SampleConfig(hops=2, max_subgraph_users=500))
    audit(".", "flickr", run_dir)                       # independent brute-force check

    python -m src.data.features --dataset flickr --hops 2 --max-subgraph-users 500 --audit

Leak protection
---------------
* Labels are never read (only split_meta.parquet, which has none).
* Events: only `time <= t_obs`. Edges: only `created_at <= t_obs` (same rule as the splitter).
* Static graphs (no edge times) are refused unless allow_static_graph=True.
* The cache folder name hashes the sampling settings AND the data (post ids, t0, t_obs and the
  size of events/edges files), so changing the subset or the splits can never reuse old features.
* index.parquet stores the real max event time used; extract_all fails if it exceeds t_obs.
* audit() recomputes features and subgraph edges with plain pandas and compares.

Assumes integer user and post ids (as produced by the Flickr adapter).

Per-user features (user_x columns, see USER_FEATURE_NAMES)
    log_out_degree, log_in_degree   edges visible at t_obs
    is_active                       user adopted (favorited) the post by t_obs
    adoption_time_frac              when they adopted, 0..1 of the observation window (0 if inactive)
    is_author                       the post's author
    hop                             0 = seed (author / adopters), then hop/hops
Per-post features (post_x, see post_feature_names)
    log_observed_count, bin_frac_* (share of observed events in each equal slice of the
    observation window), log_accel (log1p(events in 2nd half) - log1p(events in 1st half)),
    log_author_out_degree, log_author_in_degree
Subgraph edge types: follows (user->user, both in the sample), interacts (adopter->post, attr =
adoption_time_frac), authored (author->post). Local node indices refer to `node_user_ids`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

try:
    import torch
except ImportError:
    torch = None

log = logging.getLogger(__name__)

# ============================================================================ #
# HYPERPARAMETERS / SETTINGS  (change here, or pass your own SampleConfig)
# ============================================================================ #
OUTPUT_DIR = "outputs"   # features go to <base_path>/<OUTPUT_DIR>/<dataset>/samples/<hash>/
N_JOBS = -1              # parallel workers (-1 = all CPUs)


@dataclass
class SampleConfig:
    hops: int = 1                          # neighbourhood expansion depth (0 = only author + adopters)
    max_subgraph_users: int = 200         # cap on users per post subgraph
    n_time_bins: int = 4                   # bins for bin_frac_* (>= 2)
    neighbor_limit_per_node: int | None = None  # max out-neighbours expanded per node (None = all).
                                           # Set e.g. 200 if posts with very popular users are slow.
    neighbor_limit_keep: str = "latest"    # which edges to keep under the limit: "latest" | "earliest"
    seed_order: str = "earliest"           # if > cap users adopted, keep the "earliest" or "latest" adopters
    sample_seed: int = 0                   # seed for random sub-sampling when a hop exceeds the cap
    build_subgraph_edges: bool = True      # also build follows / interacts / authored edges
    shard_size: int = 1000                 # posts per output shard (one parallel task each)
    edge_time_available: bool = True       # filled in automatically from the dataset manifest
    # Bump when sampling logic changes so stale caches are never reused.
    version: int = 3

    def validate(self) -> None:
        if self.hops < 0:
            raise ValueError("hops must be >= 0")
        if self.max_subgraph_users < 1:
            raise ValueError("max_subgraph_users must be >= 1")
        if self.n_time_bins < 2:
            raise ValueError("n_time_bins must be >= 2")
        if self.neighbor_limit_per_node is not None and self.neighbor_limit_per_node < 1:
            raise ValueError("neighbor_limit_per_node must be None or >= 1")
        if self.neighbor_limit_keep not in ("latest", "earliest"):
            raise ValueError("neighbor_limit_keep must be 'latest' or 'earliest'")
        if self.seed_order not in ("earliest", "latest"):
            raise ValueError("seed_order must be 'earliest' or 'latest'")
        if self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")


USER_FEATURE_NAMES = ["log_out_degree", "log_in_degree", "is_active", "adoption_time_frac", "is_author", "hop"]


def to_target(y: float) -> float:
    return float(np.log1p(y))


def post_feature_names(n_time_bins: int) -> list[str]:
    names = ["log_observed_count"]
    names.extend(f"bin_frac_{i}" for i in range(n_time_bins))
    names.extend(["log_accel", "log_author_out_degree", "log_author_in_degree"])
    return names


# --------------------------------------------------------------------------- #
# Graph index: everything is a flat numpy array (CSR), so it can be written to
# disk once and memory-mapped by every worker with zero per-task serialisation.
# --------------------------------------------------------------------------- #

def _require_int(name: str, arr: np.ndarray) -> None:
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integer ids, got dtype {arr.dtype}")


def _to_idx(universe: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Map raw ids -> dense indices in `universe` (sorted). Unknown ids -> -1."""
    ids = np.asarray(ids)
    if len(universe) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    pos = np.searchsorted(universe, ids)
    pos_c = np.minimum(pos, len(universe) - 1)
    ok = universe[pos_c] == ids
    return np.where(ok, pos_c, -1).astype(np.int64)


def _build_csr(src, dst, trank, n_users, stride, use_time, need_nbr):
    """CSR sorted by (src, time). `keys = src*stride + time_rank` is globally
    sorted, so "neighbours of u created at or before t_obs" is a prefix of u's
    slice that one vectorised searchsorted can find for many nodes at once."""
    order = np.lexsort((trank, src)) if use_time else np.argsort(src, kind="stable")
    src_s = src[order]
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(src_s, minlength=n_users), out=indptr[1:])
    nbr = None
    if need_nbr:
        nbr = dst[order].astype(np.int32 if n_users < 2**31 - 1 else np.int64)
    keys = (src_s * stride + trank[order]) if use_time else None
    return indptr, nbr, keys


def build_indexes(posts: pd.DataFrame, events: pd.DataFrame, edges: pd.DataFrame, scfg: SampleConfig):
    log.info("Building CSR graph index...")
    use_time = scfg.edge_time_available

    post_ids = posts["post_id"].to_numpy()
    authors = posts["author"].to_numpy()
    for name, arr in (("post_id", post_ids), ("author", authors), ("events.user", events["user"].to_numpy()),
                      ("edges.source", edges["source"].to_numpy()), ("edges.target", edges["target"].to_numpy())):
        _require_int(name, arr)
    authors = authors.astype(np.int64)

    # Only events of posts we actually process are needed.
    post_universe = np.unique(post_ids)
    events = events[events["post_id"].isin(post_universe)]

    e_src, e_dst = edges["source"].to_numpy(), edges["target"].to_numpy()
    ev_users = events["user"].to_numpy()
    universe = np.unique(np.concatenate([e_src, e_dst, ev_users, authors]))
    n_users = len(universe)

    # ---- events: CSR by post, sorted by time ----
    ev_pidx = np.searchsorted(post_universe, events["post_id"].to_numpy())
    ev_time = events["time"].to_numpy()
    ev_user = _to_idx(universe, ev_users)
    order = np.lexsort((ev_time, ev_pidx))
    ev_pidx, ev_time, ev_user = ev_pidx[order], ev_time[order], ev_user[order]
    ev_ptr = np.searchsorted(ev_pidx, np.arange(len(post_universe) + 1)).astype(np.int64)

    # ---- edges: out-CSR (with neighbours) and in-CSR (degrees only) ----
    src, dst = _to_idx(universe, e_src), _to_idx(universe, e_dst)
    if use_time:
        unique_times, trank = np.unique(edges["created_at"].to_numpy(), return_inverse=True)
        trank = trank.reshape(-1).astype(np.int64)
    else:
        unique_times, trank = np.empty(0, dtype=np.int64), np.zeros(len(src), dtype=np.int64)
    stride = len(unique_times) + 1
    if n_users * stride >= 2**62:
        raise OverflowError("n_users * n_unique_edge_times too large for int64 keys")

    out_indptr, out_nbr, out_keys = _build_csr(src, dst, trank, n_users, stride, use_time, True)
    in_indptr, _, in_keys = _build_csr(dst, src, trank, n_users, stride, use_time, False)

    graph = dict(
        universe=universe, unique_times=unique_times, stride=int(stride),
        ev_ptr=ev_ptr, ev_time=ev_time, ev_user=ev_user,
        out_indptr=out_indptr, out_nbr=out_nbr, out_keys=out_keys,
        in_indptr=in_indptr, in_keys=in_keys,
    )
    post_pidx = np.searchsorted(post_universe, post_ids).astype(np.int64)
    author_idx = _to_idx(universe, authors)
    return graph, post_pidx, author_idx


def _degree(g, side, nodes, tq):
    """Vectorised time-valid degree for an array of dense node indices."""
    indptr, keys = g[f"{side}_indptr"], g[f"{side}_keys"]
    start = indptr[nodes]
    if keys is None:
        return indptr[nodes + 1] - start
    return np.searchsorted(keys, nodes * g["stride"] + tq, side="left") - start


def _out_ranges(g, nodes, tq, limit, keep):
    """[start, end) slice of each node's time-valid out-neighbours, optionally limited."""
    indptr, keys = g["out_indptr"], g["out_keys"]
    start = indptr[nodes]
    if keys is None:
        end = indptr[nodes + 1]
    else:
        end = np.searchsorted(keys, nodes * g["stride"] + tq, side="left")
    if limit is not None:
        if keep == "latest":
            start = np.maximum(start, end - limit)
        else:
            end = np.minimum(end, start + limit)
    return start, end


def _gather(start, end):
    """Flat positions of all slices [start_i, end_i), plus the per-slice counts."""
    counts = end - start
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), counts
    offs = np.cumsum(counts) - counts
    idx = np.arange(total, dtype=np.int64) + np.repeat(start - offs, counts)
    return idx, counts


def build_post_sample(g, p_idx: int, author_idx: int, t0: int, t_obs: int, scfg: SampleConfig):
    """Features and subgraph of one post at t_obs. Returns (record, post_features, extras)."""
    # Events are time-sorted per post -> observed events are a prefix.
    s, e = int(g["ev_ptr"][p_idx]), int(g["ev_ptr"][p_idx + 1])
    obs_count = int(np.searchsorted(g["ev_time"][s:e], t_obs, side="right"))
    obs_users = np.asarray(g["ev_user"][s:s + obs_count])
    obs_times = np.asarray(g["ev_time"][s:s + obs_count])

    # One entry per user, kept in time order (first adoption).
    _, first = np.unique(obs_users, return_index=True)
    first.sort()
    active, act_times = obs_users[first], obs_times[first]

    window = max(int(t_obs) - int(t0), 1)
    tq = int(np.searchsorted(g["unique_times"], t_obs, side="right")) if scfg.edge_time_available else 0
    cap = scfg.max_subgraph_users

    # ---- seeds (hop 0): author first so it can never be truncated away, then adopters ----
    pool = active if scfg.seed_order == "earliest" else active[::-1]
    if author_idx >= 0:
        seeds = np.concatenate(([author_idx], pool[pool != author_idx])).astype(np.int64)
    else:
        seeds = pool.astype(np.int64)
    seeds = seeds[:cap]

    # ---- hop expansion along time-valid out-edges ----
    rng = np.random.default_rng([scfg.sample_seed, int(p_idx)])  # deterministic per post
    levels, hop_vals = [seeds], [0.0]
    seen, frontier = seeds, seeds
    for hop in range(1, scfg.hops + 1):
        remaining = cap - seen.size
        if frontier.size == 0 or remaining <= 0:
            break
        start, end = _out_ranges(g, frontier, tq, scfg.neighbor_limit_per_node, scfg.neighbor_limit_keep)
        idx, _ = _gather(start, end)
        nbrs = np.unique(np.asarray(g["out_nbr"][idx]).astype(np.int64))
        new = np.setdiff1d(nbrs, seen, assume_unique=True)
        if new.size == 0:
            break
        if new.size > remaining:  # random (not id-ordered) choice, so no id bias
            new = np.sort(rng.choice(new, size=remaining, replace=False))
        levels.append(new)
        hop_vals.append(hop / scfg.hops)
        seen = np.concatenate((seen, new))
        frontier = new

    nodes = np.concatenate(levels)
    node_hop = np.repeat(hop_vals, [len(lv) for lv in levels])

    # ---- adoption info per sampled node ----
    is_active = np.zeros(len(nodes), dtype=bool)
    adopt_frac = np.zeros(len(nodes), dtype=np.float64)
    if len(active):
        srt = np.argsort(active)
        su, st = active[srt], act_times[srt]
        pos = np.minimum(np.searchsorted(su, nodes), len(su) - 1)
        is_active = su[pos] == nodes
        adopt_frac = np.where(is_active, np.clip((st[pos] - t0) / window, 0.0, 1.0), 0.0)

    # ---- user features ----
    user_x = np.zeros((len(nodes), len(USER_FEATURE_NAMES)), dtype=np.float32)
    user_x[:, 0] = np.log1p(_degree(g, "out", nodes, tq))
    user_x[:, 1] = np.log1p(_degree(g, "in", nodes, tq))
    user_x[:, 2] = is_active
    user_x[:, 3] = adopt_frac
    user_x[:, 4] = nodes == author_idx if author_idx >= 0 else 0.0
    user_x[:, 5] = node_hop

    # ---- post features ----
    if author_idx >= 0:
        a = np.array([author_idx], dtype=np.int64)
        author_out, author_in = int(_degree(g, "out", a, tq)[0]), int(_degree(g, "in", a, tq)[0])
    else:
        author_out = author_in = 0

    nb = scfg.n_time_bins
    bins = np.clip(((obs_times - t0) * nb) // window, 0, nb - 1).astype(np.int64)
    bin_counts = np.bincount(bins, minlength=nb)
    bin_frac = bin_counts / obs_count if obs_count > 0 else np.zeros(nb)
    accel = np.log1p(bin_counts[nb // 2:].sum()) - np.log1p(bin_counts[:nb // 2].sum())
    feat = np.array(
        [np.log1p(obs_count), *bin_frac, accel, np.log1p(author_out), np.log1p(author_in)],
        dtype=np.float64,
    )

    # ---- subgraph edges (local indices into `nodes`) ----
    follows = np.empty((2, 0), dtype=np.int64)
    interacts = np.empty((2, 0), dtype=np.int64)
    interacts_attr = np.empty((0, 1), dtype=np.float32)
    authored = np.empty((2, 0), dtype=np.int64)
    if scfg.build_subgraph_edges:
        start, end = _out_ranges(g, nodes, tq, scfg.neighbor_limit_per_node, scfg.neighbor_limit_keep)
        idx, counts = _gather(start, end)
        if idx.size:
            src_local = np.repeat(np.arange(len(nodes), dtype=np.int64), counts)
            dst_glob = np.asarray(g["out_nbr"][idx]).astype(np.int64)
            order = np.argsort(nodes)
            sn = nodes[order]
            pos = np.minimum(np.searchsorted(sn, dst_glob), len(sn) - 1)
            ok = sn[pos] == dst_glob
            follows = np.stack([src_local[ok], order[pos[ok]]])
        act_local = np.nonzero(is_active)[0].astype(np.int64)
        interacts = np.stack([act_local, np.zeros_like(act_local)])
        interacts_attr = adopt_frac[act_local, None].astype(np.float32)
        if author_idx >= 0:
            authored = np.array([[0], [0]], dtype=np.int64)  # author is always nodes[0]

    rec = {
        "node_user_ids": g["universe"][nodes],  # back to the original user ids
        "user_x": user_x,
        "post_x": feat[None, :].astype(np.float32),
        "follows": follows,
        "interacts": interacts,
        "interacts_attr": interacts_attr,
        "authored": authored,
    }
    extras = {
        "max_event_time": int(obs_times[-1]) if obs_count else -1,
        "n_nodes": int(len(nodes)),
        "n_follow_edges": int(follows.shape[1]),
    }
    return rec, feat, extras


# --------------------------------------------------------------------------- #
# Shard-level parallelism: one task = one shard. Workers memory-map the graph
# file (shared via the OS page cache) and write their own output, so almost
# nothing is pickled between processes.
# --------------------------------------------------------------------------- #

_GRAPH_CACHE = {}


def _get_graph(path: str):
    if path not in _GRAPH_CACHE:
        _GRAPH_CACHE.clear()
        _GRAPH_CACHE[path] = load(path, mmap_mode="r")
    return _GRAPH_CACHE[path]


def _process_shard(graph_path, run_dir, shard_id, post_ids, p_idx, author_idx, t0, t_obs, scfg):
    run_dir = Path(run_dir)
    pt_path = run_dir / f"shard_{shard_id}.pt"
    meta_path = run_dir / f"shard_{shard_id}.meta.parquet"
    if pt_path.exists() and meta_path.exists():  # resume support
        return shard_id

    g = _get_graph(graph_path)
    names = post_feature_names(scfg.n_time_bins)
    feats = np.empty((len(post_ids), len(names)), dtype=np.float64)
    extras = []
    shard_data = {}
    for i, pid in enumerate(post_ids.tolist()):
        rec, feats[i], ex = build_post_sample(g, int(p_idx[i]), int(author_idx[i]), int(t0[i]), int(t_obs[i]), scfg)
        shard_data[pid] = rec
        extras.append(ex)

    tmp = pt_path.with_name(pt_path.name + ".tmp")
    torch.save(shard_data, tmp)
    os.replace(tmp, pt_path)

    meta = pd.DataFrame(feats, columns=names)
    meta.insert(0, "post_id", post_ids)
    meta.insert(1, "shard", shard_id)
    meta.insert(2, "t_obs", t_obs)
    for key in ("max_event_time", "n_nodes", "n_follow_edges"):
        meta[key] = [x[key] for x in extras]
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    meta.to_parquet(tmp_meta, index=False)
    os.replace(tmp_meta, meta_path)  # written last: marks the shard as complete
    return shard_id


def _edge_time_available(canon_dir: Path) -> bool:
    manifest = canon_dir / "manifest.json"
    if manifest.exists():
        return bool(json.loads(manifest.read_text()).get("edge_time_available", True))
    return True


def extract_all(
    base_path: str | Path,
    dataset_name: str,
    scfg: SampleConfig | None = None,
    *,
    output_dir: str = OUTPUT_DIR,
    n_jobs: int = N_JOBS,
    allow_static_graph: bool = False,
    force: bool = False,
) -> Path:
    """Extract features for every post in split_meta. Returns the folder with the results."""
    if torch is None:
        raise RuntimeError("torch is required to save shards (pip install torch).")

    base = Path(base_path)
    canon_dir = base / "data" / "processed" / dataset_name
    splits_dir = base / "data" / "splits" / dataset_name
    out_dir = base / output_dir / dataset_name / "samples"

    scfg = scfg or SampleConfig()
    scfg = replace(scfg, edge_time_available=_edge_time_available(canon_dir))
    scfg.validate()
    if not scfg.edge_time_available and not allow_static_graph:
        raise ValueError(
            f"{dataset_name} has a static graph (no edge times): degrees and neighbours would include "
            f"edges created after t_obs (leak). Pass allow_static_graph=True to accept this explicitly."
        )

    posts = pd.read_parquet(splits_dir / "split_meta.parquet")
    posts = posts.drop(columns=[c for c in ("label", "growth") if c in posts.columns])  # never needed
    missing = {"post_id", "author", "t0", "t_obs"} - set(posts.columns)
    if missing:
        raise ValueError(f"split_meta.parquet is missing columns {sorted(missing)}")

    # The folder name depends on the settings AND the data, so stale features are never reused.
    h = hashlib.sha256(json.dumps(asdict(scfg), sort_keys=True).encode())
    for col in ("post_id", "t0", "t_obs"):
        h.update(posts[col].to_numpy().tobytes())
    for fname in ("events.parquet", "edges.parquet"):
        h.update(str((canon_dir / fname).stat().st_size).encode())
    run_dir = out_dir / h.hexdigest()[:10]

    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "index.parquet").exists():
        log.info("Features already exist at %s. Skipping extraction.", run_dir)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sample_config.json").write_text(json.dumps(asdict(scfg), indent=2))
    log.info("Extracting features to %s...", run_dir)

    events = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    edges_cols = ["source", "target"] + (["created_at"] if scfg.edge_time_available else [])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=edges_cols)

    graph, post_pidx, author_idx = build_indexes(posts, events, edges, scfg)
    del events, edges

    graph_path = run_dir / "_graph_index.joblib"
    dump(graph, graph_path)  # uncompressed so workers can mmap it
    del graph

    post_ids = posts["post_id"].to_numpy()
    t0 = posts["t0"].to_numpy()
    t_obs = posts["t_obs"].to_numpy()
    n_posts = len(posts)
    n_shards = (n_posts + scfg.shard_size - 1) // scfg.shard_size
    log.info("Processing %d posts in %d shards (n_jobs=%s, cpus=%s).", n_posts, n_shards, n_jobs, os.cpu_count())

    def shard_args(sid):
        a, b = sid * scfg.shard_size, min((sid + 1) * scfg.shard_size, n_posts)
        return (str(graph_path), str(run_dir), sid, post_ids[a:b], post_pidx[a:b],
                author_idx[a:b], t0[a:b], t_obs[a:b], scfg)

    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_shard)(*shard_args(sid)) for sid in range(n_shards)
    )

    names = post_feature_names(scfg.n_time_bins)
    meta_files = [run_dir / f"shard_{s}.meta.parquet" for s in range(n_shards)]
    df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    if not (df["post_id"].to_numpy() == post_ids).all():
        raise RuntimeError("Shard order does not match split_meta order")
    if (df["max_event_time"] > df["t_obs"]).any():
        raise RuntimeError("CRITICAL LEAK: an event after t_obs was used for a feature")

    df[["post_id"] + names].to_parquet(run_dir / "tabular_features.parquet", index=False)
    df[["post_id", "shard", "t_obs", "max_event_time", "n_nodes", "n_follow_edges"]].to_parquet(
        run_dir / "index.parquet", index=False)  # written last: marks the run as complete

    for f in meta_files:
        f.unlink()
    graph_path.unlink()

    log.info("Extraction complete!")
    return run_dir


# --------------------------------------------------------------------------- #
# Audit: recompute with plain pandas (no CSR tricks) and compare
# --------------------------------------------------------------------------- #

def audit(base_path: str | Path, dataset_name: str, run_dir: str | Path, n_posts: int = 30, seed: int = 0) -> dict:
    """Independent check of sampled posts. Raises RuntimeError listing any mismatch."""
    if torch is None:
        raise RuntimeError("torch is required to read shards")
    base, run_dir = Path(base_path), Path(run_dir)
    canon_dir = base / "data" / "processed" / dataset_name
    scfg = SampleConfig(**json.loads((run_dir / "sample_config.json").read_text()))

    posts = pd.read_parquet(base / "data" / "splits" / dataset_name / "split_meta.parquet").set_index("post_id", drop=False)
    index = pd.read_parquet(run_dir / "index.parquet").set_index("post_id")
    tab = pd.read_parquet(run_dir / "tabular_features.parquet").set_index("post_id")
    events = pd.read_parquet(canon_dir / "events.parquet")
    edges = pd.read_parquet(canon_dir / "edges.parquet")
    timed = scfg.edge_time_available

    rng = np.random.default_rng(seed)
    picks = rng.choice(posts.index.to_numpy(), size=min(n_posts, len(posts)), replace=False)
    problems, shards = [], {}

    for pid in picks:
        pm, t_obs = posts.loc[pid], int(posts.loc[pid, "t_obs"])
        tag = f"[post {pid}]"
        ev = events[(events["post_id"] == pid) & (events["time"] <= t_obs)]
        ed_valid = edges[edges["created_at"] <= t_obs] if timed else edges

        if not np.isclose(tab.loc[pid, "log_observed_count"], np.log1p(len(ev)), atol=1e-5):
            problems.append(f"{tag} observed_count feature wrong")
        a = pm["author"]
        for name, count in (("log_author_out_degree", (ed_valid["source"] == a).sum()),
                            ("log_author_in_degree", (ed_valid["target"] == a).sum())):
            if not np.isclose(tab.loc[pid, name], np.log1p(count), atol=1e-5):
                problems.append(f"{tag} {name} wrong")
        if index.loc[pid, "max_event_time"] > t_obs:
            problems.append(f"{tag} used an event after t_obs")

        shard = int(index.loc[pid, "shard"])
        if shard not in shards:
            shards = {shard: torch.load(run_dir / f"shard_{shard}.pt", weights_only=False)}
        rec = shards[shard][int(pid)]
        users = rec["node_user_ids"]
        if len(users) > scfg.max_subgraph_users:
            problems.append(f"{tag} subgraph larger than the cap")
        if len(users) and users[0] != a:
            problems.append(f"{tag} author is not node 0")

        # interacts: exactly the adopters (by t_obs) that are in the sample
        adopters = set(ev["user"].tolist()) & set(users.tolist())
        got = set(users[rec["interacts"][0]].tolist())
        if got != adopters:
            problems.append(f"{tag} interacts users differ from observed adopters")

        if scfg.build_subgraph_edges:
            got_edges = set(zip(users[rec["follows"][0]].tolist(), users[rec["follows"][1]].tolist()))
            in_sample = ed_valid[ed_valid["source"].isin(users) & ed_valid["target"].isin(users)]
            truth = set(zip(in_sample["source"].tolist(), in_sample["target"].tolist()))
            if not got_edges <= truth:
                problems.append(f"{tag} follows contains an edge not visible at t_obs / not in the graph")
            elif scfg.neighbor_limit_per_node is None and got_edges != truth:
                problems.append(f"{tag} follows is missing {len(truth - got_edges)} visible edges")

    if problems:
        raise RuntimeError(f"Audit found {len(problems)} problem(s):\n - " + "\n - ".join(problems[:10]))
    log.info("Audit passed on %d posts", len(picks))
    return {"checked": int(len(picks)), "problems": 0}


def _parse_args() -> argparse.Namespace:
    d = SampleConfig()
    p = argparse.ArgumentParser(description="Extract per-post features and subgraphs at t_obs.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--hops", type=int, default=d.hops)
    p.add_argument("--max-subgraph-users", type=int, default=d.max_subgraph_users)
    p.add_argument("--n-time-bins", type=int, default=d.n_time_bins)
    p.add_argument("--neighbor-limit", type=int, default=d.neighbor_limit_per_node)
    p.add_argument("--shard-size", type=int, default=d.shard_size)
    p.add_argument("--no-subgraph-edges", action="store_true")
    p.add_argument("--n-jobs", type=int, default=N_JOBS)
    p.add_argument("--allow-static-graph", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute even if features exist")
    p.add_argument("--audit", action="store_true", help="verify a sample of posts with plain pandas")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    cfg = SampleConfig(
        hops=a.hops, max_subgraph_users=a.max_subgraph_users, n_time_bins=a.n_time_bins,
        neighbor_limit_per_node=a.neighbor_limit, shard_size=a.shard_size,
        build_subgraph_edges=not a.no_subgraph_edges,
    )
    out = extract_all(a.base_path, a.dataset, cfg, n_jobs=a.n_jobs,
                      allow_static_graph=a.allow_static_graph, force=a.force)
    if a.audit:
        audit(a.base_path, a.dataset, out)