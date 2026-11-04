import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

try:
    import torch
    from torch_geometric.data import Dataset, HeteroData
except ImportError:
    torch, Dataset, HeteroData = None, object, object

from src.config import Config, load_config

log = logging.getLogger(__name__)


@dataclass
class SampleConfig:
    hops: int = 2
    edge_time_available: bool = True
    n_time_bins: int = 4
    max_subgraph_users: int = 1000
    shard_size: int = 1000
    # Bump when sampling logic changes so stale caches are never reused.
    version: int = 2


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
    if "author" in posts.columns:
        authors = posts["author"].fillna(-1).to_numpy().astype(np.int64)
    else:
        authors = np.full(len(posts), -1, dtype=np.int64)

    # Only events of posts we actually process are needed.
    post_universe = np.unique(post_ids)
    events = events[events["post_id"].isin(post_universe)]

    e_src, e_dst = edges["source"].to_numpy(), edges["target"].to_numpy()
    ev_users = events["user"].to_numpy()
    universe = np.unique(np.concatenate([e_src, e_dst, ev_users, authors[authors != -1]]))
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


def _neighbors(g, nodes, tq):
    """All time-valid out-neighbours of `nodes`, concatenated (may repeat)."""
    indptr, keys = g["out_indptr"], g["out_keys"]
    start = indptr[nodes]
    if keys is None:
        end = indptr[nodes + 1]
    else:
        end = np.searchsorted(keys, nodes * g["stride"] + tq, side="left")
    counts = end - start
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.cumsum(counts) - counts
    idx = np.arange(total, dtype=np.int64) + np.repeat(start - offs, counts)
    return g["out_nbr"][idx].astype(np.int64, copy=False)


def build_post_sample(g, p_idx: int, author_idx: int, t_obs, scfg: SampleConfig):
    # Events are time-sorted per post -> observed events are a prefix.
    s, e = g["ev_ptr"][p_idx], g["ev_ptr"][p_idx + 1]
    obs_count = int(np.searchsorted(g["ev_time"][s:e], t_obs, side="right"))
    active = np.unique(g["ev_user"][s:s + obs_count])

    tq = int(np.searchsorted(g["unique_times"], t_obs, side="right")) if scfg.edge_time_available else 0

    # Seeds (hop 0): author first so it can never be truncated away.
    if author_idx >= 0:
        seeds = np.concatenate(([author_idx], active[active != author_idx])).astype(np.int64)
    else:
        seeds = active

    cap = scfg.max_subgraph_users
    levels, hop_vals = [seeds], [0.0]
    seen, frontier = seeds, seeds
    for hop in range(1, scfg.hops + 1):
        # Once the cap is reached, deeper levels would be truncated anyway.
        if frontier.size == 0 or seen.size >= cap:
            break
        nbrs = np.unique(_neighbors(g, frontier, tq))
        new = np.setdiff1d(nbrs, seen, assume_unique=True)
        if new.size == 0:
            break
        levels.append(new)
        hop_vals.append(hop / 2.0)
        seen = np.concatenate((seen, new))
        frontier = new

    # Levels are in priority order (seeds, hop 1, hop 2, ...): truncate by slicing.
    nodes = np.concatenate(levels)[:cap]
    node_hop = np.repeat(hop_vals, [len(lv) for lv in levels])[:cap]

    user_x = np.zeros((len(nodes), 6), dtype=np.float32)
    user_x[:, 0] = np.log1p(_degree(g, "out", nodes, tq))
    user_x[:, 2] = np.isin(nodes, active)
    user_x[:, 5] = node_hop

    if author_idx >= 0:
        a = np.array([author_idx], dtype=np.int64)
        author_out = int(_degree(g, "out", a, tq)[0])
        author_in = int(_degree(g, "in", a, tq)[0])
    else:
        author_out = author_in = 0

    # NOTE: bin_frac_* and log_accel are placeholders (same as the original code).
    feat = np.array(
        [np.log1p(obs_count)]
        + [1.0 if i == 0 else 0.0 for i in range(scfg.n_time_bins)]
        + [-np.log1p(obs_count) if obs_count > 0 else 0.0, np.log1p(author_out), np.log1p(author_in)],
        dtype=np.float64,
    )

    rec = {
        "node_user_ids": g["universe"][nodes],  # back to the original user ids
        "user_x": user_x,
        "post_x": feat[None, :].astype(np.float32),
        "follows": np.empty((2, 0), dtype=np.int64),
        "interacts": np.empty((2, 0), dtype=np.int64),
        "interacts_attr": np.empty((0, 1), dtype=np.float32),
        "authored": np.empty((2, 0), dtype=np.int64),
    }
    return rec, feat


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


def _process_shard(graph_path, run_dir, shard_id, post_ids, p_idx, author_idx, t_obs, scfg):
    run_dir = Path(run_dir)
    pt_path = run_dir / f"shard_{shard_id}.pt"
    meta_path = run_dir / f"shard_{shard_id}.meta.parquet"
    if pt_path.exists() and meta_path.exists():  # resume support
        return shard_id

    g = _get_graph(graph_path)
    names = post_feature_names(scfg.n_time_bins)
    feats = np.empty((len(post_ids), len(names)), dtype=np.float64)
    shard_data = {}
    for i, pid in enumerate(post_ids.tolist()):
        rec, feats[i] = build_post_sample(g, int(p_idx[i]), int(author_idx[i]), t_obs[i], scfg)
        shard_data[pid] = rec

    tmp = pt_path.with_name(pt_path.name + ".tmp")
    torch.save(shard_data, tmp)
    os.replace(tmp, pt_path)

    meta = pd.DataFrame(feats, columns=names)
    meta.insert(0, "post_id", post_ids)
    meta.insert(1, "shard", shard_id)
    meta.insert(2, "max_event_time", t_obs)
    meta.insert(3, "t_obs", t_obs)
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    meta.to_parquet(tmp_meta, index=False)
    os.replace(tmp_meta, meta_path)  # written last: marks the shard as complete
    return shard_id


def extract_all(cfg: Config, n_jobs: int = -1):
    if torch is None:
        raise RuntimeError("torch is required to save shards (pip install torch).")

    dataset_name = cfg.dataset_name
    base_dir = cfg.run.base_path

    canon_dir = base_dir / "data" / "processed" / dataset_name
    splits_dir = base_dir / "data" / "splits" / dataset_name
    out_dir = base_dir / cfg.run.output_dir / dataset_name / "samples"

    scfg = SampleConfig(
        hops=cfg.im.candidate_hops,
        edge_time_available=cfg.active.edge_time_available,
    )

    config_hash = hashlib.sha256(json.dumps(asdict(scfg), sort_keys=True).encode()).hexdigest()[:8]
    run_dir = out_dir / config_hash

    if (run_dir / "index.parquet").exists():
        log.info(f"Features already exist at {run_dir}. Skipping extraction.")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting features to {run_dir}...")

    events = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    edges_cols = ["source", "target"] + (["created_at"] if scfg.edge_time_available else [])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=edges_cols)
    posts = pd.read_parquet(splits_dir / "split_meta.parquet")
    if "label" in posts.columns:
        posts = posts.drop(columns=["label"])

    graph, post_pidx, author_idx = build_indexes(posts, events, edges, scfg)
    del events, edges

    graph_path = run_dir / "_graph_index.joblib"
    dump(graph, graph_path)  # uncompressed so workers can mmap it
    del graph

    post_ids = posts["post_id"].to_numpy()
    t_obs = posts["t_obs"].to_numpy()
    n_posts = len(posts)
    n_shards = (n_posts + scfg.shard_size - 1) // scfg.shard_size
    log.info(f"Processing {n_posts} posts in {n_shards} shards (n_jobs={n_jobs}, cpus={os.cpu_count()}).")

    def shard_args(sid):
        a, b = sid * scfg.shard_size, min((sid + 1) * scfg.shard_size, n_posts)
        return (str(graph_path), str(run_dir), sid, post_ids[a:b], post_pidx[a:b],
                author_idx[a:b], t_obs[a:b], scfg)

    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_shard)(*shard_args(sid)) for sid in range(n_shards)
    )

    names = post_feature_names(scfg.n_time_bins)
    meta_files = [run_dir / f"shard_{s}.meta.parquet" for s in range(n_shards)]
    df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    df[["post_id"] + names].to_parquet(run_dir / "tabular_features.parquet", index=False)
    df[["post_id", "shard", "max_event_time", "t_obs"]].to_parquet(run_dir / "index.parquet", index=False)

    for f in meta_files:
        f.unlink()
    graph_path.unlink()

    log.info("Extraction complete!")
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    c = load_config()
    extract_all(c)