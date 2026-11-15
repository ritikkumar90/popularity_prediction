"""Per-story features and subgraphs at t_obs for Digg (feeds XGBoost, R-GCN and HGT).

For every story in data/splits/digg/split_meta.parquet this builds, using ONLY what is known at
t_obs (votes with time <= t_obs, friendships with created_at <= t_obs):

  tabular_features.parquet   one row per story, flat features        -> XGBoost
  shard_*.pt                 one dict per story: node features +     -> R-GCN / HGT
                             edges of a sampled user subgraph
  index.parquet              post_id, shard, sizes (written last = run is complete)
  sample_config.json         the settings of this run

Usage
-----
    from src.ml.features import extract_all, audit, leak_test, SampleConfig
    run_dir = extract_all(".", "digg", SampleConfig(hops=1))
    audit(".", "digg", run_dir)          # recompute everything with plain pandas and compare
    leak_test(".", "digg")               # delete the future from the data, features must not change

    python -m src.ml.features --dataset digg --hops 1 --audit --leak-test

Edge direction
--------------
Edges are the canonical ones: source -> target means information flows from source to target
(target sees source's votes). With the Digg adapter default, source = the followed user (the
"friend") and target = the follower. So, in the names below:
    audience  = out-degree = number of users who can see this user's votes
    following = in-degree  = number of users whose votes this user sees
If the direction turns out to be the other way round, flip it once in the adapter
(DiggAdapter(reverse_edges=True)); nothing here has to change.

Leak protection
---------------
* Labels are never read.
* Votes: time <= t_obs. Friendships: created_at <= t_obs. Vote history of a user ("prior votes")
  and an author's earlier stories use only time < t0 of the story.
* The cache folder name hashes the settings and the data, so changed splits never reuse old files.
* extract_all checks that every story's observed vote count equals the splitter's observed_count.
* audit() re-implements every feature with pandas; leak_test() rebuilds a story from data that was
  cut off at t_obs and requires identical output.

Per-story features (post_x / tabular_features.parquet, see post_feature_names)
    log_obs_count, bin_frac_*      volume at t_obs and how the votes are spread over the window
    log_accel, log_last_bin        2nd-half vs 1st-half growth; votes in the last bin
    log_author_*                   submitter: audience, following, earlier votes, earlier stories
    log_audience_*, in_graph_frac  audience of the early voters (sum / max / mean / distinct non-voters)
    exposed_frac, log_exposure_edges  how many early voters follow someone who voted before them
    mean/max_log_prior_votes       how active the early voters were before the story existed
    hour_sin, hour_cos, is_weekend submission time (UTC)
Per-user features (user_x, see USER_FEATURE_NAMES)
    log_audience, log_following, log_prior_votes, is_active, adoption_time_frac, is_author, hop,
    log_exposers (number of voted-before accounts the user follows = how exposed they were)
Subgraph (local node indices refer to rec["node_user_ids"], node 0 is the submitter)
    nodes     early voters (capped), then audience users reached through friendship edges
              (ranked by how many early voters they follow, so the most exposed come first)
    follows   user -> user, both in the sample, direction = information flow
    interacts voter -> story (attr = adoption_time_frac), authored: submitter -> story
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

log = logging.getLogger(__name__)

# ============================================================================ #
# SETTINGS  (change here, or pass your own SampleConfig)
# ============================================================================ #
OUTPUT_DIR = "outputs"   # features go to <base_path>/<OUTPUT_DIR>/<dataset>/samples/<hash>/
DAY = 86400


@dataclass
class SampleConfig:
    hops: int = 2                    # 0 = only voters (+ submitter); 1 = + their audience; 2 = + audience of audience
    max_adopter_nodes: int = 300     # cap on voter nodes per story (submitter always kept)
    max_neighbor_nodes: int = 500    # cap on audience nodes PER HOP
    n_time_bins: int = 4             # bins for bin_frac_* (>= 2)
    seed_order: str = "earliest"     # if more voters than the cap: keep the "earliest" or "latest"
    sample_seed: int = 0             # tie-break seed (stories are seeded by post_id, so it is stable)
    build_subgraph_edges: bool = True
    shard_size: int = 1000           # stories per shard file
    version: int = 5                 # bump when the logic changes, so stale caches are never reused

    def validate(self) -> None:
        if not 0 <= self.hops <= 2:
            raise ValueError("hops must be 0, 1 or 2")
        if self.max_adopter_nodes < 1 or self.max_neighbor_nodes < 1:
            raise ValueError("max_adopter_nodes and max_neighbor_nodes must be >= 1")
        if self.n_time_bins < 2:
            raise ValueError("n_time_bins must be >= 2")
        if self.seed_order not in ("earliest", "latest"):
            raise ValueError("seed_order must be 'earliest' or 'latest'")
        if self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")


USER_FEATURE_NAMES = ["log_audience", "log_following", "log_prior_votes", "is_active",
                      "adoption_time_frac", "is_author", "hop", "log_exposers"]
N_USER_FEATURES = len(USER_FEATURE_NAMES)


def post_feature_names(n_time_bins: int) -> list[str]:
    names = ["log_obs_count"]
    names += [f"bin_frac_{i}" for i in range(n_time_bins)]
    names += ["log_accel", "log_last_bin",
              "log_author_audience", "log_author_following",
              "log_author_prior_votes", "log_author_prior_posts",
              "log_audience_sum", "log_audience_max", "log_audience_mean", "log_audience_unique",
              "in_graph_frac",
              "exposed_frac", "log_exposure_edges",
              "mean_log_prior_votes", "max_log_prior_votes",
              "hour_sin", "hour_cos", "is_weekend"]
    return names


def n_post_features(n_time_bins: int) -> int:
    return len(post_feature_names(n_time_bins))


# --------------------------------------------------------------------------- #
# Index structures (flat numpy arrays)
# --------------------------------------------------------------------------- #

def _require_int(name: str, arr: np.ndarray) -> None:
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integer ids, got dtype {arr.dtype}")


def _to_idx(universe: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Raw ids -> dense indices in the sorted `universe`. Unknown ids -> -1."""
    ids = np.asarray(ids)
    if len(universe) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(universe, ids), len(universe) - 1)
    return np.where(universe[pos] == ids, pos, -1).astype(np.int64)


def _gather(start: np.ndarray, end: np.ndarray):
    """Flat positions of all slices [start_i, end_i), plus the per-slice counts."""
    counts = end - start
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), counts
    offs = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) + np.repeat(start - offs, counts), counts


class TimedCSR:
    """Rows = dense owner ids; the entries of each row are sorted by time.

    keys = owner * stride + (time - t_min) is globally sorted, so "entries of owner u with
    time <= t" is a prefix of u's slice, found for many owners at once with one searchsorted.
    """

    def __init__(self, owner: np.ndarray, other: np.ndarray | None, times: np.ndarray, n_rows: int):
        owner = np.asarray(owner, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        if len(owner) and owner.min() < 0:
            raise ValueError("TimedCSR got an unknown (negative) owner index")
        order = np.lexsort((times, owner))
        owner_s, times_s = owner[order], times[order]
        self.indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(np.bincount(owner_s, minlength=n_rows), out=self.indptr[1:])
        self.other = None if other is None else np.asarray(other, dtype=np.int64)[order]
        if len(times):
            self.t_min, t_max = int(times.min()), int(times.max())
        else:
            self.t_min, t_max = 0, 0
        self.stride = t_max - self.t_min + 2
        if n_rows * self.stride >= 2 ** 62:
            raise OverflowError("n_rows * time range too large for int64 keys")
        self.keys = owner_s * self.stride + (times_s - self.t_min)

    def _end(self, nodes: np.ndarray, t: int, inclusive: bool) -> np.ndarray:
        q = int(np.clip(int(t) - self.t_min, -1, self.stride - 1))
        return np.searchsorted(self.keys, nodes * self.stride + q, side="right" if inclusive else "left")

    def count(self, nodes: np.ndarray, t: int, inclusive: bool = True) -> np.ndarray:
        """Entries per node with time <= t (inclusive) or time < t."""
        nodes = np.asarray(nodes, dtype=np.int64)
        return self._end(nodes, t, inclusive) - self.indptr[nodes]

    def ranges(self, nodes: np.ndarray, t: int):
        """[start, end) slice of each node's entries with time <= t."""
        nodes = np.asarray(nodes, dtype=np.int64)
        return self.indptr[nodes], self._end(nodes, t, True)


def build_indexes(posts: pd.DataFrame, all_posts: pd.DataFrame, events: pd.DataFrame, edges: pd.DataFrame):
    """posts = stories to process (split_meta); all_posts/events/edges = canonical tables."""
    post_ids = posts["post_id"].to_numpy()
    authors = posts["author"].to_numpy()
    ev_post, ev_user_raw = events["post_id"].to_numpy(), events["user"].to_numpy()
    e_src, e_dst = edges["source"].to_numpy(), edges["target"].to_numpy()
    ap_author = all_posts["author"].to_numpy()
    for name, arr in (("post_id", post_ids), ("author", authors), ("events.post_id", ev_post),
                      ("events.user", ev_user_raw), ("edges.source", e_src), ("edges.target", e_dst),
                      ("posts.author", ap_author)):
        _require_int(name, arr)
    if len(np.unique(post_ids)) != len(post_ids):
        raise ValueError("split_meta has duplicated post_id values")

    universe = np.unique(np.concatenate([e_src, e_dst, ev_user_raw, ap_author, authors]))
    n_users = len(universe)

    # ---- every vote, by user (prior-activity features) ----
    ev_user = _to_idx(universe, ev_user_raw)
    ev_time = events["time"].to_numpy().astype(np.int64)
    votes = TimedCSR(ev_user, None, ev_time, n_users)

    # ---- votes of the stories we process, by story, time-sorted ----
    post_universe = np.sort(post_ids)
    keep = np.isin(ev_post, post_universe)
    k_pidx = np.searchsorted(post_universe, ev_post[keep])
    k_user, k_time = ev_user[keep], ev_time[keep]
    order = np.lexsort((k_time, k_pidx))
    k_pidx, k_user, k_time = k_pidx[order], k_user[order], k_time[order]
    pair = k_pidx.astype(np.int64) * n_users + k_user
    if np.unique(pair).size != pair.size:
        raise ValueError("events contain the same (post, user) twice; run the adapter's cleaning first")
    ev_ptr = np.searchsorted(k_pidx, np.arange(len(post_universe) + 1)).astype(np.int64)

    # ---- friendships ----
    created = edges["created_at"].to_numpy().astype(np.int64)
    src, dst = _to_idx(universe, e_src), _to_idx(universe, e_dst)
    out = TimedCSR(src, dst, created, n_users)     # neighbours = audience of a user
    inn = TimedCSR(dst, None, created, n_users)    # degree only = whom a user follows

    # ---- stories per author, by t0 ----
    author_posts = TimedCSR(_to_idx(universe, ap_author), None,
                            all_posts["t0"].to_numpy().astype(np.int64), n_users)

    graph = dict(universe=universe, votes=votes, out=out, inn=inn, author_posts=author_posts,
                 ev_ptr=ev_ptr, ev_time=k_time, ev_user=k_user,
                 # scratch arrays, set and cleared inside build_post_sample (avoid sorting big edge lists)
                 _voted=np.zeros(n_users, dtype=bool), _taken=np.zeros(n_users, dtype=bool),
                 _vtime=np.zeros(n_users, dtype=np.int64), _local=np.full(n_users, -1, dtype=np.int64))
    post_pidx = np.searchsorted(post_universe, post_ids).astype(np.int64)
    author_idx = _to_idx(universe, authors)
    return graph, post_pidx, author_idx


def _hour_features(t0: int) -> tuple[float, float, float]:
    hour = (int(t0) % DAY) / 3600.0
    weekday = ((int(t0) // DAY) + 3) % 7        # 1970-01-01 was a Thursday; Monday = 0
    return float(np.sin(2 * np.pi * hour / 24)), float(np.cos(2 * np.pi * hour / 24)), float(weekday >= 5)


# --------------------------------------------------------------------------- #
# One story
# --------------------------------------------------------------------------- #

def build_post_sample(g, post_id: int, p_idx: int, author_idx: int, t0: int, t_obs: int, scfg: SampleConfig):
    """Features and subgraph of one story at t_obs. Returns (record, post_features, extras)."""
    out, inn, votes = g["out"], g["inn"], g["votes"]
    voted, taken, vtime, local = g["_voted"], g["_taken"], g["_vtime"], g["_local"]
    n_users = len(g["universe"])
    t0, t_obs = int(t0), int(t_obs)

    # Votes are time-sorted per story -> the observed ones are a prefix.
    s, e = int(g["ev_ptr"][p_idx]), int(g["ev_ptr"][p_idx + 1])
    m = int(np.searchsorted(g["ev_time"][s:e], t_obs, side="right"))
    active = g["ev_user"][s:s + m]            # one entry per user (checked in build_indexes), time order
    act_times = g["ev_time"][s:s + m]
    window = max(t_obs - t0, 1)

    voted[active] = True                      # scratch: who voted by t_obs, and when
    vtime[active] = act_times
    nodes = np.empty(0, dtype=np.int64)
    try:
        # ---- audience and exposure of the voters (all friendship edges visible at t_obs) ----
        empty = np.empty(0, dtype=np.int64)
        dst, aud_deg, following = empty, empty, empty
        cnt = np.zeros(n_users, dtype=np.int64)   # cnt[v] = number of "exposers" of user v (see below)
        if m > 0:
            a_start, a_end = out.ranges(active, t_obs)
            idx, aud_deg = _gather(a_start, a_end)
            dst = out.other[idx]
            # An edge u -> v exposes v if u voted before v (strictly), or v has not voted by t_obs.
            exposing = ~voted[dst] | (np.repeat(act_times, aud_deg) < vtime[dst])
            cnt = np.bincount(dst[exposing], minlength=n_users)
            following = inn.count(active, t_obs)

        # ---- story features ----
        nb = scfg.n_time_bins
        bins = np.clip(((act_times - t0) * nb) // window, 0, nb - 1).astype(np.int64)
        bin_counts = np.bincount(bins, minlength=nb)
        bin_frac = bin_counts / m if m > 0 else np.zeros(nb)
        accel = np.log1p(bin_counts[nb // 2:].sum()) - np.log1p(bin_counts[:nb // 2].sum())
        last_bin = np.log1p(bin_counts[-1])

        if author_idx >= 0:
            a = np.array([author_idx], dtype=np.int64)
            au_aud = int(out.count(a, t_obs)[0])
            au_fol = int(inn.count(a, t_obs)[0])
            au_votes = int(votes.count(a, t0, inclusive=False)[0])
            au_posts = int(g["author_posts"].count(a, t0, inclusive=False)[0])
        else:
            au_aud = au_fol = au_votes = au_posts = 0

        audience_nodes = empty                    # distinct users who follow a voter but have not voted
        if m > 0:
            log_prior = np.log1p(votes.count(active, t0, inclusive=False))   # votes before the story existed
            ex_adopters = cnt[active]
            nz = np.flatnonzero(cnt)
            audience_nodes = nz[~voted[nz]]
            in_graph = float(np.mean((aud_deg > 0) | (following > 0)))
            aud_sum, aud_max, aud_mean = float(aud_deg.sum()), float(aud_deg.max()), float(aud_deg.mean())
            exposed_frac = float(np.mean(ex_adopters > 0))
            exposure_edges = float(ex_adopters.sum())
            mean_prior, max_prior = float(log_prior.mean()), float(log_prior.max())
        else:
            in_graph, exposed_frac, exposure_edges = 0.0, 0.0, 0.0
            aud_sum = aud_max = aud_mean = mean_prior = max_prior = 0.0

        h_sin, h_cos, weekend = _hour_features(t0)
        feat = np.array(
            [np.log1p(m), *bin_frac, accel, last_bin,
             np.log1p(au_aud), np.log1p(au_fol), np.log1p(au_votes), np.log1p(au_posts),
             np.log1p(aud_sum), np.log1p(aud_max), np.log1p(aud_mean), np.log1p(audience_nodes.size), in_graph,
             exposed_frac, np.log1p(exposure_edges),
             mean_prior, max_prior,
             h_sin, h_cos, weekend],
            dtype=np.float64,
        )

        # ---- subgraph nodes: submitter, voters (capped), then audience by exposure ----
        pool = active if scfg.seed_order == "earliest" else active[::-1]
        if author_idx >= 0:
            seeds = np.concatenate(([author_idx], pool[pool != author_idx])).astype(np.int64)
        else:
            seeds = pool.astype(np.int64)
        seeds = seeds[:scfg.max_adopter_nodes]

        rng = np.random.default_rng([int(scfg.sample_seed), int(post_id)])
        levels, hop_vals = [seeds], [0.0]
        frontier = seeds
        for hop in range(1, scfg.hops + 1):
            if hop == 1:
                cand = audience_nodes               # audience of ALL voters, consistent with the features
            else:
                st, en = out.ranges(frontier, t_obs)
                cand = np.unique(out.other[_gather(st, en)[0]])
                cand = cand[~voted[cand]]           # voters are never "audience", even beyond the cap
            already = np.concatenate(levels)
            taken[already] = True
            new = cand[~taken[cand]]
            taken[already] = False
            if new.size == 0:
                break
            if new.size > scfg.max_neighbor_nodes:  # most exposed first, random among ties
                order = np.lexsort((rng.random(new.size), -cnt[new]))
                new = np.sort(new[order[:scfg.max_neighbor_nodes]])
            levels.append(new)
            hop_vals.append(hop / scfg.hops)
            frontier = new

        nodes = np.concatenate(levels)
        node_hop = np.repeat(hop_vals, [len(lv) for lv in levels])

        is_active = voted[nodes]
        adopt_frac = np.where(is_active, np.clip((vtime[nodes] - t0) / window, 0.0, 1.0), 0.0)

        user_x = np.zeros((len(nodes), N_USER_FEATURES), dtype=np.float32)
        user_x[:, 0] = np.log1p(out.count(nodes, t_obs))
        user_x[:, 1] = np.log1p(inn.count(nodes, t_obs))
        user_x[:, 2] = np.log1p(votes.count(nodes, t0, inclusive=False))
        user_x[:, 3] = is_active
        user_x[:, 4] = adopt_frac
        user_x[:, 5] = (nodes == author_idx) if author_idx >= 0 else 0.0
        user_x[:, 6] = node_hop
        user_x[:, 7] = np.log1p(cnt[nodes])

        # ---- subgraph edges (local indices into `nodes`) ----
        follows = np.empty((2, 0), dtype=np.int64)
        interacts = np.empty((2, 0), dtype=np.int64)
        interacts_attr = np.empty((0, 1), dtype=np.float32)
        authored = np.empty((2, 0), dtype=np.int64)
        if scfg.build_subgraph_edges:
            st, en = out.ranges(nodes, t_obs)
            idx, counts = _gather(st, en)
            if idx.size:
                local[nodes] = np.arange(len(nodes), dtype=np.int64)
                dst_local = local[out.other[idx]]
                local[nodes] = -1
                ok = dst_local >= 0
                src_local = np.repeat(np.arange(len(nodes), dtype=np.int64), counts)
                follows = np.stack([src_local[ok], dst_local[ok]])
            act_local = np.nonzero(is_active)[0].astype(np.int64)
            interacts = np.stack([act_local, np.zeros_like(act_local)])
            interacts_attr = adopt_frac[act_local, None].astype(np.float32)
            if author_idx >= 0:
                authored = np.array([[0], [0]], dtype=np.int64)   # the submitter is always nodes[0]

        rec = {
            "node_user_ids": g["universe"][nodes],
            "user_x": user_x,
            "post_x": feat[None, :].astype(np.float32),
            "follows": follows,
            "interacts": interacts,
            "interacts_attr": interacts_attr,
            "authored": authored,
        }
        extras = {
            "max_event_time": int(act_times[-1]) if m else -1,
            "n_obs": int(m),
            "n_nodes": int(len(nodes)),
            "n_follow_edges": int(follows.shape[1]),
        }
        return rec, feat, extras
    finally:
        voted[active] = False                    # always clear the scratch arrays
        vtime[active] = 0


# --------------------------------------------------------------------------- #
# Whole dataset
# --------------------------------------------------------------------------- #

def _edge_time_available(canon_dir: Path) -> bool:
    manifest = canon_dir / "manifest.json"
    if manifest.exists():
        return bool(json.loads(manifest.read_text()).get("edge_time_available", True))
    return True


def extract_all(
    base_path: str | Path,
    dataset_name: str = "digg",
    scfg: SampleConfig | None = None,
    *,
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Extract features for every story in split_meta. Returns the folder with the results."""
    if torch is None:
        raise RuntimeError("torch is required to save shards (pip install torch).")

    base = Path(base_path)
    canon_dir = base / "data" / "processed" / dataset_name
    splits_dir = base / "data" / "splits" / dataset_name
    out_dir = base / output_dir / dataset_name / "samples"

    scfg = scfg or SampleConfig()
    scfg.validate()
    if not _edge_time_available(canon_dir):
        raise ValueError(f"{dataset_name} has no edge times; this module needs timed edges (no static graphs).")

    posts = pd.read_parquet(splits_dir / "split_meta.parquet")
    posts = posts.drop(columns=[c for c in ("label", "growth") if c in posts.columns])  # never needed
    missing = {"post_id", "author", "t0", "t_obs"} - set(posts.columns)
    if missing:
        raise ValueError(f"split_meta.parquet is missing columns {sorted(missing)}")

    h = hashlib.sha256(json.dumps(asdict(scfg), sort_keys=True).encode())
    for col in ("post_id", "t0", "t_obs"):
        h.update(posts[col].to_numpy().tobytes())
    for fname in ("events.parquet", "edges.parquet", "posts.parquet"):
        h.update(str((canon_dir / fname).stat().st_size).encode())
    run_dir = out_dir / h.hexdigest()[:10]

    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "index.parquet").exists():
        log.info("Features already exist at %s. Skipping extraction.", run_dir)
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)                  # unfinished earlier run
    run_dir.mkdir(parents=True)
    (run_dir / "sample_config.json").write_text(json.dumps(asdict(scfg), indent=2))
    log.info("Extracting features to %s ...", run_dir)

    events = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=["source", "target", "created_at"])
    all_posts = pd.read_parquet(canon_dir / "posts.parquet", columns=["post_id", "author", "t0"])

    graph, post_pidx, author_idx = build_indexes(posts, all_posts, events, edges)
    del events, edges

    post_ids = posts["post_id"].to_numpy()
    t0, t_obs = posts["t0"].to_numpy(), posts["t_obs"].to_numpy()
    names = post_feature_names(scfg.n_time_bins)
    n_posts = len(posts)
    n_shards = (n_posts + scfg.shard_size - 1) // scfg.shard_size
    feats = np.empty((n_posts, len(names)), dtype=np.float64)
    extras: list[dict] = []

    for sid in range(n_shards):
        a, b = sid * scfg.shard_size, min((sid + 1) * scfg.shard_size, n_posts)
        shard_data = {}
        for i in range(a, b):
            rec, feats[i], ex = build_post_sample(
                graph, int(post_ids[i]), int(post_pidx[i]), int(author_idx[i]), int(t0[i]), int(t_obs[i]), scfg)
            shard_data[int(post_ids[i])] = rec
            extras.append(ex)
        tmp = run_dir / f"shard_{sid}.pt.tmp"
        torch.save(shard_data, tmp)
        os.replace(tmp, run_dir / f"shard_{sid}.pt")
        log.info("  shard %d/%d done (%d stories)", sid + 1, n_shards, b - a)

    ex = pd.DataFrame(extras)
    # --- hard checks ---
    if (ex["max_event_time"].to_numpy() > t_obs).any():
        raise RuntimeError("CRITICAL LEAK: a vote after t_obs was used for a feature")
    if "observed_count" in posts.columns and (ex["n_obs"].to_numpy() != posts["observed_count"].to_numpy()).any():
        bad = int((ex["n_obs"].to_numpy() != posts["observed_count"].to_numpy()).sum())
        raise RuntimeError(f"{bad} stories: observed vote count differs from the splitter's observed_count")
    if not np.isfinite(feats).all():
        raise RuntimeError("NaN or inf in the extracted features")

    tab = pd.DataFrame(feats, columns=names)
    tab.insert(0, "post_id", post_ids)
    tab.to_parquet(run_dir / "tabular_features.parquet", index=False)
    idx_df = pd.DataFrame({
        "post_id": post_ids,
        "shard": np.arange(n_posts) // scfg.shard_size,
        "t_obs": t_obs,
        "max_event_time": ex["max_event_time"].to_numpy(),
        "n_nodes": ex["n_nodes"].to_numpy(),
        "n_follow_edges": ex["n_follow_edges"].to_numpy(),
    })
    idx_df.to_parquet(run_dir / "index.parquet", index=False)   # written last: marks the run as complete

    log.info("Extraction complete: %d stories, nodes/story median %d, max %d",
             n_posts, int(idx_df["n_nodes"].median()), int(idx_df["n_nodes"].max()))
    return run_dir


# --------------------------------------------------------------------------- #
# Audit 1: recompute with plain pandas (no CSR tricks) and compare
# --------------------------------------------------------------------------- #

def _reference(pm, events: pd.DataFrame, edges: pd.DataFrame, all_posts: pd.DataFrame, nb: int):
    """Every story feature, re-implemented with pandas. Returns (features dict, helper tables)."""
    t0, t_obs = int(pm["t0"]), int(pm["t_obs"])
    window = max(t_obs - t0, 1)
    author, pid = pm["author"], pm["post_id"]

    ev = events[(events["post_id"] == pid) & (events["time"] <= t_obs)]
    ed = edges[edges["created_at"] <= t_obs]
    adopt_time = dict(zip(ev["user"].tolist(), ev["time"].tolist()))
    adopters = list(adopt_time)
    m = len(adopters)
    out_deg = ed.groupby("source").size()
    in_deg = ed.groupby("target").size()
    prior = events.loc[events["time"] < t0, "user"].value_counts()

    f = {"log_obs_count": np.log1p(m)}
    times = ev["time"].to_numpy()
    counts = np.bincount(np.clip(((times - t0) * nb) // window, 0, nb - 1).astype(int), minlength=nb)
    for i in range(nb):
        f[f"bin_frac_{i}"] = counts[i] / m if m else 0.0
    f["log_accel"] = np.log1p(counts[nb // 2:].sum()) - np.log1p(counts[:nb // 2].sum())
    f["log_last_bin"] = np.log1p(counts[-1])
    f["log_author_audience"] = np.log1p(out_deg.get(author, 0))
    f["log_author_following"] = np.log1p(in_deg.get(author, 0))
    f["log_author_prior_votes"] = np.log1p(prior.get(author, 0))
    f["log_author_prior_posts"] = np.log1p(int(((all_posts["author"] == author) & (all_posts["t0"] < t0)).sum()))

    sub = ed[ed["source"].isin(adopters)]
    t_src = sub["source"].map(adopt_time)
    t_dst = sub["target"].map(adopt_time)
    exposing = t_dst.isna() | (t_src < t_dst)
    exp_counts = sub.loc[exposing, "target"].value_counts()
    aud = np.array([out_deg.get(u, 0) for u in adopters], dtype=np.float64)
    pv = np.log1p(np.array([prior.get(u, 0) for u in adopters], dtype=np.float64))
    if m:
        f["log_audience_sum"], f["log_audience_max"] = np.log1p(aud.sum()), np.log1p(aud.max())
        f["log_audience_mean"] = np.log1p(aud.mean())
        f["in_graph_frac"] = float(np.mean([(out_deg.get(u, 0) > 0) or (in_deg.get(u, 0) > 0) for u in adopters]))
        f["exposed_frac"] = float(np.mean([exp_counts.get(u, 0) > 0 for u in adopters]))
        f["log_exposure_edges"] = np.log1p(sum(exp_counts.get(u, 0) for u in adopters))
        f["mean_log_prior_votes"], f["max_log_prior_votes"] = pv.mean(), pv.max()
    else:
        for k in ("log_audience_sum", "log_audience_max", "log_audience_mean", "in_graph_frac", "exposed_frac",
                  "log_exposure_edges", "mean_log_prior_votes", "max_log_prior_votes"):
            f[k] = 0.0
    f["log_audience_unique"] = np.log1p(sub.loc[t_dst.isna(), "target"].nunique())
    ts = pd.Timestamp(t0, unit="s", tz="UTC")
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["is_weekend"] = float(ts.dayofweek >= 5)
    helpers = dict(adopt_time=adopt_time, exp_counts=exp_counts, out_deg=out_deg, in_deg=in_deg,
                   prior=prior, ed=ed, sub=sub, t_dst=t_dst, t0=t0, window=window)
    return f, helpers


def audit(base_path: str | Path, dataset_name: str, run_dir: str | Path, n_posts: int = 30, seed: int = 0) -> dict:
    """Independent check of sampled stories. Raises RuntimeError listing every mismatch."""
    if torch is None:
        raise RuntimeError("torch is required to read shards")
    base, run_dir = Path(base_path), Path(run_dir)
    canon_dir = base / "data" / "processed" / dataset_name
    scfg = SampleConfig(**json.loads((run_dir / "sample_config.json").read_text()))
    names = post_feature_names(scfg.n_time_bins)

    posts = pd.read_parquet(base / "data" / "splits" / dataset_name / "split_meta.parquet").set_index("post_id", drop=False)
    index = pd.read_parquet(run_dir / "index.parquet").set_index("post_id")
    tab = pd.read_parquet(run_dir / "tabular_features.parquet").set_index("post_id")
    events = pd.read_parquet(canon_dir / "events.parquet")
    edges = pd.read_parquet(canon_dir / "edges.parquet")
    all_posts = pd.read_parquet(canon_dir / "posts.parquet")

    rng = np.random.default_rng(seed)
    picks = rng.choice(posts.index.to_numpy(), size=min(n_posts, len(posts)), replace=False)
    problems, shards = [], {}

    for pid in picks:
        pm = posts.loc[pid]
        tag = f"[post {pid}]"
        f, hp = _reference(pm, events, edges, all_posts, scfg.n_time_bins)
        for name in names:
            if not np.isclose(tab.loc[pid, name], f[name], atol=1e-5):
                problems.append(f"{tag} {name}: stored {tab.loc[pid, name]:.6f} != recomputed {f[name]:.6f}")
        if index.loc[pid, "max_event_time"] > int(pm["t_obs"]):
            problems.append(f"{tag} used a vote after t_obs")

        shard = int(index.loc[pid, "shard"])
        if shard not in shards:
            shards = {shard: torch.load(run_dir / f"shard_{shard}.pt", weights_only=False)}
        rec = shards[shard][int(pid)]
        users, ux = rec["node_user_ids"], rec["user_x"]
        adopters, author = set(hp["adopt_time"]), pm["author"]
        t0, window = hp["t0"], hp["window"]

        if len(users) == 0 or users[0] != author:
            problems.append(f"{tag} submitter is not node 0")
        hops = ux[:, 6]
        seed_nodes = set(users[hops == 0].tolist())
        if len(adopters) + (author not in adopters) <= scfg.max_adopter_nodes:
            if seed_nodes != adopters | {author}:
                problems.append(f"{tag} voter nodes differ from the observed voters")
        elif len(seed_nodes) != scfg.max_adopter_nodes or not seed_nodes <= adopters | {author}:
            problems.append(f"{tag} voter nodes are not a capped subset of the voters")
        # audience nodes
        aud_nodes = users[hops > 0]
        if set(aud_nodes.tolist()) & adopters:
            problems.append(f"{tag} an audience node is a voter")
        if scfg.hops >= 1:
            hop1 = users[np.isclose(hops, 1.0 / scfg.hops)]
            cands = hp["sub"].loc[hp["t_dst"].isna(), "target"].unique()
            if len(hop1) != min(scfg.max_neighbor_nodes, len(cands)):
                problems.append(f"{tag} hop-1 has {len(hop1)} nodes, expected {min(scfg.max_neighbor_nodes, len(cands))}")
            if len(cands) > scfg.max_neighbor_nodes:
                ec = hp["exp_counts"]
                picked = set(hop1.tolist())
                lo = min(ec.get(u, 0) for u in picked)
                hi = max([ec.get(u, 0) for u in cands if u not in picked] or [0])
                if lo < hi:
                    problems.append(f"{tag} audience not ranked by exposure (picked min {lo} < left-out max {hi})")
        # node features
        for j, u in enumerate(users.tolist()):
            exp = [np.log1p(hp["out_deg"].get(u, 0)), np.log1p(hp["in_deg"].get(u, 0)),
                   np.log1p(hp["prior"].get(u, 0)), float(u in adopters),
                   float(np.clip((hp["adopt_time"][u] - t0) / window, 0, 1)) if u in adopters else 0.0,
                   float(u == author), None, np.log1p(hp["exp_counts"].get(u, 0))]
            for c, v in enumerate(exp):
                if v is not None and not np.isclose(ux[j, c], v, atol=1e-5):
                    problems.append(f"{tag} node {u} {USER_FEATURE_NAMES[c]}: {ux[j, c]:.5f} != {v:.5f}")
                    break
        # edges
        if scfg.build_subgraph_edges:
            got_int = set(users[rec["interacts"][0]].tolist())
            if got_int != adopters & set(users.tolist()):
                problems.append(f"{tag} interacts users differ from observed voters in the sample")
            ed = hp["ed"]
            insample = ed[ed["source"].isin(users) & ed["target"].isin(users)]
            truth = set(zip(insample["source"].tolist(), insample["target"].tolist()))
            got = set(zip(users[rec["follows"][0]].tolist(), users[rec["follows"][1]].tolist()))
            if got != truth:
                problems.append(f"{tag} follows: {len(got - truth)} extra, {len(truth - got)} missing edges")

    if problems:
        raise RuntimeError(f"Audit found {len(problems)} problem(s):\n - " + "\n - ".join(problems[:15]))
    log.info("Audit passed on %d stories", len(picks))
    return {"checked": int(len(picks)), "problems": 0}


# --------------------------------------------------------------------------- #
# Audit 2: leak test. Remove the future from the data; the output must not change.
# --------------------------------------------------------------------------- #

def leak_test(base_path: str | Path, dataset_name: str = "digg", n_posts: int = 10, seed: int = 0,
              scfg: SampleConfig | None = None) -> dict:
    """For sampled stories, rebuild features from votes and friendships cut off at t_obs.

    Any dependence on information after t_obs shows up as a difference. Needs no torch.
    """
    base = Path(base_path)
    canon_dir = base / "data" / "processed" / dataset_name
    scfg = scfg or SampleConfig()
    posts = pd.read_parquet(base / "data" / "splits" / dataset_name / "split_meta.parquet")
    posts = posts.drop(columns=[c for c in ("label", "growth") if c in posts.columns])
    events = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=["source", "target", "created_at"])
    all_posts = pd.read_parquet(canon_dir / "posts.parquet", columns=["post_id", "author", "t0"])

    g_full, pidx_full, aidx_full = build_indexes(posts, all_posts, events, edges)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(posts), size=min(n_posts, len(posts)), replace=False)
    problems = []

    for i in picks:
        row = posts.iloc[[i]]
        pid, t0, t_obs = int(row["post_id"].iloc[0]), int(row["t0"].iloc[0]), int(row["t_obs"].iloc[0])
        rec_f, feat_f, _ = build_post_sample(g_full, pid, int(pidx_full[i]), int(aidx_full[i]), t0, t_obs, scfg)

        ev_cut = events[events["time"] <= t_obs]
        ed_cut = edges[edges["created_at"] <= t_obs]
        ap_cut = all_posts[all_posts["t0"] <= t_obs]
        g_cut, pidx_cut, aidx_cut = build_indexes(row, ap_cut, ev_cut, ed_cut)
        rec_c, feat_c, _ = build_post_sample(g_cut, pid, int(pidx_cut[0]), int(aidx_cut[0]), t0, t_obs, scfg)

        tag = f"[post {pid}]"
        if not np.allclose(feat_f, feat_c, atol=1e-6):
            diff = [n for n, a, b in zip(post_feature_names(scfg.n_time_bins), feat_f, feat_c)
                    if not np.isclose(a, b, atol=1e-6)]
            problems.append(f"{tag} features change when the future is removed: {diff}")
        if not np.array_equal(rec_f["node_user_ids"], rec_c["node_user_ids"]):
            problems.append(f"{tag} sampled users change when the future is removed")
        elif not np.allclose(rec_f["user_x"], rec_c["user_x"], atol=1e-6):
            problems.append(f"{tag} user features change when the future is removed")
        else:
            ef = set(zip(rec_f["node_user_ids"][rec_f["follows"][0]].tolist(), rec_f["node_user_ids"][rec_f["follows"][1]].tolist()))
            ec = set(zip(rec_c["node_user_ids"][rec_c["follows"][0]].tolist(), rec_c["node_user_ids"][rec_c["follows"][1]].tolist()))
            if ef != ec:
                problems.append(f"{tag} subgraph edges change when the future is removed")

    if problems:
        raise RuntimeError(f"Leak test failed on {len(problems)} stories:\n - " + "\n - ".join(problems[:10]))
    log.info("Leak test passed: %d stories give identical output without any data after t_obs", len(picks))
    return {"checked": int(len(picks)), "problems": 0}


def _parse_args() -> argparse.Namespace:
    d = SampleConfig()
    p = argparse.ArgumentParser(description="Extract per-story features and subgraphs at t_obs.")
    p.add_argument("--dataset", default="digg")
    p.add_argument("--base-path", default=".")
    p.add_argument("--hops", type=int, default=d.hops)
    p.add_argument("--max-adopter-nodes", type=int, default=d.max_adopter_nodes)
    p.add_argument("--max-neighbor-nodes", type=int, default=d.max_neighbor_nodes)
    p.add_argument("--n-time-bins", type=int, default=d.n_time_bins)
    p.add_argument("--shard-size", type=int, default=d.shard_size)
    p.add_argument("--no-subgraph-edges", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute even if features exist")
    p.add_argument("--audit", action="store_true", help="recompute sampled stories with plain pandas and compare")
    p.add_argument("--leak-test", action="store_true", help="rebuild stories from data cut off at t_obs and compare")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    cfg = SampleConfig(
        hops=a.hops, max_adopter_nodes=a.max_adopter_nodes, max_neighbor_nodes=a.max_neighbor_nodes,
        n_time_bins=a.n_time_bins, shard_size=a.shard_size, build_subgraph_edges=not a.no_subgraph_edges,
    )
    out = extract_all(a.base_path, a.dataset, cfg, force=a.force)
    if a.audit:
        audit(a.base_path, a.dataset, out)
    if a.leak_test:
        leak_test(a.base_path, a.dataset, scfg=cfg)