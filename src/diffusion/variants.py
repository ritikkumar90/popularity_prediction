"""The two diffusion cases, on top of one fast discrete-time cascade engine.

    Case 1  predict_popularity()  how many users will have adopted a post at t_end, given what was
                                  seen up to t_obs (growth is simulated only from t_obs to t_end).
    Case 2  choose_seeds()        the author posts and may add k = 1, 2, 3 extra seed users;
                                  which seeds maximise the expected number of adopters by t_end?

Model (same one that fit.py estimates)
--------------------------------------
Time is cut into bins of `bin_seconds`. When u adopts in bin s, each follower v (edge u -> v) adopts
at s + d with probability p_u * q[d], d = 0..D (d = 0: same bin). Edges, p_u and q come from fit.py.

Why it is fast and why seed selection is stable
-----------------------------------------------
* The graph is one global CSR (numpy arrays); nothing is rebuilt per post. Edge visibility is a
  per-edge test, `created_bin <= min(adoption bin of the source, time of knowledge)`, so a
  simulation never uses an edge from the future.
* A "world" is a fixed assignment of random numbers to edges. The number for (world, edge) is a hash,
  so nothing is stored, and the same world gives the same random numbers to EVERY candidate seed
  (common random numbers). Marginal gains are then comparable, greedy and CELF agree, and adding a
  seed can never lower the spread inside a world.
* Cascades run in compiled code (numba) with a bucket queue per time bin, parallel over
  (seed set, world) pairs.

Conditioning on what was observed (Case 1 and Case 2 "addon")
-------------------------------------------------------------
For an observed adopter u (adopted in bin s) and a follower v who had NOT adopted by the end of bin
`cond_bin`, the lag of that edge is known to be > cond_bin - s. The engine draws (fires?, lag) from the
conditional distribution
        P(fires | not yet) = p (1 - Q) / (1 - p Q),      lag ~ q[d] / (1 - Q) for d > cond_bin - s
with Q = cdf of q at cond_bin - s. Restarting the delay from zero would over-predict growth.

Counting rule (matches splitter.py: label = events with time <= t_end)
----------------------------------------------------------------------
The engine returns n_new = users activated by the simulation, i.e. NOT in the initial set.
    Case 1 prediction  = observed_count + n_new (+ optional external term)
    Case 2 spread      = n_new with the chosen seeds counted as new adopters (they are readers
                         too); `gain` = spread(seeds) - spread(author alone).

What selectors.py must provide (contract used by choose_seeds)
--------------------------------------------------------------
    greedy(objective, candidates, k)  and  celf(objective, candidates, k)
  where `objective` is a SeedObjective:
    objective.value(seeds)         -> float, expected spread of the author plus these seeds
    objective.values(base, cands)  -> np.ndarray, expected spread of base + [c] for every c in cands
    objective.n_evals              -> counter of seed sets evaluated so far
  and both return an object with .seeds (list of dense node ids, in pick order),
  .values (expected spread after 1, 2, ... seeds) and .n_evals.

Typical use (simulator.py will do this)
---------------------------------------
    engine = CascadeEngine.from_files(".", "flickr")
    posts  = PostSet.from_files(engine, ".", "flickr", split="test")
    pred   = predict_popularity(engine, posts)                  # Case 1 -> DataFrame
    seeds  = choose_seeds(engine, posts, mode="cold")           # Case 2 -> DataFrame

Self-test on a synthetic graph (no data needed):   python -m src.diffusion.variants --selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import numba
    from numba import njit, prange
except ImportError as exc:  # numba is a hard requirement: a pure-python cascade loop is far too slow
    raise ImportError("variants.py needs numba (pip install numba)") from exc

log = logging.getLogger(__name__)

# ============================================================================ #
# SETTINGS (every function takes these as keyword arguments; simulator.py exposes them as flags)
# ============================================================================ #
OUTPUT_DIR = "outputs"
SPLIT = "test"                  # which split to run on
N_WORLDS = 200                  # Case 1: Monte Carlo worlds per post
EXTERNAL_PER_BIN = 0.0          # Case 1: extra adopters per remaining bin from outside the network (calibrate on val)
SEED_MODE = "cold"              # Case 2: "cold" = seeds start at t0 | "addon" = seeds start right after t_obs
K_VALUES = (1, 2, 3)
SELECT_METHODS = ("celf",)      # "greedy" and/or "celf" (implemented in selectors.py)
N_CANDIDATES = 1000             # Case 2: candidate pool size (top users by p_u * visible out-degree)
N_SELECT_WORLDS = 200           # Case 2: worlds used while choosing seeds
N_EVAL_WORLDS = 500             # Case 2: fresh, independent worlds used to score the chosen seeds
SEED = 0
MAX_CHUNKS = None               # None = number of numba threads; lower it to save memory on huge graphs

_MAX_RUN_ID = 2**31 - 2
_STATIC_BIN = -(2**31) + 1      # created-bin of every edge when the graph is treated as static

# ============================================================================ #
# Hash-based random numbers (one pair of uniforms per (world, edge); nothing is stored)
# ============================================================================ #
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0


@njit(cache=True)
def _mix(x):
    x = (x ^ (x >> _S30)) * _M1
    x = (x ^ (x >> _S27)) * _M2
    return x ^ (x >> _S31)


@njit(cache=True)
def _unit(x):
    return (x >> _S11) * _INV53


# ============================================================================ #
# Kernel: one cascade
# ============================================================================ #
@njit(cache=True)
def _push(head, ent_node, ent_next, n_ent, b, v):
    if n_ent == len(ent_node):
        new_node = np.empty(2 * n_ent, np.int32)
        new_next = np.empty(2 * n_ent, np.int64)
        new_node[:n_ent] = ent_node
        new_next[:n_ent] = ent_next
        ent_node = new_node
        ent_next = new_next
    ent_node[n_ent] = v
    ent_next[n_ent] = head[b]
    head[b] = n_ent
    return ent_node, ent_next, n_ent + 1


@njit(cache=True)
def _cascade(wkey, indptr, indices, cbin, p, cdf, init_nodes, init_bins, seeds,
             cond_bin, vis_bin, end_bin, strict, stamp, tbin, act, run):
    """Number of users activated by the simulation (not in init_nodes) with bin <= end_bin.

    init_nodes / init_bins   users already active (observed adopters, the author) and their bins
    seeds                    extra users activated in bin cond_bin + 1 (padded with -1)
    cond_bin                 last bin that is already observed; non-adopters were seen not to adopt up to it
    vis_bin                  last bin whose edges are known (an edge is usable if created_bin <=
                             min(source adoption bin, vis_bin))
    stamp/tbin/act          per-thread scratch arrays (validity = value == run), never cleared
    """
    D = len(cdf) - 1
    base = cond_bin + 1
    nbk = end_bin - base + 1
    if nbk < 0:
        nbk = 0
    head = np.full(nbk + 1, -1, np.int64)
    ent_node = np.empty(1024, np.int32)
    ent_next = np.empty(1024, np.int64)
    n_ent = 0
    n_init = len(init_nodes)

    for i in range(n_init):
        act[init_nodes[i]] = run
    for i in range(len(seeds)):
        s = seeds[i]
        if s < 0 or nbk == 0:
            continue
        if act[s] == run or stamp[s] == run:
            continue
        stamp[s] = run
        tbin[s] = base
        ent_node, ent_next, n_ent = _push(head, ent_node, ent_next, n_ent, 0, s)

    n_new = 0
    qi = 0
    cur = -1                       # -1: expanding the initial users; >= 0: bucket being processed
    while True:
        if cur < 0:
            if qi < n_init:
                u = init_nodes[qi]
                su = init_bins[qi]
                qi += 1
                T = cond_bin - su
                if T < 0:
                    Qc = 0.0
                    lag_min = 0
                elif T >= D or cdf[T] >= 1.0:
                    continue       # no delay mass left: this user cannot trigger anyone any more
                else:
                    Qc = cdf[T]
                    lag_min = T + 1
            else:
                cur = 0
                continue
        else:
            if cur >= nbk:
                break
            e = head[cur]
            if e < 0:
                cur += 1
                continue
            head[cur] = ent_next[e]
            u = ent_node[e]
            if act[u] == run:
                continue           # stale entry: already activated earlier
            act[u] = run
            n_new += 1
            su = base + cur
            Qc = 0.0
            lag_min = 0

        pu = p[u]
        pc = pu * (1.0 - Qc) / (1.0 - pu * Qc)
        if pc <= 0.0:
            continue
        lim = su if su < vis_bin else vis_bin
        for k in range(indptr[u], indptr[u + 1]):
            if strict and cbin[k] > lim:
                continue
            v = indices[k]
            if act[v] == run:
                continue
            z = _mix(wkey + np.uint64(k) * _GOLD)
            if _unit(z) >= pc:
                continue
            tgt = Qc + _unit(_mix(z + _GOLD)) * (1.0 - Qc)
            lag = np.searchsorted(cdf, tgt, side="right")
            if lag > D:
                lag = D
            if lag < lag_min:
                lag = lag_min
            bn = su + lag
            if bn > end_bin:
                continue
            if stamp[v] != run or bn < tbin[v]:
                stamp[v] = run
                tbin[v] = bn
                ent_node, ent_next, n_ent = _push(head, ent_node, ent_next, n_ent, bn - base, v)
    return n_new


@njit(parallel=True, cache=True)
def _eval_kernel(indptr, indices, cbin, p, cdf, init_nodes, init_bins, cond_bin, vis_bin, end_bin,
                 strict, sets, n_worlds, salt, stamp, tbin, act, run_ctr, out):
    """out[set * n_worlds + world] = n_new. Tasks are split into contiguous chunks, one per thread."""
    n_chunks = stamp.shape[0]
    total = sets.shape[0] * n_worlds
    for c in prange(n_chunks):
        lo = (total * c) // n_chunks
        hi = (total * (c + 1)) // n_chunks
        st = stamp[c]
        tb = tbin[c]
        ac = act[c]
        for t in range(lo, hi):
            si = t // n_worlds
            w = t - si * n_worlds
            run_ctr[c] += 1
            wkey = _mix(np.uint64(w) * _GOLD + np.uint64(salt))
            out[t] = _cascade(wkey, indptr, indices, cbin, p, cdf, init_nodes, init_bins, sets[si],
                              cond_bin, vis_bin, end_bin, strict, st, tb, ac, run_ctr[c])


# ============================================================================ #
# Engine
# ============================================================================ #
class CascadeEngine:
    """Graph (CSR) + fitted parameters + scratch buffers. Dense node ids are positions in `self.nodes`."""

    def __init__(self, nodes, indptr, indices, cbin, p, delay_pmf, bin_seconds, strict, n_chunks=None):
        self.nodes = nodes
        self.indptr, self.indices, self.cbin, self.p = indptr, indices, cbin, p
        pmf = np.asarray(delay_pmf, np.float64)
        self.cdf = np.cumsum(pmf / pmf.sum())
        self.cdf[-1] = 1.0
        self.bin_seconds = int(bin_seconds)
        self.strict = bool(strict)
        n = len(nodes)
        n_chunks = int(n_chunks or MAX_CHUNKS or numba.get_num_threads())
        self._alloc(n_chunks, n)
        self._pool_cache: dict = {}
        self._deg_cache: dict = {}

    def _alloc(self, n_chunks, n):
        self._stamp = np.zeros((n_chunks, n), np.int32)
        self._tbin = np.zeros((n_chunks, n), np.int32)
        self._act = np.zeros((n_chunks, n), np.int32)
        self._run = np.zeros(n_chunks, np.int64)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, src_raw, dst_raw, edge_bin, delay_pmf, bin_seconds, *, strict=True,
                    p_global=0.05, p_users=None, n_chunks=None):
        """src_raw -> dst_raw: information flows from src to dst. edge_bin: bin in which the edge became
        visible (ignored when strict=False). p_users = (raw user ids, p values); everyone else gets p_global."""
        src_raw, dst_raw = np.asarray(src_raw), np.asarray(dst_raw)
        nodes = np.unique(np.concatenate([src_raw, dst_raw]))
        n = len(nodes)
        src = np.searchsorted(nodes, src_raw).astype(np.int32)
        dst = np.searchsorted(nodes, dst_raw).astype(np.int32)
        order = np.argsort(src, kind="stable")
        indices = np.ascontiguousarray(dst[order])
        if strict:
            eb = np.asarray(edge_bin, np.int64)[order]
            if len(eb) and (eb.max() >= 2**31 or eb.min() <= _STATIC_BIN):
                raise ValueError("edge bins do not fit in int32; use a larger bin_seconds")
            cbin = eb.astype(np.int32)
        else:
            cbin = np.full(len(indices), _STATIC_BIN, np.int32)
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
        p = np.full(n, float(p_global), np.float64)
        eng = cls(nodes, indptr, indices, cbin, p, delay_pmf, bin_seconds, strict, n_chunks)
        if p_users is not None:
            ids, vals = np.asarray(p_users[0]), np.asarray(p_users[1], np.float64)
            idx = eng.dense(ids)
            ok = idx >= 0
            eng.p[idx[ok]] = vals[ok]
        return eng

    @classmethod
    def from_files(cls, base_path=".", dataset_name="flickr", output_dir=OUTPUT_DIR, n_chunks=None):
        """Load the fit written by fit.py and the full edge table."""
        base = Path(base_path)
        fit_dir = base / output_dir / dataset_name / "fit"
        params = json.loads((fit_dir / "fitted_params.json").read_text())
        infl = pd.read_parquet(fit_dir / "source_influence.parquet", columns=["user", "p"])
        strict = bool(params.get("strict_edge_timing", True))
        bs = int(params["bin_seconds"])

        ecols = ["source", "target"] + (["created_at"] if strict else [])
        edges = pd.read_parquet(base / "data" / "processed" / dataset_name / "edges.parquet", columns=ecols)
        s, t = edges["source"].to_numpy(), edges["target"].to_numpy()
        if params.get("reverse_edges", False):
            s, t = t, s        # must match the direction used when fitting
        eb = (edges["created_at"].to_numpy(np.int64) // bs) if strict else None
        del edges
        log.info("Engine: %d edges, strict_edge_timing=%s, bin=%d s, p_global=%.3e",
                 len(s), strict, bs, params["p_global"])
        return cls.from_arrays(s, t, eb, params["delay_pmf"], bs, strict=strict,
                               p_global=params["p_global"], p_users=(infl["user"], infl["p"]),
                               n_chunks=n_chunks)

    # ------------------------------------------------------------------ helpers
    def dense(self, raw_ids) -> np.ndarray:
        """Raw user ids -> dense node ids (int32), -1 for users that are not in the graph."""
        raw_ids = np.asarray(raw_ids)
        if len(self.nodes) == 0:
            return np.full(len(raw_ids), -1, np.int32)
        idx = np.searchsorted(self.nodes, raw_ids)
        idx_c = np.minimum(idx, len(self.nodes) - 1)
        return np.where(self.nodes[idx_c] == raw_ids, idx_c, -1).astype(np.int32)

    def visible_outdeg(self, vis_bin: int) -> np.ndarray:
        """Out-degree counting only edges visible at vis_bin (cached per bin)."""
        if not self.strict:
            return np.diff(self.indptr)
        deg = self._deg_cache.get(vis_bin)
        if deg is None:
            c = np.zeros(len(self.cbin) + 1, np.int32)
            np.cumsum(self.cbin <= vis_bin, out=c[1:], dtype=np.int32)
            deg = (c[self.indptr[1:]] - c[self.indptr[:-1]]).astype(np.int64)
            self._deg_cache[vis_bin] = deg
        return deg

    def pool(self, vis_bin: int, kind: str, size: int) -> np.ndarray:
        """Top `size` users, best first, by 'score' = p_u * visible out-degree or by 'degree'."""
        size = int(-(-size // 1000) * 1000)           # round up so the cache is rarely rebuilt
        key = (vis_bin, kind)
        hit = self._pool_cache.get(key)
        if hit is not None and (len(hit) >= size or len(hit) < size and hit.size == self.n_nodes):
            return hit
        deg = self.visible_outdeg(vis_bin)
        score = deg * self.p if kind == "score" else deg.astype(np.float64)
        size = min(size, self.n_nodes)
        top = np.argpartition(-score, size - 1)[:size] if size < self.n_nodes else np.arange(self.n_nodes)
        top = top[np.argsort(-score[top], kind="stable")]
        top = top[score[top] > 0].astype(np.int32)
        self._pool_cache[key] = top
        return top

    # ------------------------------------------------------------------ simulation
    def eval_sets(self, init_nodes, init_bins, cond_bin, vis_bin, end_bin, sets=None,
                  n_worlds=N_WORLDS, salt=0) -> np.ndarray:
        """n_new for every (seed set, world): int32 array of shape (n_sets, n_worlds).

        sets: int array (n_sets, k) of dense ids padded with -1; None = one empty set (no extra seeds).
        Worlds 0..n_worlds-1 are the same for every set (common random numbers); `salt` picks another
        independent collection of worlds.
        """
        sets = np.full((1, 1), -1, np.int32) if sets is None else np.ascontiguousarray(sets, np.int32)
        if sets.ndim == 1:
            sets = sets.reshape(-1, 1)
        n_sets = sets.shape[0]
        total = n_sets * n_worlds
        n_chunks = self._stamp.shape[0]
        if self._run.max() + total // n_chunks + 2 > _MAX_RUN_ID:     # run ids are int32 stamps
            self._alloc(n_chunks, self.n_nodes)
        out = np.empty(total, np.int32)
        _eval_kernel(self.indptr, self.indices, self.cbin, self.p, self.cdf,
                     np.ascontiguousarray(init_nodes, np.int32), np.ascontiguousarray(init_bins, np.int64),
                     int(cond_bin), int(vis_bin), int(end_bin), self.strict, sets, int(n_worlds), int(salt),
                     self._stamp, self._tbin, self._act, self._run, out)
        return out.reshape(n_sets, n_worlds)


# ============================================================================ #
# Posts: metadata + events, and the starting state of each case
# ============================================================================ #
@dataclass
class PostState:
    init_nodes: np.ndarray      # dense ids active at the start
    init_bins: np.ndarray       # their adoption bins
    cond_bin: int               # last observed bin (seeds start in cond_bin + 1)
    vis_bin: int                # last bin whose edges are known
    end_bin: int                # last bin that counts (t_end)
    observed_count: int         # events with time <= t_obs (Case 1)


class PostSet:
    """Posts of one split with their events (time <= t_end), ready to build starting states."""

    def __init__(self, engine, meta, ptr, ev_time, ev_user_raw):
        self.engine, self.meta = engine, meta.reset_index(drop=True)
        self.ptr, self.ev_time, self.ev_user_raw = ptr, ev_time, ev_user_raw
        self.ev_user = engine.dense(ev_user_raw)
        self.author = engine.dense(self.meta["author"].to_numpy())
        self.t0 = self.meta["t0"].to_numpy(np.int64)
        self.t_obs = self.meta["t_obs"].to_numpy(np.int64)
        self.t_end = self.meta["t_end"].to_numpy(np.int64)

    def __len__(self):
        return len(self.meta)

    @classmethod
    def from_files(cls, engine, base_path=".", dataset_name="flickr", split=SPLIT, max_posts=None, seed=SEED):
        base = Path(base_path)
        meta = pd.read_parquet(base / "data" / "splits" / dataset_name / "split_meta.parquet")
        need = {"post_id", "author", "t0", "t_obs", "t_end", "split"}
        if need - set(meta.columns):
            raise ValueError(f"split_meta.parquet is missing columns {sorted(need - set(meta.columns))}")
        meta = meta[meta["split"] == split].reset_index(drop=True)
        if meta.empty:
            raise ValueError(f"No posts in split '{split}'")
        if max_posts is not None and max_posts < len(meta):
            labels = pd.read_parquet(base / "data" / "splits" / dataset_name / "labels.parquet")
            meta = meta.merge(labels[['post_id', 'label']], on='post_id', how='left')
            meta = meta.sort_values('label', ascending=False).head(max_posts).drop(columns=['label']).reset_index(drop=True)
        
        ev = pd.read_parquet(base / "data" / "processed" / dataset_name / "events.parquet",
                             columns=["post_id", "user", "time"])
        pos = pd.Index(meta["post_id"]).get_indexer(ev["post_id"])
        ev_time = ev["time"].to_numpy(np.int64)
        ok = (pos >= 0)
        ok[ok] = ev_time[ok] <= meta["t_end"].to_numpy(np.int64)[pos[ok]]
        pos, ev_time, ev_user = pos[ok], ev_time[ok], ev["user"].to_numpy()[ok]
        order = np.lexsort((ev_time, pos))
        pos, ev_time, ev_user = pos[order], ev_time[order], ev_user[order]
        ptr = np.searchsorted(pos, np.arange(len(meta) + 1), side="left").astype(np.int64)
        log.info("PostSet: %d '%s' posts, %d events", len(meta), split, len(ev_time))
        return cls(engine, meta, ptr, ev_time, ev_user)

    def _slice(self, i):
        return slice(self.ptr[i], self.ptr[i + 1])

    def state(self, i: int, mode: str = "obs") -> PostState:
        """Starting state of post i.

        mode "obs"   everything seen up to t_obs is active (Case 1, and Case 2 "addon")
        mode "cold"  only the author, active at t0; nothing is observed (Case 2 "cold")
        """
        bs = self.engine.bin_seconds
        b0, b_obs, b_end = int(self.t0[i] // bs), int(self.t_obs[i] // bs), int(self.t_end[i] // bs)
        a = int(self.author[i])
        if mode == "cold":
            nodes = np.array([a] if a >= 0 else [], np.int32)
            bins = np.full(len(nodes), b0, np.int64)
            return PostState(nodes, bins, b0 - 1, b0, b_end, 0)
        if mode != "obs":
            raise ValueError("mode must be 'obs' or 'cold'")
        sl = self._slice(i)
        times, users = self.ev_time[sl], self.ev_user[sl]
        k = int(np.searchsorted(times, self.t_obs[i], side="right"))
        users, times = users[:k], times[:k]
        keep = users >= 0                       # users outside the graph cannot spread
        nodes, bins = users[keep], times[keep] // bs
        if a >= 0 and not (nodes == a).any():   # author without an event of their own (owner_file rule)
            nodes, bins = np.append(nodes, a), np.append(bins, b0)
        return PostState(nodes.astype(np.int32), bins.astype(np.int64), b_obs, b_obs, b_end, k)

    def first_adopters_after(self, i: int, after_time: int, exclude: np.ndarray, n: int) -> np.ndarray:
        """First n distinct graph users who actually adopted after `after_time` (evaluation baseline only;
        uses events up to t_end. Ties inside one bin are in arbitrary order)."""
        sl = self._slice(i)
        times, users = self.ev_time[sl], self.ev_user[sl]
        users = users[times > after_time]
        users = users[(users >= 0) & ~np.isin(users, exclude)]
        _, first = np.unique(users, return_index=True)
        return users[np.sort(first)][:n]


# ============================================================================ #
# Case 1: popularity prediction
# ============================================================================ #
def predict_popularity(engine: CascadeEngine, posts: PostSet, *, n_worlds: int = N_WORLDS,
                       external_per_bin: float = EXTERNAL_PER_BIN, seed: int = SEED,
                       log_every: int = 1000) -> pd.DataFrame:
    """Predict popularity at t_end for every post, using only what is known at t_obs.

    Columns: post_id, split, observed_count, pred_label, pred_growth, pred_p10, pred_p50, pred_p90,
             sim_new_mean, sim_new_std
    pred_label = observed_count + simulated new adopters (+ external_per_bin * remaining bins).
    Labels are never read here.
    """
    n = len(posts)
    out = {k: np.zeros(n) for k in ("observed_count", "pred_label", "pred_growth", "pred_p10",
                                    "pred_p50", "pred_p90", "sim_new_mean", "sim_new_std")}
    t_start = time.time()
    for i in range(n):
        st = posts.state(i, "obs")
        vals = engine.eval_sets(st.init_nodes, st.init_bins, st.cond_bin, st.vis_bin, st.end_bin,
                                None, n_worlds, salt=(seed << 20) + i)[0].astype(np.float64)
        ext = external_per_bin * max(st.end_bin - st.cond_bin, 0)
        q10, q50, q90 = np.quantile(vals, [0.1, 0.5, 0.9])
        mean = vals.mean()
        out["observed_count"][i] = st.observed_count
        out["sim_new_mean"][i], out["sim_new_std"][i] = mean, vals.std()
        out["pred_growth"][i] = mean + ext
        out["pred_label"][i] = st.observed_count + mean + ext
        out["pred_p10"][i], out["pred_p50"][i], out["pred_p90"][i] = (st.observed_count + ext + q for q in (q10, q50, q90))
        if log_every and (i + 1) % log_every == 0:
            log.info("popularity: %d / %d posts (%.1f posts/s)", i + 1, n, (i + 1) / (time.time() - t_start))
    df = pd.DataFrame({"post_id": posts.meta["post_id"].to_numpy(), "split": posts.meta["split"].to_numpy(), **out})
    df["observed_count"] = df["observed_count"].astype(np.int64)
    return df


# ============================================================================ #
# Case 2: seed selection
# ============================================================================ #
class SeedObjective:
    """Expected spread of (initial users + seeds) on a FIXED set of worlds (common random numbers)."""

    def __init__(self, engine: CascadeEngine, state: PostState, n_worlds: int, salt: int):
        self.engine, self.state, self.n_worlds, self.salt = engine, state, int(n_worlds), int(salt)
        self.n_evals = 0

    def _run(self, sets):
        s = self.state
        self.n_evals += len(sets)
        return self.engine.eval_sets(s.init_nodes, s.init_bins, s.cond_bin, s.vis_bin, s.end_bin,
                                     sets, self.n_worlds, self.salt)

    def vals(self, seeds) -> np.ndarray:
        """n_new in every world for one seed set (array of length n_worlds)."""
        seeds = list(seeds) or [-1]
        return self._run(np.asarray([seeds], np.int32))[0]

    def value(self, seeds) -> float:
        return float(self.vals(seeds).mean())

    def values(self, base, cands) -> np.ndarray:
        cands = np.asarray(cands, np.int32)
        sets = np.full((len(cands), len(base) + 1), -1, np.int32)
        if len(base):
            sets[:, :len(base)] = np.asarray(base, np.int32)
        sets[:, -1] = cands
        return self._run(sets).mean(axis=1)


def _get_selector(name: str):
    try:
        from src.diffusion import selectors
    except ImportError as exc:
        raise ImportError("choose_seeds needs src/diffusion/selectors.py (greedy, celf); see the contract "
                          "in the variants.py docstring") from exc
    if not hasattr(selectors, name):
        raise ValueError(f"selectors.py has no function '{name}'")
    return getattr(selectors, name)


def choose_seeds(engine: CascadeEngine, posts: PostSet, *, mode: str = SEED_MODE, k_values=K_VALUES,
                 methods=SELECT_METHODS, baselines: bool = True, n_candidates: int = N_CANDIDATES,
                 n_select_worlds: int = N_SELECT_WORLDS, n_eval_worlds: int = N_EVAL_WORLDS,
                 seed: int = SEED, log_every: int = 50) -> pd.DataFrame:
    """For every post: pick up to max(k_values) extra seeds with each method and score them.

    mode "cold"   author posts at t0, seeds start at t0 (nothing observed)
    mode "addon"  seeds are added right after t_obs on top of what was observed (never an observed adopter)

    Seeds are CHOSEN on one set of worlds and SCORED on a different, independent set, so the reported
    spread is not inflated by picking the luckiest users. Greedy/CELF results for k=1,2,3 are prefixes
    of one run with kmax seeds.

    Columns: post_id, mode, method, k, seeds (raw user ids, comma separated), spread, spread_se,
             base_spread, gain, select_spread, n_evals
      spread         expected n_new with the seeds counted as new adopters (scored on fresh worlds)
      base_spread    the same with the author alone; gain = spread - base_spread
      select_spread  the optimistic value seen during selection (spread - select_spread ~ selection bias)
    Methods: the ones in `methods`, plus baselines 'random', 'degree' (visible out-degree),
    'score' (p_u * degree) and, for reference, 'actual' (the first real adopters after the seed start).
    """
    if mode not in ("cold", "addon"):
        raise ValueError("mode must be 'cold' or 'addon'")
    k_values = sorted({int(k) for k in k_values})
    kmax = k_values[-1]
    selectors_ = {m: _get_selector(m) for m in methods}
    state_mode = "cold" if mode == "cold" else "obs"
    rows = []
    t_start = time.time()

    def add_row(pid, method, k, seeds, obj_eval, base_mean, select_spread=np.nan, n_evals=0):
        v = obj_eval.vals(seeds).astype(np.float64)
        rows.append({
            "post_id": pid, "mode": mode, "method": method, "k": k,
            "seeds": ",".join(str(x) for x in engine.nodes[np.asarray(seeds, np.int64)]),
            "spread": v.mean(), "spread_se": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0,
            "base_spread": base_mean, "gain": v.mean() - base_mean,
            "select_spread": select_spread, "n_evals": n_evals,
        })

    for i in range(len(posts)):
        pid = posts.meta["post_id"].iloc[i]
        st = posts.state(i, state_mode)
        salt = (seed << 20) + 2 * i
        obj_sel = SeedObjective(engine, st, n_select_worlds, salt)
        obj_eval = SeedObjective(engine, st, n_eval_worlds, salt + 1)
        base_mean = float(obj_eval.vals([]).mean())

        want = n_candidates + len(st.init_nodes)
        keep_free = lambda a: a[~np.isin(a, st.init_nodes)]
        cand = keep_free(engine.pool(st.vis_bin, "score", want))[:n_candidates]
        if len(cand) < kmax:
            log.warning("post %s: only %d candidates (< %d); skipped", pid, len(cand), kmax)
            continue

        for m, fn in selectors_.items():
            res = fn(obj_sel, cand, kmax)
            for k in k_values:
                add_row(pid, m, k, list(res.seeds[:k]), obj_eval, base_mean,
                        select_spread=float(res.values[k - 1]), n_evals=int(res.n_evals))

        if baselines:
            rng = np.random.default_rng([seed, i])
            by_deg = keep_free(engine.pool(st.vis_bin, "degree", want))
            picks = {"random": rng.choice(cand, kmax, replace=False), "degree": by_deg[:kmax], "score": cand[:kmax]}
            after = int(posts.t0[i]) - 1 if mode == "cold" else int(posts.t_obs[i])
            picks["actual"] = posts.first_adopters_after(i, after, st.init_nodes, kmax)
            for name, seeds in picks.items():
                for k in k_values:
                    if len(seeds) >= k:
                        add_row(pid, name, k, list(seeds[:k]), obj_eval, base_mean)
        if log_every and (i + 1) % log_every == 0:
            log.info("seeds: %d / %d posts (%.2f posts/s)", i + 1, len(posts), (i + 1) / (time.time() - t_start))
    return pd.DataFrame(rows)


# ============================================================================ #
# Self-test on tiny synthetic graphs (the sanity checks from the design discussion)
# ============================================================================ #
def selftest() -> None:
    def star(n, pmf, p, eb=0, strict=True):
        src = np.zeros(n, np.int64)
        dst = np.arange(1, n + 1)
        return CascadeEngine.from_arrays(src, dst, np.full(n, eb), pmf, 86400, strict=strict,
                                         p_global=p, n_chunks=2)

    a = np.array([0], np.int32)
    zero = np.array([0], np.int64)
    ok = lambda name, cond: print(("PASS  " if cond else "FAIL  ") + name) or cond
    results = []

    # 1. p = 0: nothing spreads
    e = star(100, [0.5, 0.5], 0.0)
    results.append(ok("p=0 -> spread 0", e.eval_sets(a, zero, -1, 0, 5, None, 50)[0].sum() == 0))

    # 2. chain, p=1, lag exactly 1 bin: spread equals the number of hops inside the deadline
    chain = CascadeEngine.from_arrays(np.arange(5), np.arange(1, 6), np.zeros(5), [0.0, 1.0], 86400,
                                      strict=True, p_global=1.0, n_chunks=2)
    r3 = chain.eval_sets(a, zero, -1, 0, 3, None, 4)
    r9 = chain.eval_sets(a, zero, -1, 0, 9, None, 4)
    results.append(ok("chain, deadline 3 bins -> 3 new users", (r3 == 3).all()))
    results.append(ok("chain, long deadline -> all 5 reachable users", (r9 == 5).all()))

    # 3. model frequencies: p=0.3, same bin with prob 0.5
    N = 20000
    e = star(N, [0.5, 0.5], 0.3)
    f = e.eval_sets(a, zero, -1, 0, 0, None, 1)[0, 0] / N
    results.append(ok(f"p=0.3, lag 0 w.p. 0.5, deadline = same bin -> share {f:.3f} ~ 0.15", abs(f - 0.15) < 0.01))
    f = e.eval_sets(a, zero, -1, 0, 1, None, 1)[0, 0] / N
    results.append(ok(f"... deadline one bin later -> share {f:.3f} ~ 0.30", abs(f - 0.30) < 0.01))

    # 4. conditioning: parent adopted in bin 0, bins 0..2 observed, child not adopted yet, uniform lag 0..4
    #    -> the lag is 3 or 4 with equal probability
    e = star(N, [0.2] * 5, 1.0)
    f3 = e.eval_sets(a, zero, 2, 2, 3, None, 1)[0, 0] / N
    f4 = e.eval_sets(a, zero, 2, 2, 4, None, 1)[0, 0] / N
    results.append(ok(f"conditioning: by bin 3 share {f3:.3f} ~ 0.5, by bin 4 share {f4:.3f} ~ 1.0",
                      abs(f3 - 0.5) < 0.02 and f4 > 0.999))
    fc = e.eval_sets(a, zero, -1, 0, 3, None, 1)[0, 0] / N
    results.append(ok(f"no observation (cold) by bin 3: share {fc:.3f} ~ 0.8", abs(fc - 0.8) < 0.02))

    # 5. edge visibility: created after vis_bin / after the source adopted -> unusable; static graph -> usable
    e = star(1000, [0.0, 1.0], 1.0, eb=5)
    results.append(ok("edge created after vis_bin is not used", e.eval_sets(a, zero, -1, 2, 9, None, 3).sum() == 0))
    results.append(ok("edge created after the source adopted is not used", e.eval_sets(a, zero, -1, 9, 9, None, 3).sum() == 0))
    e = star(1000, [0.0, 1.0], 1.0, eb=5, strict=False)
    results.append(ok("static graph ignores edge times", (e.eval_sets(a, zero, -1, 2, 9, None, 3) == 1000).all()))

    # 6. random graph: determinism, per-world monotonicity in the seed set, CRN across calls
    rng = np.random.default_rng(1)
    n, m = 3000, 15000
    g = CascadeEngine.from_arrays(rng.integers(0, n, m), rng.integers(0, n, m), np.zeros(m), [0.4, 0.3, 0.2, 0.1],
                                  86400, strict=True, p_global=0.25, n_chunks=3)
    init = np.array([0], np.int32)
    r1 = g.eval_sets(init, zero, -1, 0, 6, None, 100, salt=7)
    r2 = g.eval_sets(init, zero, -1, 0, 6, None, 100, salt=7)
    results.append(ok("same salt -> identical worlds", (r1 == r2).all()))
    sets = np.array([[5, -1, -1], [5, 17, -1], [5, 17, 42]], np.int32)
    r = g.eval_sets(init, zero, -1, 0, 6, sets, 100, salt=7)
    results.append(ok("adding a seed never lowers the spread in any world", (r[1] >= r[0]).all() and (r[2] >= r[1]).all()))
    results.append(ok("chunking does not change results", (g.eval_sets(init, zero, -1, 0, 6, sets, 100, salt=7) == r).all()))
    print("ALL PASSED" if all(results) else "SOME TESTS FAILED")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="variants.py: engine self-test (the runner is simulator.py)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        ap.print_help()