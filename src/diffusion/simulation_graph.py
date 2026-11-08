"""Core graph processing and Numba simulation kernels for diffusion."""
from __future__ import annotations

import heapq
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

try:
    from numba import njit
    from numba.typed import List
except ImportError as exc:
    raise ImportError("simulation_graph.py needs numba: pip install numba") from exc

log = logging.getLogger(__name__)

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_K1 = np.uint64(0xD6E8FEB86659FD93)
_K2 = np.uint64(0xA0761D6478BD642F)
_K3 = np.uint64(0xE7037ED1A0B428DB)
_K4 = np.uint64(0x8EBC6AF09C88C6E3)
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_S33 = np.uint64(33)
_S11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0


@njit(cache=True)
def _fmix(x):
    x ^= x >> _S33
    x *= _M1
    x ^= x >> _S33
    x *= _M2
    x ^= x >> _S33
    return x


@njit(cache=True)
def _hash3(a, b, c):
    return _fmix(np.uint64(a) * _K1 + np.uint64(b) * _K2 + c * _K3 + _K4)


@njit(cache=True)
def _unif(h):
    return (h >> _S11) * _INV53


@njit(cache=True)
def _precompute_edge_data(indptr, nbr, raw, seed_p):
    edge_base = np.empty(len(nbr), dtype=np.float32)
    nbr_raw = np.empty(len(nbr), dtype=np.int64)
    for u in range(len(raw)):
        ru = raw[u]
        for e in range(indptr[u], indptr[u + 1]):
            rv = raw[nbr[e]]
            nbr_raw[e] = rv
            edge_base[e] = _unif(_hash3(ru, rv, seed_p))
    return edge_base, nbr_raw


@njit(cache=True)
def _calc_timed_degrees(cands, indptr, ctime, t_obs):
    deg = np.empty(len(cands), dtype=np.int64)
    for i in range(len(cands)):
        u = cands[i]
        lo, hi = indptr[u], indptr[u + 1]
        while lo < hi:
            m = (lo + hi) // 2
            if ctime[m] <= t_obs:
                lo = m + 1
            else:
                hi = m
        deg[i] = lo - indptr[u]
    return deg


@njit(cache=True)
def _get_base_dist(u, base_nodes, base_dists):
    idx = np.searchsorted(base_nodes, u)
    if idx < len(base_nodes) and base_nodes[idx] == u:
        return base_dists[idx]
    return np.inf


@njit(cache=True)
def _expand(u, dt, base, heap, indptr, nbr, ctime, raw, edge_base, nbr_raw, t_obs, horizon, rate, pmax, seed_p,
            run_key, timed, stamp, fstamp, dist, token):
    lo, hi = indptr[u], indptr[u + 1]
    if timed:
        a, b = lo, hi
        while a < b:
            m = (a + b) // 2
            if ctime[m] <= t_obs:
                a = m + 1
            else:
                b = m
        hi = a
    ru = raw[u]
    q = 1.0
    if dt > 0.0:
        q = math.exp(-rate * dt)
    for e in range(lo, hi):
        v = nbr[e]
        if fstamp[v] == token:
            continue
        rv = nbr_raw[e]
        pe = edge_base[e] * pmax
        if dt > 0.0:
            den = 1.0 - pe * (1.0 - q)
            pe = pe * q / den if den > 0.0 else 0.0
        h = _hash3(ru, rv, run_key)
        if _unif(h) >= pe:
            continue
        arr = base - math.log1p(-_unif(_fmix(h + _GOLD))) / rate
        if arr > horizon:
            continue
        if stamp[v] != token or arr < dist[v]:
            stamp[v] = token
            dist[v] = arr
            heapq.heappush(heap, (arr, v))


@njit(cache=True)
def precompute_baselines(indptr, nbr, ctime, raw, edge_base, nbr_raw, act, act_dt, t_obs, horizon, rate, pmax,
                         seed_p, seed_run, offset, runs, timed, stamp, fstamp, dist, token0):
    out_nodes, out_dists = List(), List()
    reached = np.empty(len(raw), dtype=np.int64)
    for r in range(runs):
        token = token0 + r
        run_key = _fmix(np.uint64(seed_run) * _K1 + np.uint64(offset + r) * _K2 + _K4)
        heap = [(0.0, 0)]
        heap.pop()
        rcount = 0
        for i in range(len(act)):
            a = act[i]
            stamp[a] = token
            fstamp[a] = token
            dist[a] = 0.0
            reached[rcount] = a
            rcount += 1
        for i in range(len(act)):
            _expand(act[i], act_dt[i], 0.0, heap, indptr, nbr, ctime, raw, edge_base, nbr_raw, t_obs, horizon,
                    rate, pmax, seed_p, run_key, timed, stamp, fstamp, dist, token)
        while len(heap) > 0:
            d, u = heapq.heappop(heap)
            if fstamp[u] == token or d > dist[u]:
                continue
            fstamp[u] = token
            reached[rcount] = u
            rcount += 1
            _expand(u, 0.0, d, heap, indptr, nbr, ctime, raw, edge_base, nbr_raw, t_obs, horizon,
                    rate, pmax, seed_p, run_key, timed, stamp, fstamp, dist, token)
        run_nodes = reached[:rcount].copy()
        run_dists = np.empty(rcount, dtype=np.float64)
        for i in range(rcount):
            run_dists[i] = dist[run_nodes[i]]
        order = np.argsort(run_nodes)
        out_nodes.append(run_nodes[order])
        out_dists.append(run_dists[order])
    return out_nodes, out_dists


@njit(cache=True)
def delta_spread(indptr, nbr, ctime, raw, edge_base, nbr_raw, seeds, t_obs, horizon, rate, pmax, seed_p, seed_run,
                 offset, runs, timed, stamp, fstamp, dist, token0, base_nodes_list, base_dists_list):
    total = 0.0
    for r in range(runs):
        token = token0 + r
        run_key = _fmix(np.uint64(seed_run) * _K1 + np.uint64(offset + r) * _K2 + _K4)
        heap = [(0.0, 0)]
        heap.pop()
        base_nodes = base_nodes_list[r]
        base_dists = base_dists_list[r]
        count = len(base_nodes)
        for i in range(len(seeds)):
            s = seeds[i]
            b_dist = _get_base_dist(s, base_nodes, base_dists)
            if b_dist > 0.0:
                if b_dist == np.inf and stamp[s] != token:
                    count += 1
                stamp[s] = token
                dist[s] = 0.0
                heapq.heappush(heap, (0.0, s))
        while len(heap) > 0:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if fstamp[u] == token:
                continue
            fstamp[u] = token
            lo, hi = indptr[u], indptr[u + 1]
            if timed:
                a, b = lo, hi
                while a < b:
                    m = (a + b) // 2
                    if ctime[m] <= t_obs:
                        a = m + 1
                    else:
                        b = m
                hi = a
            ru = raw[u]
            for e in range(lo, hi):
                v = nbr[e]
                if fstamp[v] == token:
                    continue
                rv = nbr_raw[e]
                pe = edge_base[e] * pmax
                h = _hash3(ru, rv, run_key)
                if _unif(h) >= pe:
                    continue
                delay = -math.log1p(-_unif(_fmix(h + _GOLD))) / rate
                arr = d + delay
                if arr > horizon:
                    continue
                b_dist = _get_base_dist(v, base_nodes, base_dists)
                best_known = b_dist
                if stamp[v] == token and dist[v] < best_known:
                    best_known = dist[v]
                if arr < best_known:
                    if b_dist == np.inf and stamp[v] != token:
                        count += 1
                    stamp[v] = token
                    dist[v] = arr
                    heapq.heappush(heap, (arr, v))
        total += count
    return total / runs


def _to_idx(universe: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    if len(universe) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(universe, ids), len(universe) - 1)
    return np.where(universe[pos] == ids, pos, -1).astype(np.int64)


def _edge_time_available(canon_dir: Path) -> bool:
    manifest = canon_dir / "manifest.json"
    if manifest.exists():
        return bool(json.loads(manifest.read_text()).get("edge_time_available", True))
    return True


def build_graph(src_raw, dst_raw, ct_raw, extra_users, timed: bool, seed_p: int) -> dict:
    universe = np.unique(np.concatenate([src_raw, dst_raw, extra_users]))
    n = len(universe)
    if n >= 3_000_000_000:
        raise OverflowError("too many users for int64 pair keys")
    s, d = _to_idx(universe, src_raw), _to_idx(universe, dst_raw)
    keep = s != d
    s, d = s[keep], d[keep]
    ct = ct_raw[keep] if timed else None
    key = s * n + d
    o = np.lexsort((ct, key)) if timed else np.argsort(key, kind="stable")
    _, first = np.unique(key[o], return_index=True)
    sel = o[first]
    s, d = s[sel], d[sel]
    ct = ct[sel] if timed else np.zeros(len(s), dtype=np.int64)
    order = np.lexsort((ct, s)) if timed else np.argsort(s, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(s, minlength=n), out=indptr[1:])
    deg = np.diff(indptr)
    pool = np.argsort(-deg, kind="stable")
    pool = pool[deg[pool] > 0]
    nbr = d[order].astype(np.int64)
    raw = universe.astype(np.int64)
    edge_base, nbr_raw = _precompute_edge_data(indptr, nbr, raw, np.uint64(seed_p))
    return dict(raw=raw, indptr=indptr, nbr=nbr,
                ctime=ct[order], cand_pool=pool.astype(np.int64), timed=bool(timed),
                edge_base=edge_base, nbr_raw=nbr_raw, n=n)


_WORKER: dict = {}


def _get_graph(path: str) -> dict:
    if _WORKER.get("path") != path:
        G = load(path, mmap_mode="r")
        n = G["n"]
        _WORKER.clear()
        _WORKER.update(path=path, G=G, token=1,
                       stamp=np.zeros(n, np.int64), fstamp=np.zeros(n, np.int64),
                       dist=np.zeros(n, np.float64))
    return _WORKER["G"]


class PostSim:
    def __init__(self, G, cfg, t_obs, t_end, act, act_dt):
        self.G, self.cfg = G, cfg
        self.t_obs, self.horizon = float(t_obs), float(t_end - t_obs)
        self.act, self.act_dt = np.asarray(act, np.int64), np.asarray(act_dt, np.float64)
        self._baselines = {}

    def _get_baseline(self, runs, offset):
        key = (runs, offset)
        if key not in self._baselines:
            G, c, w = self.G, self.cfg, _WORKER
            nodes, dists = precompute_baselines(
                G["indptr"], G["nbr"], G["ctime"], G["raw"], G["edge_base"], G["nbr_raw"], self.act, self.act_dt,
                self.t_obs, self.horizon, c["rate"], c["p_max"], np.uint64(c["seed_p"]),
                c["seed"], int(offset), int(runs), G["timed"], w["stamp"], w["fstamp"], w["dist"], w["token"]
            )
            w["token"] += int(runs)
            self._baselines[key] = (nodes, dists)
        return self._baselines[key]

    def spread(self, seeds=(), runs: int = 200, offset: int = 0) -> float:
        nodes, dists = self._get_baseline(runs, offset)
        if len(seeds) == 0:
            return float(sum(len(n) for n in nodes)) / runs
        G, c, w = self.G, self.cfg, _WORKER
        seeds_arr = np.asarray(list(seeds), dtype=np.int64)
        out = delta_spread(
            G["indptr"], G["nbr"], G["ctime"], G["raw"], G["edge_base"], G["nbr_raw"], seeds_arr,
            self.t_obs, self.horizon, c["rate"], c["p_max"], np.uint64(c["seed_p"]),
            c["seed"], int(offset), int(runs), G["timed"], w["stamp"], w["fstamp"],
            w["dist"], w["token"], nodes, dists
        )
        w["token"] += int(runs)
        return float(out)

    def candidates(self, n: int) -> list:
        pool = self.G["cand_pool"]
        is_active = np.zeros(self.G["n"], dtype=np.bool_)
        is_active[self.act] = True
        cands = pool[~is_active[pool]]
        cands = cands[: max(20 * n, 2000)]
        if len(cands) == 0 or n <= 0:
            return []
        if self.G["timed"]:
            deg = _calc_timed_degrees(cands, self.G["indptr"], self.G["ctime"], self.t_obs)
        else:
            deg = self.G["indptr"][cands + 1] - self.G["indptr"][cands]
        order = np.argsort(-deg, kind="stable")[:n]
        return [int(x) for x in cands[order][deg[order] > 0]]

    def raw_ids(self, local_ids) -> list:
        return [int(self.G["raw"][i]) for i in local_ids]


def load_posts(base: Path, dataset: str, split: str, n_posts: int, seed: int, allow_static: bool):
    canon, splits = base / "data" / "processed" / dataset, base / "data" / "splits" / dataset
    timed = _edge_time_available(canon)
    if not timed and not allow_static:
        raise ValueError("Static graph (no edge times): pass --allow-static-graph to accept leak risk.")
    meta = pd.read_parquet(splits / "split_meta.parquet")
    meta = meta[meta["split"] == split].reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"No posts in split '{split}'")
    if n_posts and n_posts < len(meta):
        keep = np.sort(np.random.default_rng(seed).choice(len(meta), size=n_posts, replace=False))
        meta = meta.iloc[keep].reset_index(drop=True)
    t_obs, t_end = meta["t_obs"].to_numpy(), meta["t_end"].to_numpy()
    ev = pd.read_parquet(canon / "events.parquet", columns=["post_id", "user", "time"])
    ev = ev[ev["post_id"].isin(meta["post_id"])]
    authors_ev = meta[["post_id", "author", "t0"]].rename(columns={"author": "user", "t0": "time"})
    ev = pd.concat([ev, authors_ev], ignore_index=True)
    ev = ev.sort_values(["post_id", "time"], kind="stable").drop_duplicates(subset=["post_id", "user"], keep="first")
    code = pd.Index(meta["post_id"]).get_indexer(ev["post_id"])
    keep = ev["time"].to_numpy() <= t_obs[code]
    ev_user_raw, ev_time, ev_code = ev["user"].to_numpy()[keep], ev["time"].to_numpy()[keep], code[keep].astype(np.int64)
    cols = ["source", "target"] + (["created_at"] if timed else [])
    edges = pd.read_parquet(canon / "edges.parquet", columns=cols)
    if timed:
        edges = edges[edges["created_at"] <= t_obs.max()]
    seed_p = int(seed) * 7919 + 12345
    G = build_graph(edges["source"].to_numpy(), edges["target"].to_numpy(),
                    edges["created_at"].to_numpy() if timed else None, ev_user_raw, timed, seed_p)
    del edges
    n = len(G["raw"])
    ev_user = _to_idx(G["raw"], ev_user_raw)
    o = np.lexsort((ev_time, ev_code))
    ev_code, ev_user, ev_time = ev_code[o], ev_user[o], ev_time[o]
    _, first = np.unique(ev_code * n + ev_user, return_index=True)
    first.sort()
    a_post, a_user, a_time = ev_code[first], ev_user[first], ev_time[first]
    ptr = np.searchsorted(a_post, np.arange(len(meta) + 1))
    tasks, keep_rows = [], []
    for i in range(len(meta)):
        a, b = int(ptr[i]), int(ptr[i + 1])
        if a == b:
            continue
        keep_rows.append(i)
        tasks.append((meta["post_id"].iloc[i], float(t_obs[i]), float(t_end[i]), a_user[a:b],
                      float(t_obs[i]) - a_time[a:b].astype(np.float64)))
    meta = meta.iloc[keep_rows].reset_index(drop=True)
    return G, tasks, meta