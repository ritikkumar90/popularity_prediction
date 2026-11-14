"""Fit a discrete-time independent-cascade model on the TRAIN split only.

Model
-----
Time is cut into bins of `bin_seconds` (auto-detected from the event timestamps: epoch-day
data gives 86400). When user u adopts a post in bin s, every follower v (edge u -> v) is
exposed. The chance that v adopts at lag d bins (d = 0 means the same bin) is

        P(v adopts at s + d | exposed by u)  =  p_u * q[d],      d = 0..D,   sum_d q[d] = 1

    p_u   influence of source u (shrunk towards the global value p_global)
    q     delay pmf (shared by all edges)

Estimation (EM on a Poisson likelihood)
---------------------------------------
* Candidate pairs: u and v both adopted the post, edge u -> v existed when u adopted
  (unless --ignore-edge-time), and v did not adopt before u.
* E-step: if v has several candidate parents, credit is split in proportion to p_u * q[d].
* M-step: p_u  = (credited + prior*p_global) / effective_exposure
          q[d] = credited_at_lag_d / exposure_at_risk_at_lag_d   (right-censoring correction)
  Exposures count EVERY follower of an adopter (adopted or not) whose window still allows lag d.
* Adopters with no candidate parent are reported as `orphan_share` (external adoption).

Only train posts are used, cut at each post's own t_end. The post author is the root at t0
(its own event rows are replaced by one synthetic event at t0, so the root is never a child).

Edge direction follows the canonical tables: information flows source -> target.
If results look odd, rerun with --reverse-edges and compare `orphan_share`
(the correct direction explains more adoptions, so its orphan_share is lower).

Usage
-----
    python -m src.diffusion.fit --dataset flickr --base-path .
    python -m src.diffusion.fit --max-train-posts 2000          # smoke test
    python -m src.diffusion.fit --help

Writes  <base-path>/<output-dir>/<dataset>/fit/fitted_params.json
        <base-path>/<output-dir>/<dataset>/fit/source_influence.parquet   (user, p, exposure columns)
Reads   data/processed/<dataset>/{events,edges}.parquet, data/splits/<dataset>/split_meta.parquet
        split_meta needs: post_id, author, t0, t_end, split.  Labels are never read.

Memory note: the exposure table is n_adopters x (D+1) float64. Use a larger --bin-seconds or a
smaller --max-delay-bins if it is too big.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pure-python fallback: correct but very slow
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

log = logging.getLogger(__name__)

# ============================================================================ #
# SETTINGS (each one has a command-line flag)
# ============================================================================ #
OUTPUT_DIR = "outputs"
BIN_SECONDS = None          # None = auto-detect from event timestamps
MAX_DELAY_BINS = None       # None = longest (t0 -> t_end) window among train posts
N_ITER = 50
TOL = 1e-6                  # relative change of the log-likelihood to stop
PRIOR_EXPOSURES = 50.0      # shrinkage strength of p_u towards p_global (in effective exposures)
P_MIN, P_MAX = 1e-6, 0.95
IGNORE_EDGE_TIME = False    # False: an edge must exist when the source adopts (leak-free)
REVERSE_EDGES = False
SEED = 0
_NO_EDGE_TIME = -(1 << 60)


# ============================================================================ #
# Helpers
# ============================================================================ #
def _detect_bin_seconds(times: np.ndarray) -> int:
    """Largest common divisor of the event times (capped at 1 day, floored at 1 hour)."""
    if len(times) > 5_000_000:
        times = times[np.random.default_rng(0).choice(len(times), 5_000_000, replace=False)]
    g = int(np.gcd.reduce(times.astype(np.int64)))
    return int(min(max(g, 3600), 86400))


def _dense_lookup(sorted_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position of each value in sorted_ids, or -1 when absent."""
    idx = np.searchsorted(sorted_ids, values)
    idx_c = np.minimum(idx, len(sorted_ids) - 1)
    return np.where(sorted_ids[idx_c] == values, idx_c, -1)


# ============================================================================ #
# Kernel: one pass over every (post, adopter, follower)
# ============================================================================ #
@njit(cache=True)
def _build_pairs(ev_ptr, ev_user, ev_bin, post_end_bin, post_author,
                 indptr, indices, cbin, D, strict, N, stamp, abin, evidx, cap):
    """Return candidate pairs (source, child event, lag) and fill N[u, T] = #valid exposures
    of source u whose remaining window is T bins (capped at D)."""
    p_src = np.empty(cap, np.int32)
    p_child = np.empty(cap, np.int64)
    p_lag = np.empty(cap, np.int32)
    n = 0
    n_posts = len(ev_ptr) - 1
    for j in range(n_posts):
        a = ev_ptr[j]
        b = ev_ptr[j + 1]
        for i in range(a, b):
            u = ev_user[i]
            stamp[u] = j
            abin[u] = ev_bin[i]
            evidx[u] = i
        end_bin = post_end_bin[j]
        author = post_author[j]
        for i in range(a, b):
            u = ev_user[i]
            s_u = ev_bin[i]
            T = end_bin - s_u
            if T < 0:
                continue
            Tc = T if T < D else D
            for k in range(indptr[u], indptr[u + 1]):
                if strict and cbin[k] > s_u:
                    continue                      # edge did not exist yet
                v = indices[k]
                if v >= 0 and stamp[v] == j:      # v adopted this post
                    if v == author:
                        continue
                    d = abin[v] - s_u
                    if d < 0:
                        continue                  # v was already active: no exposure
                    N[u, Tc] += 1.0
                    if d <= D:
                        if n == cap:
                            cap = cap * 2
                            t1 = np.empty(cap, np.int32)
                            t2 = np.empty(cap, np.int64)
                            t3 = np.empty(cap, np.int32)
                            t1[:n] = p_src[:n]
                            t2[:n] = p_child[:n]
                            t3[:n] = p_lag[:n]
                            p_src = t1
                            p_child = t2
                            p_lag = t3
                        p_src[n] = u
                        p_child[n] = evidx[v]
                        p_lag[n] = d
                        n += 1
                else:                             # v never adopted this post (or is outside the data)
                    N[u, Tc] += 1.0
    return p_src[:n], p_child[:n], p_lag[:n]


# ============================================================================ #
# EM
# ============================================================================ #
def _em(p_src, child_inv, n_children, p_lag, N_ge, n_iter, tol, prior):
    n_nodes, D1 = N_ge.shape

    # initial values
    atrisk0 = N_ge.sum(axis=0)
    q = np.bincount(p_lag, minlength=D1) / np.maximum(atrisk0, 1.0)
    q = np.maximum(q, 1e-12)
    q /= q.sum()
    p_bar = float(np.clip(len(p_src) / max(N_ge[:, 0].sum(), 1.0), P_MIN, P_MAX))
    p = np.full(n_nodes, p_bar)

    ll_prev = -np.inf
    ll = -np.inf
    for it in range(n_iter):
        w = p[p_src] * q[p_lag]
        denom = np.bincount(child_inv, weights=w, minlength=n_children)
        r = w / denom[child_inv]                                   # E-step
        succ_src = np.bincount(p_src, weights=r, minlength=n_nodes)
        succ_lag = np.bincount(p_lag, weights=r, minlength=D1)

        eff = N_ge @ q                                             # effective exposure (current q)
        ll = float(np.log(denom).sum() - p @ eff)
        p_bar = float(np.clip(succ_src.sum() / max(eff.sum(), 1e-12), P_MIN, P_MAX))
        p = np.clip((succ_src + prior * p_bar) / (eff + prior), P_MIN, P_MAX)   # M-step
        risk = p @ N_ge
        q = np.maximum(succ_lag / np.maximum(risk, 1e-12), 1e-12)
        q /= q.sum()

        if it % 5 == 0:
            log.info("EM iter %3d  log-lik %.6e  p_global %.4e", it, ll, p_bar)
        if np.isfinite(ll_prev) and abs(ll - ll_prev) <= tol * abs(ll_prev):
            log.info("EM converged at iter %d", it)
            break
        ll_prev = ll

    eff = N_ge @ q
    return p, q, p_bar, ll, it + 1, succ_src, eff


# ============================================================================ #
# Main
# ============================================================================ #
def fit(
    base_path: str | Path = ".",
    dataset_name: str = "flickr",
    *,
    output_dir: str = OUTPUT_DIR,
    bin_seconds: int | None = BIN_SECONDS,
    max_delay_bins: int | None = MAX_DELAY_BINS,
    n_iter: int = N_ITER,
    tol: float = TOL,
    prior_exposures: float = PRIOR_EXPOSURES,
    ignore_edge_time: bool = IGNORE_EDGE_TIME,
    reverse_edges: bool = REVERSE_EDGES,
    max_train_posts: int | None = None,
    seed: int = SEED,
    allow_static_graph: bool = False,
) -> dict:
    base = Path(base_path)
    canon_dir = base / "data" / "processed" / dataset_name
    splits_dir = base / "data" / "splits" / dataset_name
    out_dir = base / output_dir / dataset_name / "fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = canon_dir / "manifest.json"
    timed = bool(json.loads(manifest.read_text()).get("edge_time_available", True)) if manifest.exists() else True
    if not timed and not allow_static_graph:
        raise ValueError(f"{dataset_name} has a static graph (no edge times); pass allow_static_graph=True "
                         f"to accept that exposure cannot be time-checked.")
    strict = timed and not ignore_edge_time
    if not HAVE_NUMBA:
        log.warning("numba is not installed: the pair-building pass will be extremely slow. pip install numba")

    # ---- 1. Train posts ----
    meta = pd.read_parquet(splits_dir / "split_meta.parquet")
    need = {"post_id", "author", "t0", "t_end", "split"}
    if need - set(meta.columns):
        raise ValueError(f"split_meta.parquet is missing columns {sorted(need - set(meta.columns))}")
    train = meta[meta["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise ValueError("Train split is empty.")
    if max_train_posts is not None and max_train_posts < len(train):
        keep = np.sort(np.random.default_rng(seed).choice(len(train), max_train_posts, replace=False))
        train = train.iloc[keep].reset_index(drop=True)
        log.warning("Fitting on a random subset of %d train posts (test mode).", len(train))
    n_posts = len(train)

    # ---- 2. Events: train posts, t0 <= time <= t_end, author replaced by one root event at t0 ----
    ev = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    ev = ev.merge(train[["post_id", "author", "t0", "t_end"]], on="post_id", how="inner")
    n_all = len(ev)
    ev = ev[(ev["time"] >= ev["t0"]) & (ev["time"] <= ev["t_end"]) & (ev["user"] != ev["author"])]
    log.info("Train events kept: %d of %d (dropped: before t0, after t_end, or by the author)", len(ev), n_all)
    roots = pd.DataFrame({"post_id": train["post_id"], "user": train["author"], "time": train["t0"]})
    ev = pd.concat([ev[["post_id", "user", "time"]], roots], ignore_index=True)

    pidx = pd.Index(train["post_id"]).get_indexer(ev["post_id"])
    ev_time = ev["time"].to_numpy(np.int64)
    ev_user_raw = ev["user"].to_numpy()
    order = np.lexsort((ev_user_raw, ev_time, pidx))
    pidx, ev_time, ev_user_raw = pidx[order], ev_time[order], ev_user_raw[order]
    ev_ptr = np.searchsorted(pidx, np.arange(n_posts + 1), side="left").astype(np.int64)
    n_events = len(ev_time)

    # ---- 3. Time bins ----
    if bin_seconds is None:
        bin_seconds = _detect_bin_seconds(ev_time)
        log.info("Auto-detected bin size: %d s", bin_seconds)
    ev_bin = ev_time // bin_seconds
    post_end_bin = (train["t_end"].to_numpy(np.int64) // bin_seconds)
    post_t0_bin = (train["t0"].to_numpy(np.int64) // bin_seconds)
    D = int((post_end_bin - post_t0_bin).max())
    if max_delay_bins is not None:
        D = min(D, int(max_delay_bins))
    if D < 1:
        raise ValueError("Max delay is < 1 bin: the horizon is shorter than one time bin. Use a smaller --bin-seconds.")

    # ---- 4. Dense user ids = users that adopt in the train data (authors included) ----
    nodes = np.unique(ev_user_raw)
    n_nodes = len(nodes)
    ev_user = _dense_lookup(nodes, ev_user_raw).astype(np.int32)
    post_author = _dense_lookup(nodes, train["author"].to_numpy()).astype(np.int32)
    est_gb = n_nodes * (D + 1) * 8 / 1e9
    log.info("Train posts %d | events %d | adopters %d | D = %d bins | exposure table %.2f GB",
             n_posts, n_events, n_nodes, D, est_gb)
    if est_gb > 6:
        raise ValueError(f"Exposure table would need {est_gb:.1f} GB. Use a larger --bin-seconds or smaller --max-delay-bins.")

    # ---- 5. Graph as CSR over adopter sources (all their followers kept, adopters or not) ----
    ecols = ["source", "target"] + (["created_at"] if timed else [])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=ecols)
    s_raw, t_raw = edges["source"].to_numpy(), edges["target"].to_numpy()
    if reverse_edges:
        s_raw, t_raw = t_raw, s_raw
        log.warning("Edges reversed (--reverse-edges).")
    keep = np.ones(len(edges), bool)
    if timed:
        ctime = edges["created_at"].to_numpy(np.int64)
        keep &= ctime <= int(train["t_end"].max())
    src = _dense_lookup(nodes, s_raw)
    keep &= src >= 0
    src = src[keep]
    dst = _dense_lookup(nodes, t_raw[keep]).astype(np.int32)       # -1 = never adopts in train
    cbin = (ctime[keep] // bin_seconds) if timed else np.full(len(src), _NO_EDGE_TIME, np.int64)
    del edges, s_raw, t_raw
    o = np.argsort(src, kind="stable")
    indices, cbin = dst[o], cbin[o].astype(np.int64)
    indptr = np.zeros(n_nodes + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n_nodes), out=indptr[1:])
    log.info("Edges out of train adopters: %d (%.1f%% point to a train adopter)",
             len(indices), 100.0 * float((indices >= 0).mean()) if len(indices) else 0.0)

    # ---- 6. Candidate pairs + exposure table ----
    N = np.zeros((n_nodes, D + 1), np.float64)
    stamp = np.full(n_nodes, -1, np.int64)
    abin = np.zeros(n_nodes, np.int64)
    evidx = np.zeros(n_nodes, np.int64)
    p_src, p_child, p_lag = _build_pairs(
        ev_ptr, ev_user, ev_bin.astype(np.int64), post_end_bin, post_author,
        indptr, indices, cbin, D, strict, N, stamp, abin, evidx, max(1_000_000, 2 * n_events))
    n_pairs = len(p_src)
    if n_pairs == 0:
        raise ValueError(
            "No candidate parent-child pairs found. Likely causes: edges point the wrong way "
            "(try --reverse-edges), edges were all created after the adoptions (try --ignore-edge-time), "
            "or the graph was restricted to users that never adopt together.")
    for d in range(D - 1, -1, -1):                 # N[u, d] = #exposures with remaining window >= d
        N[:, d] += N[:, d + 1]
    uniq_child, child_inv = np.unique(p_child, return_inverse=True)
    n_children = len(uniq_child)
    n_nonroot = n_events - n_posts
    orphan_share = 1.0 - n_children / max(n_nonroot, 1)
    log.info("Pairs %d | explained adoptions %d of %d | orphan share %.1f%%",
             n_pairs, n_children, n_nonroot, 100 * orphan_share)

    # ---- 7. EM ----
    p, q, p_bar, ll, iters, succ_src, eff = _em(
        p_src, child_inv, n_children, p_lag, N, n_iter, tol, prior_exposures)

    lags = np.arange(D + 1)
    mean_bins = float((q * lags).sum())
    median_bin = int(np.searchsorted(np.cumsum(q), 0.5))
    mask = eff >= 10
    pq = p[mask] if mask.any() else p
    params = {
        "model": "discrete-time IC: P(v adopts at lag d | u adopted, edge u->v) = p_u * q[d]",
        "bin_seconds": int(bin_seconds),
        "max_delay_bins": int(D),
        "delay_pmf": q.tolist(),
        "delay_mean_bins": mean_bins,
        "delay_median_bin": median_bin,
        "same_bin_mass": float(q[0]),
        "p_global": p_bar,
        "p_source_quantiles": {k: float(np.quantile(pq, v)) for k, v in
                               {"p10": .1, "p50": .5, "p90": .9, "p99": .99}.items()},
        "prior_exposures": prior_exposures,
        "orphan_share": float(orphan_share),
        "orphans_per_post": float((n_nonroot - n_children) / n_posts),
        "n_train_posts": int(n_posts),
        "n_events": int(n_events),
        "n_adopters": int(n_nodes),
        "n_pairs": int(n_pairs),
        "n_explained_adoptions": int(n_children),
        "log_likelihood": ll,
        "em_iterations": int(iters),
        "edge_time_available": timed,
        "strict_edge_timing": strict,
        "reverse_edges": bool(reverse_edges),
    }
    (out_dir / "fitted_params.json").write_text(json.dumps(params, indent=2))
    pd.DataFrame({
        "user": nodes, "p": p, "credited": succ_src,
        "exposures": N[:, 0], "effective_exposure": eff,
    }).to_parquet(out_dir / "source_influence.parquet", index=False)
    log.info("Fit complete: p_global %.3e | delay mean %.2f bins | same-bin mass %.2f | orphan share %.1f%%",
             p_bar, mean_bins, q[0], 100 * orphan_share)
    return params


def load_fit(base_path: str | Path = ".", dataset_name: str = "flickr",
             output_dir: str = OUTPUT_DIR) -> tuple[dict, pd.DataFrame]:
    """Return (params dict, per-source influence table). Users not in the table use params['p_global']."""
    d = Path(base_path) / output_dir / dataset_name / "fit"
    if not (d / "fitted_params.json").exists():
        raise FileNotFoundError(f"Missing {d}/fitted_params.json. Run `python -m src.diffusion.fit` first.")
    return json.loads((d / "fitted_params.json").read_text()), pd.read_parquet(d / "source_influence.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the discrete-time cascade model on the train split.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--bin-seconds", type=int, default=BIN_SECONDS, help="default: auto-detect")
    p.add_argument("--max-delay-bins", type=int, default=MAX_DELAY_BINS)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--tol", type=float, default=TOL)
    p.add_argument("--prior-exposures", type=float, default=PRIOR_EXPOSURES)
    p.add_argument("--ignore-edge-time", action="store_true", default=IGNORE_EDGE_TIME,
                   help="treat the graph as static (an edge counts even if created after the adoption)")
    p.add_argument("--reverse-edges", action="store_true", default=REVERSE_EDGES)
    p.add_argument("--allow-static-graph", action="store_true")
    p.add_argument("--max-train-posts", type=int, default=None, help="random subset of train posts (tests)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    fit(base_path=a.base_path, dataset_name=a.dataset, output_dir=a.output_dir,
        bin_seconds=a.bin_seconds, max_delay_bins=a.max_delay_bins, n_iter=a.n_iter, tol=a.tol,
        prior_exposures=a.prior_exposures, ignore_edge_time=a.ignore_edge_time,
        reverse_edges=a.reverse_edges, max_train_posts=a.max_train_posts, seed=a.seed,
        allow_static_graph=a.allow_static_graph)