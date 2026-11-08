"""Fit the delay rate of the diffusion model on the TRAIN split only.

The edge probabilities are NOT learned any more: simulator.py draws a fixed random p for every edge
(0..P_MAX). The only thing learned here is how long a follower waits after a source adopts:

    delay_rate   1 / mean delay (or ln2 / median delay)  ->  delays are Exponential(delay_rate)

Usage
-----
    python -m src.diffusion.fit --dataset flickr --base-path .
    python -m src.diffusion.fit --dataset flickr --max-train-posts 2000     # quick test
    python -m src.diffusion.fit --help

Writes  <base-path>/<output-dir>/<dataset>/fit/fitted_params.json
Reads   data/processed/<dataset>/{events,edges}.parquet   and   data/splits/<dataset>/split_meta.parquet
        split_meta needs: post_id, split, t_obs, t_end.  Labels are never read.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ============================================================================ #
# SETTINGS
# ============================================================================ #
OUTPUT_DIR = "outputs"
ESTIMATOR = "mean"          # "mean": rate = 1/mean delay | "median": rate = ln2/median delay
MAX_DELAY = None            # None = each post's own window (t_end - t_obs)
SEED = 0
FALLBACK_DELAY_RATE = 1.0 / 3600.0   # used if no delay could be measured


def _edge_time_available(canon_dir: Path) -> bool:
    manifest = canon_dir / "manifest.json"
    if manifest.exists():
        return bool(json.loads(manifest.read_text()).get("edge_time_available", True))
    return True


def fit(
    base_path: str | Path = ".",
    dataset_name: str = "flickr",
    *,
    output_dir: str = OUTPUT_DIR,
    estimator: str = ESTIMATOR,
    max_delay: float | None = MAX_DELAY,
    max_train_posts: int | None = None,
    seed: int = SEED,
    allow_static_graph: bool = False,
) -> dict:
    if estimator not in ("mean", "median"):
        raise ValueError("estimator must be 'mean' or 'median'")

    base = Path(base_path)
    canon_dir = base / "data" / "processed" / dataset_name
    splits_dir = base / "data" / "splits" / dataset_name
    out_dir = base / output_dir / dataset_name / "fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    timed = _edge_time_available(canon_dir)
    if not timed and not allow_static_graph:
        raise ValueError(
            f"{dataset_name} has a static graph (no edge times): an edge created after an adoption "
            f"would count as exposure (leak). Pass allow_static_graph=True to accept this explicitly."
        )

    # ---- 1. Train posts ----
    meta = pd.read_parquet(splits_dir / "split_meta.parquet")
    missing = {"post_id", "split", "t_obs", "t_end"} - set(meta.columns)
    if missing:
        raise ValueError(f"split_meta.parquet is missing columns {sorted(missing)}")
    train = meta[meta["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise ValueError("Train split is empty! Re-run the splitter with different windows.")
    
    if max_train_posts is not None and max_train_posts < len(train):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(train), size=max_train_posts, replace=False))
        train = train.iloc[keep].reset_index(drop=True)
        log.warning("Fitting on a random subset of %d train posts (test mode).", len(train))
    n_posts = len(train)

    # ---- 2. Events: train posts only, cut at each post's own t_end ----
    events = pd.read_parquet(canon_dir / "events.parquet", columns=["post_id", "user", "time"])
    events = events.merge(train[["post_id", "t_obs", "t_end"]], on="post_id", how="inner")
    events = events[events["time"] <= events["t_end"]]

    if events.empty:
        raise ValueError("No train events found (check split_meta / events).")

    # First adoption of each user per post
    events = events.sort_values(["post_id", "time"]).drop_duplicates(subset=["post_id", "user"])
    train_users = events["user"].unique()
    log.info("Train posts: %d | activations: %d | users: %d", n_posts, len(events), len(train_users))

    # ---- 3. Edges between train adopters ----
    ecols = ["source", "target"] + (["created_at"] if timed else [])
    edges = pd.read_parquet(canon_dir / "edges.parquet", columns=ecols)
    if timed:
        edges = edges[edges["created_at"] <= train["t_end"].max()]

    # Filter edges to those strictly between train_users (massive speedup)
    edges = edges[edges["source"].isin(train_users) & edges["target"].isin(train_users)]
    if timed:
        edges = edges.sort_values("created_at").drop_duplicates(subset=["source", "target"])
    else:
        edges = edges.drop_duplicates(subset=["source", "target"])
    log.info("Edges between train adopters: %d", len(edges))

    # ---- 4. Relational Join for Delays ----
    # Join target events (B) with in-edges (Memory safe: limits out-degree explosion)
    ev_tgt = events[["post_id", "user", "time"]].rename(columns={"user": "target", "time": "t_v"})
    tgt_joined = ev_tgt.merge(edges, on="target", how="inner")

    # Join with source events (A) on post_id & source to ensure both adopted in the SAME post
    ev_src = events[["post_id", "user", "time", "t_obs", "t_end"]].rename(
        columns={"user": "source", "time": "t_u"}
    )
    pairs = tgt_joined.merge(ev_src, on=["post_id", "source"], how="inner")

    # ---- 5. Valid Exposure & Delay Calculation ----
    if timed:
        # Edge must be created before the target adopts
        pairs = pairs[pairs["created_at"] < pairs["t_v"]]
        # Exposure begins at adoption OR when the edge is created, whichever is later
        pairs["exposure_start"] = np.maximum(pairs["t_u"], pairs["created_at"])
    else:
        pairs["exposure_start"] = pairs["t_u"]

    # Target must adopt AFTER exposure starts
    pairs = pairs[pairs["exposure_start"] < pairs["t_v"]]
    pairs["delay"] = pairs["t_v"] - pairs["exposure_start"]

    # Cap delay
    pairs["cap"] = pairs["t_end"] - pairs["t_obs"] if max_delay is None else float(max_delay)
    pairs = pairs[pairs["delay"] <= pairs["cap"]]

    # ---- 6. Credit only the most recent earlier in-neighbour ----
    # This prevents inflated means by selecting the maximum exposure_start (minimum delay)
    if not pairs.empty:
        idx = pairs.groupby(["post_id", "target"])["exposure_start"].idxmax()
        best_pairs = pairs.loc[idx]
        delays = best_pairs["delay"].to_numpy()
    else:
        delays = np.empty(0)
        
    pairs_checked = len(pairs)

    # ---- 7. Delay Rate ----
    delay_rate, delay_mean, delay_median = FALLBACK_DELAY_RATE, None, None
    if len(delays):
        delay_mean, delay_median = float(delays.mean()), float(np.median(delays))
        rate = 1.0 / delay_mean if estimator == "mean" else math.log(2.0) / delay_median
        if np.isfinite(rate) and rate > 0:
            delay_rate = float(rate)
    else:
        log.warning("No delay could be measured; using fallback delay_rate %.3e", FALLBACK_DELAY_RATE)

    params = {
        "delay_rate": delay_rate,
        "estimator": estimator,
        "delay_mean": delay_mean,
        "delay_median": delay_median,
        "max_delay": None if max_delay is None else float(max_delay),
        "n_delays": int(len(delays)),
        "n_pairs_checked": int(pairs_checked),
        "n_train_posts": int(n_posts),
        "edge_time_available": timed,
    }
    (out_dir / "fitted_params.json").write_text(json.dumps(params, indent=2))
    log.info("Fit complete -> delay_rate %.3e (mean delay %s, median %s, %d delays from %d pairs)",
             delay_rate, delay_mean, delay_median, len(delays), pairs_checked)
    return params


def load_fit(base_path: str | Path = ".", dataset_name: str = "flickr", output_dir: str = OUTPUT_DIR) -> dict:
    """Read the saved fit (a dict; simulator.py uses params["delay_rate"])."""
    path = Path(base_path) / output_dir / dataset_name / "fit" / "fitted_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python -m src.diffusion.fit` first.")
    return json.loads(path.read_text())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the delay rate on the train split.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--estimator", choices=["mean", "median"], default=ESTIMATOR)
    p.add_argument("--max-delay", type=float, default=MAX_DELAY,
                   help="ignore delays longer than this (time units); default: each post's t_end - t_obs")
    p.add_argument("--max-train-posts", type=int, default=None, help="random subset of train posts (tests)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--allow-static-graph", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    fit(base_path=a.base_path, dataset_name=a.dataset, output_dir=a.output_dir, estimator=a.estimator,
        max_delay=a.max_delay, max_train_posts=a.max_train_posts, seed=a.seed,
        allow_static_graph=a.allow_static_graph)