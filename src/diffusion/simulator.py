"""Diffusion simulator: Orchestrator, CLI, and Output generation."""
from __future__ import annotations

import argparse
import importlib
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump

from src.diffusion.fit import load_fit
from src.diffusion.variants import VARIANTS, Opts
from src.diffusion.simulation_graph import load_posts, _get_graph, PostSim

log = logging.getLogger(__name__)

# ============================================================================ #
# SETTINGS
# ============================================================================ #
DATASET = "flickr"
BASE_PATH = "."
OUTPUT_DIR = "outputs"
SPLIT = "test"
N_POSTS = 5
SEED = 0
P_MAX = "0.05"
VARIANT_NAMES = "all"
IM_ALGOS = "greedy,celf"
K_LIST = "1,2,3"
SELECTION_RUNS = 30
EVAL_RUNS = 200
CANDIDATE_TOP_N = 50
N_JOBS = -1
CHUNK_POSTS = 8
SHOW_POSTS = 5
TAG = ""

def _run_chunk(graph_path: str, tasks: list, cfg: dict) -> list:
    G = _get_graph(graph_path)
    opts = Opts(
        k_list=tuple(cfg["k_list"]), 
        selection_runs=cfg["selection_runs"], 
        eval_runs=cfg["eval_runs"],
        candidate_top_n=cfg["candidate_top_n"],
        im_algos=tuple(cfg["im_algos"])
    )
    rows = []
    for post_id, t_obs, t_end, act, act_dt in tasks:
        post = PostSim(G, cfg, t_obs, t_end, act, act_dt)
        row = {"post_id": post_id, "observed": int(len(act))}
        for name in cfg["variants"]:
            t0 = time.perf_counter()
            row.update(VARIANTS[name](post, opts))
            row[f"sec_{name}"] = time.perf_counter() - t0
        rows.append(row)
    return rows

_MODULE = __spec__.name if (__name__ == "__main__" and __spec__ is not None) else __name__

def _task(module: str, graph_path: str, tasks: list, cfg: dict) -> list:
    return importlib.import_module(module)._run_chunk(graph_path, tasks, cfg)

def simulate(graph_path: str, tasks: list, cfg: dict, n_jobs: int, chunk_posts: int) -> pd.DataFrame:
    if n_jobs != 1 and _MODULE == "__main__":
        log.warning("Run with `python -m src.diffusion.simulator` for parallel workers; using n_jobs=1.")
        n_jobs = 1

    tasks.sort(key=lambda t: len(t[3]), reverse=True)
    n_chunks = max(1, len(tasks) // chunk_posts)
    chunks = [[] for _ in range(n_chunks)]
    for i, t in enumerate(tasks):
        chunks[i % n_chunks].append(t)
    chunks = [c for c in chunks if c]

    log.info("Simulating %d posts in %d balanced chunks (n_jobs=%s, variants=%s, im_algos=%s, p_max=%g)...",
             len(tasks), len(chunks), n_jobs, cfg["variants"], cfg["im_algos"], cfg["p_max"])

    if n_jobs == 1:
        res = [_run_chunk(graph_path, c, cfg) for c in chunks]
    else:
        res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_task)(_MODULE, graph_path, c, cfg) for c in chunks)
    return pd.DataFrame([r for chunk in res for r in chunk])

def _msle(p, y) -> float:
    return float(np.mean((np.log1p(p) - np.log1p(y)) ** 2))

def score(df: pd.DataFrame) -> pd.DataFrame:
    y = df["label"].to_numpy(float)
    rows = [dict(method="observed_only", n=len(df), msle=_msle(df["observed"], y),
                 mae=float(np.mean(np.abs(df["observed"] - y))), mean_pred=float(df["observed"].mean()),
                 mean_label=float(y.mean()), mean_gain=np.nan, scored=True)]
    if "baseline_pred" in df:
        p = df["baseline_pred"].to_numpy(float)
        rows.append(dict(method="baseline", n=len(df), msle=_msle(p, y), mae=float(np.mean(np.abs(p - y))),
                         mean_pred=float(p.mean()), mean_label=float(y.mean()), mean_gain=np.nan, scored=True))
    for col in [c for c in df.columns if c.startswith(("greedy_k", "celf_k")) and not c.endswith("_gain")]:
        rows.append(dict(method=col, n=len(df), msle=np.nan, mae=np.nan, mean_pred=float(df[col].mean()),
                         mean_label=float(y.mean()), mean_gain=float(df[f"{col}_gain"].mean()), scored=False))
    return pd.DataFrame(rows)

def make_table(df: pd.DataFrame, k_list, im_algos) -> pd.DataFrame:
    t = pd.DataFrame({"Post ID": df["post_id"], "Ground Truth": df["label"], "Observed": df["observed"]})
    if "baseline_pred" in df: t["Baseline"] = df["baseline_pred"]
    for name in im_algos:
        title = name.capitalize()
        for k in k_list:
            if f"{name}_k{k}" in df: t[f"{title} k={k}"] = df[f"{name}_k{k}"]
    return t

def print_table(t: pd.DataFrame, n_show: int) -> None:
    top = t.sort_values("Ground Truth", ascending=False).head(n_show).astype(object)
    cols = list(top.columns)
    w = [max(len(c), 9) for c in cols]
    line = " | ".join(c.ljust(wi) for c, wi in zip(cols, w))
    print(f"\n=== Top {len(top)} posts by ground truth ===\n{'-' * len(line)}\n{line}\n{'-' * len(line)}")
    for _, r in top.iterrows():
        print(" | ".join((f"{r[c]:.1f}" if c != "Post ID" else str(r[c])).ljust(wi) for c, wi in zip(cols, w)))
    print("-" * len(line))

def _attach_labels(df: pd.DataFrame, splits_dir: Path) -> pd.DataFrame:
    labels = pd.read_parquet(splits_dir / "labels.parquet")[["post_id", "label"]]
    out = df.merge(labels, on="post_id", how="left")
    if out["label"].isna().any():
        out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out

def run(dataset=DATASET, base_path=BASE_PATH, output_dir=OUTPUT_DIR, split=SPLIT, n_posts=N_POSTS, seed=SEED,
        p_max=P_MAX, variants=VARIANT_NAMES, im_algos=IM_ALGOS, k_list=K_LIST, selection_runs=SELECTION_RUNS, eval_runs=EVAL_RUNS,
        candidate_top_n=CANDIDATE_TOP_N, n_jobs=N_JOBS, chunk_posts=CHUNK_POSTS, show_posts=SHOW_POSTS, tag=TAG,
        allow_static_graph=False):
    base = Path(base_path)
    res_dir = base / output_dir / dataset / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    p_values = [float(x) for x in str(p_max).split(",") if x.strip()]
    names = list(VARIANTS) if variants == "all" else [v.strip() for v in variants.split(",") if v.strip()]
    
    im_algos_list = [a.strip().lower() for a in str(im_algos).split(",") if a.strip()]
    ks = tuple(int(x) for x in str(k_list).split(",") if x.strip())

    params = load_fit(base_path, dataset, output_dir)
    G, tasks, meta = load_posts(base, dataset, split, n_posts, seed, allow_static_graph)
    cfg = dict(rate=float(params["delay_rate"]), seed=seed, seed_p=int(seed) * 7919 + 12345, k_list=list(ks),
               selection_runs=selection_runs, eval_runs=eval_runs, candidate_top_n=candidate_top_n, im_algos=im_algos_list, p_max=p_values[0])

    path = res_dir / f"_graph_tmp_{uuid.uuid4().hex[:10]}.joblib"
    dump(G, path)

    try:
        log.info("Warming up Numba JIT cache...")
        dummy_post = PostSim(_get_graph(str(path)), cfg, tasks[0][1], tasks[0][2], tasks[0][3][:1], tasks[0][4][:1])
        dummy_post.spread(seeds=[], runs=1, offset=0)
        cands = dummy_post.candidates(1)
        if cands: dummy_post.spread(seeds=cands, runs=1, offset=0)
    except Exception as e:
        log.warning("Warmup failed (ignoring): %s", e)

    try:
        if len(p_values) > 1:
            rows = []
            for pv in p_values:
                df = _attach_labels(simulate(str(path), tasks, {**cfg, "p_max": pv, "variants": ["baseline"]}, n_jobs, chunk_posts), base / "data" / "splits" / dataset)
                y, p = df["label"].to_numpy(float), df["baseline_pred"].to_numpy(float)
                rows.append(dict(p_max=pv, n=len(df), msle=_msle(p, y), mae=float(np.mean(np.abs(p - y)))))
            sweep = pd.DataFrame(rows)
            print(f"\n=== P_MAX sweep ===")
            print(sweep.to_string(index=False))
            sweep.to_csv(res_dir / f"diffusion_pmax_sweep{suffix}.csv", index=False)
            return sweep

        df = simulate(str(path), tasks, {**cfg, "p_max": p_values[0], "variants": names}, n_jobs, chunk_posts)
    finally:
        path.unlink(missing_ok=True)

    df = _attach_labels(df, base / "data" / "splits" / dataset).sort_values("post_id").reset_index(drop=True)
    table, metrics = make_table(df, ks, im_algos_list), score(df)
    
    df.to_csv(res_dir / f"diffusion_posts{suffix}.csv", index=False)
    table.to_csv(res_dir / f"diffusion_table{suffix}.csv", index=False)
    metrics.to_csv(res_dir / f"diffusion_metrics{suffix}.csv", index=False)

    print_table(table, show_posts)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    secs = [c for c in df.columns if c.startswith("sec_")]
    print("\n=== Mean seconds per post ===")
    print(df[secs].mean().round(3).to_string())
    return df, metrics

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--base-path", default=BASE_PATH)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--split", choices=["train", "val", "test"], default=SPLIT)
    p.add_argument("--n-posts", type=int, default=N_POSTS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--p-max", default=P_MAX)
    p.add_argument("--variants", default=VARIANT_NAMES, help="'all' or comma list (baseline,im_seeds)")
    p.add_argument("--im-algos", default=IM_ALGOS, help="comma separated list (greedy,celf)")
    p.add_argument("--k-list", default=K_LIST)
    p.add_argument("--selection-runs", type=int, default=SELECTION_RUNS)
    p.add_argument("--eval-runs", type=int, default=EVAL_RUNS)
    p.add_argument("--candidate-top-n", type=int, default=CANDIDATE_TOP_N)
    p.add_argument("--n-jobs", type=int, default=N_JOBS)
    p.add_argument("--chunk-posts", type=int, default=CHUNK_POSTS)
    p.add_argument("--show-posts", type=int, default=SHOW_POSTS)
    p.add_argument("--tag", default=TAG)
    p.add_argument("--allow-static-graph", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    run(**vars(a))