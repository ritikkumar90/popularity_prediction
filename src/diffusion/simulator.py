"""Diffusion simulator: CLI Orchestrator for the discrete-time cascade engine."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.diffusion.variants import (
    CascadeEngine,
    PostSet,
    predict_popularity,
    choose_seeds,
    OUTPUT_DIR,
    SPLIT,
    SEED,
    SEED_MODE,
    K_VALUES,
    SELECT_METHODS,
    N_CANDIDATES,
    N_WORLDS,
    EXTERNAL_PER_BIN,
    N_SELECT_WORLDS,
    N_EVAL_WORLDS
)

log = logging.getLogger(__name__)

def run_simulation(
    dataset: str,
    base_path: str,
    output_dir: str,
    split: str,
    max_posts: int | None,
    seed: int,
    tasks: list[str],
    n_chunks: int | None,
    n_worlds: int,
    external_per_bin: float,
    mode: str,
    k_values: list[int],
    methods: list[str],
    n_candidates: int,
    n_select_worlds: int,
    n_eval_worlds: int
):
    base = Path(base_path)
    res_dir = base / output_dir / dataset / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading graph and fitted parameters...")
    engine = CascadeEngine.from_files(base_path, dataset, output_dir, n_chunks=n_chunks)
    
    log.info(f"Loading posts for split '{split}'...")
    posts = PostSet.from_files(engine, base_path, dataset, split=split, max_posts=max_posts, seed=seed)

    # 1. Run Popularity Prediction
    log.info("--- Running Case 1: Popularity Prediction ---")
    pred_df = predict_popularity(
        engine, posts, n_worlds=n_worlds, external_per_bin=external_per_bin, seed=seed
    )

    # 2. Run Seed Selection (baselines disabled for simplicity)
    log.info(f"--- Running Case 2: Seed Selection (mode: {mode}) ---")
    seeds_df = choose_seeds(
        engine, posts, mode=mode, k_values=k_values, methods=methods, 
        baselines=False,  # Turned off random, degree, score, etc.
        n_candidates=n_candidates, n_select_worlds=n_select_worlds, 
        n_eval_worlds=n_eval_worlds, seed=seed
    )

    # 3. Reshape Seed Selection to Wide Format (Columns: greedy_k1, celf_k2, etc.)
    seeds_df['method_k'] = seeds_df['method'] + '_k' + seeds_df['k'].astype(str)
    seeds_wide = seeds_df.pivot(index='post_id', columns='method_k', values='spread').reset_index()

    # 4. Load Ground Truth Labels
    labels_path = base / "data" / "splits" / dataset / "labels.parquet"
    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)[['post_id', 'label']]
        labels_df.rename(columns={'label': 'ground_truth'}, inplace=True)
    else:
        log.warning(f"Labels file not found at {labels_path}. Ground truth will be empty.")
        labels_df = pd.DataFrame({'post_id': posts.meta['post_id'], 'ground_truth': pd.NA})

    # 5. Merge Everything Together
    log.info("--- Consolidating Final Table ---")
    final_df = pred_df[['post_id', 'pred_label']].rename(columns={'pred_label': 'predicted_popularity'})
    
    final_df = final_df.merge(labels_df, on='post_id', how='left')
    final_df = final_df.merge(seeds_wide, on='post_id', how='left')

    # Reorder columns to match your exact request
    cols = ['post_id', 'ground_truth', 'predicted_popularity']
    for m in methods:
        for k in k_values:
            col_name = f"{m}_k{k}"
            if col_name in final_df.columns:
                cols.append(col_name)
                
    final_df = final_df[cols]

    # Save the consolidated table
    out_file = res_dir / f"consolidated_results_{split}.csv"
    final_df.to_csv(out_file, index=False)
    
    # Print a preview to the terminal
    print(f"\n=== Consolidated Results Preview ({split} split) ===")
    print(final_df.head(10).to_string(index=False))
    print(f"\nSaved full table to: {out_file}")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the discrete-time cascade simulator.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--split", choices=["train", "val", "test"], default=SPLIT)
    p.add_argument("--max-posts", type=int, default=None, help="Limit number of posts")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tasks", default="predict,seeds", help="Comma-separated list of tasks: predict,seeds")
    p.add_argument("--n-chunks", type=int, default=None, help="Numba parallel chunks")
    
    # Case 1
    p.add_argument("--n-worlds", type=int, default=N_WORLDS, help="Monte Carlo worlds for Case 1")
    p.add_argument("--external-per-bin", type=float, default=EXTERNAL_PER_BIN)
    
    # Case 2
    p.add_argument("--mode", choices=["cold", "addon"], default=SEED_MODE, help="Seed start mode")
    p.add_argument("--k-values", default=",".join(map(str, K_VALUES)))
    p.add_argument("--methods", default=",".join(SELECT_METHODS), help="greedy,celf")
    p.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    p.add_argument("--n-select-worlds", type=int, default=N_SELECT_WORLDS)
    p.add_argument("--n-eval-worlds", type=int, default=N_EVAL_WORLDS)
    
    return p.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    
    tasks = [t.strip().lower() for t in a.tasks.split(",")]
    k_values = [int(k.strip()) for k in a.k_values.split(",")]
    methods = [m.strip().lower() for m in a.methods.split(",")]
    
    run_simulation(
        dataset=a.dataset,
        base_path=a.base_path,
        output_dir=a.output_dir,
        split=a.split,
        max_posts=a.max_posts,
        seed=a.seed,
        tasks=tasks,
        n_chunks=a.n_chunks,
        n_worlds=a.n_worlds,
        external_per_bin=a.external_per_bin,
        mode=a.mode,
        k_values=k_values,
        methods=methods,
        n_candidates=a.n_candidates,
        n_select_worlds=a.n_select_worlds,
        n_eval_worlds=a.n_eval_worlds
    )