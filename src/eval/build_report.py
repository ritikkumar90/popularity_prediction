import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

log = logging.getLogger(__name__)

def generate_report():
    cfg = load_config()
    base_dir = cfg.run.base_path
    results_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "results"
    
    ml_path = results_dir / "ml_predictions.csv"
    im_path = results_dir / "im_predictions.csv"
    
    if not ml_path.exists() or not im_path.exists():
        raise FileNotFoundError("Missing prediction CSVs. Run ML and IM inference first.")
        
    ml_df = pd.read_csv(ml_path)
    im_df = pd.read_csv(im_path)
    
    # Merge ML and IM predictions
    preds_df = ml_df.merge(im_df, on="post_id")
    
    cols = [
        "Metric_Level", "Ground truth", "HGT", "RGCN", "XGB", 
        "IM (k=1) seed by greedy", "IM (k=1) seed by celf", 
        "IM avg(k=1, 2, 3) seed by greedy", "IM avg(k=1, 2, 3) seed by celf"
    ]
    rows = []
    
    # --- Table 1: Photo id (First 5 individual posts) ---
    rows.append(["Photo id"] + [np.nan] * 8)
    for i in range(min(5, len(preds_df))):
        row = preds_df.iloc[i]
        rows.append([
            str(i + 1), row["Ground truth"], row["HGT"], row["RGCN"], row["XGB"],
            row["IM (k=1) seed by greedy"], row["IM (k=1) seed by celf"],
            row["IM avg(k=1, 2, 3) seed by greedy"], row["IM avg(k=1, 2, 3) seed by celf"]
        ])
        
    rows.append([np.nan] * 9)
    
    # --- Table 2: Average num of post ---
    rows.append(["Average num of post"] + [np.nan] * 8)
    for size in [1, 5, 10, 15, 20]:
        if size <= len(preds_df):
            subset = preds_df.head(size)
            rows.append([
                str(size),
                subset["Ground truth"].mean(), subset["HGT"].mean(),
                subset["RGCN"].mean(), subset["XGB"].mean(),
                subset["IM (k=1) seed by greedy"].mean(), subset["IM (k=1) seed by celf"].mean(),
                subset["IM avg(k=1, 2, 3) seed by greedy"].mean(), subset["IM avg(k=1, 2, 3) seed by celf"].mean()
            ])
        else:
            rows.append([str(size)] + [np.nan] * 8)
            
    rows.append([np.nan] * 9)
    
    # --- Table 3: Time (Given post p_i) ---
    rows.append(["Time (Given post p_i)"] + [np.nan] * 8)
    preds_df = preds_df.sort_values("t_obs")
    for t in [20, 40, 60, 80, 100]:
        cutoff_idx = int(len(preds_df) * (t / 100.0)) if len(preds_df) > 0 else 0
        subset = preds_df.iloc[:cutoff_idx] if cutoff_idx > 0 else preds_df.head(1)
        
        rows.append([
            f"{t} timestamp",
            subset["Ground truth"].mean(), subset["HGT"].mean(),
            subset["RGCN"].mean(), subset["XGB"].mean(),
            subset["IM (k=1) seed by greedy"].mean(), subset["IM (k=1) seed by celf"].mean(),
            subset["IM avg(k=1, 2, 3) seed by greedy"].mean(), subset["IM avg(k=1, 2, 3) seed by celf"].mean()
        ])
        
    final_df = pd.DataFrame(rows, columns=cols)
    out_path = results_dir / "final_evaluation_matrix.csv"
    final_df.to_csv(out_path, index=False)
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print("\n--- Popularity Prediction Evaluation ---")
    print(final_df.to_string(index=False, na_rep=''))
    log.info(f"\nSaved final matrix to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_report()