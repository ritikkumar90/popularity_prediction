import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch_geometric.data import Batch

from src.config import load_config
from src.ml.dataset import PostGraphDataset
from src.ml.models.hgt import PopularityHGT
from src.ml.models.rgcn import PopularityRGCN
from src.ml.train import find_latest_hash_dir

log = logging.getLogger(__name__)

def run_ml_inference():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    base_dir = cfg.run.base_path
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    ml_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "ml"
    
    # Ensure results directory exists
    results_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = find_latest_hash_dir(samples_dir)
    log.info(f"Loading test data from hash {config_hash}")
    
    # 1. Load Data
    test_dataset = PostGraphDataset(cfg, split="test", config_hash=config_hash)
    
    features_df = pd.read_parquet(samples_dir / config_hash / "tabular_features.parquet")
    labels_df = pd.read_parquet(base_dir / "data" / "splits" / cfg.dataset_name / "labels.parquet")
    meta_df = pd.read_parquet(base_dir / "data" / "splits" / cfg.dataset_name / "split_meta.parquet")
    
    merged_features = features_df.merge(labels_df, on="post_id").merge(meta_df, on="post_id")
    
    post_ids = test_dataset.index["post_id"].tolist()
    t_obs_list = test_dataset.index["t_obs"].tolist()
    
    # Strictly select only numeric features for XGBoost to prevent dtype crashes
    feature_cols = [c for c in merged_features.columns if c.startswith("log_") or c.startswith("bin_frac_")]
    test_features = merged_features.set_index("post_id").loc[post_ids].reset_index()
    X_test = test_features[feature_cols]
    
    # Create unified batch for PyTorch
    log.info("Batching PyTorch graphs...")
    graph_data_list = [test_dataset[i] for i in range(len(test_dataset))]
    batch = Batch.from_data_list(graph_data_list).to(device)
    
    gt_raw = np.expm1(batch.y.squeeze().cpu().numpy())
    
    # 2. HGT Inference
    log.info("Running HGT inference...")
    hgt_model = PopularityHGT(
        metadata=batch.metadata(),
        hidden_channels=cfg.ml.hgt.hidden_channels,
        num_heads=cfg.ml.hgt.num_heads,
        num_layers=cfg.ml.hgt.num_layers
    ).to(device)
    hgt_model.load_state_dict(torch.load(ml_dir / "hgt_best.pt", weights_only=True, map_location=device))
    hgt_model.eval()
    with torch.no_grad():
        hgt_preds = np.expm1(hgt_model(batch.x_dict, batch.edge_index_dict).cpu().numpy())
        
    # 3. R-GCN Inference
    log.info("Running R-GCN inference...")
    rgcn_model = PopularityRGCN(
        metadata=batch.metadata(),
        hidden_channels=cfg.ml.rgcn.hidden_channels,
        num_layers=cfg.ml.rgcn.num_layers
    ).to(device)
    rgcn_model.load_state_dict(torch.load(ml_dir / "rgcn_best.pt", weights_only=True, map_location=device))
    rgcn_model.eval()
    with torch.no_grad():
        rgcn_preds = np.expm1(rgcn_model(batch.x_dict, batch.edge_index_dict).cpu().numpy())
        
    # 4. XGBoost Inference
    log.info("Running XGBoost inference...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(ml_dir / "xgboost_best.json")
    xgb_preds = np.expm1(xgb_model.predict(X_test))
    
    # 5. Save Results
    results_df = pd.DataFrame({
        "post_id": post_ids,
        "t_obs": t_obs_list,
        "Ground truth": gt_raw,
        "HGT": hgt_preds,
        "RGCN": rgcn_preds,
        "XGB": xgb_preds
    })
    
    out_path = results_dir / "ml_predictions.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"Saved ML predictions for {len(post_ids)} posts to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ml_inference()