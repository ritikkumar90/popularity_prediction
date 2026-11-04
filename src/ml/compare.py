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

def compare_models():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = cfg.run.base_path
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    splits_dir = base_dir / "data" / "splits" / cfg.dataset_name
    ml_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "ml"
    
    config_hash = find_latest_hash_dir(samples_dir)
    features_dir = samples_dir / config_hash
    
    # 1. Load Test Data
    log.info("Loading test datasets...")
    test_dataset = PostGraphDataset(cfg, split="test", config_hash=config_hash)
    
    features_df = pd.read_parquet(features_dir / "tabular_features.parquet")
    labels_df = pd.read_parquet(splits_dir / "labels.parquet")
    meta_df = pd.read_parquet(splits_dir / "split_meta.parquet")
    
    df = features_df.merge(labels_df, on="post_id").merge(meta_df[["post_id", "split"]], on="post_id")
    test_df = df[df["split"] == "test"].copy()
    
    # 2. Select 5 Posts from the Test Set
    num_samples = min(5, len(test_dataset))
    selected_indices = list(range(num_samples))
    post_ids = [test_dataset.index.iloc[i]["post_id"] for i in selected_indices]
    
    # Create a unified PyG Batch for the PyTorch models
    graph_data_list = [test_dataset[i] for i in selected_indices]
    batch = Batch.from_data_list(graph_data_list).to(device)
    
    # Revert target from log1p back to absolute values
    ground_truth_log = batch.y.squeeze().cpu().numpy()
    ground_truth_raw = np.expm1(ground_truth_log)
    
    # 3. HGT Inference
    log.info("Running HGT...")
    hgt_model = PopularityHGT(
        metadata=batch.metadata(),
        hidden_channels=cfg.ml.hgt.hidden_channels,
        num_heads=cfg.ml.hgt.num_heads,
        num_layers=cfg.ml.hgt.num_layers
    ).to(device)
    hgt_model.load_state_dict(torch.load(ml_dir / "hgt_best.pt", weights_only=True, map_location=device))
    hgt_model.eval()
    
    with torch.no_grad():
        hgt_preds_raw = np.expm1(hgt_model(batch.x_dict, batch.edge_index_dict).cpu().numpy())
        
    # 4. R-GCN Inference
    log.info("Running R-GCN...")
    rgcn_model = PopularityRGCN(
        metadata=batch.metadata(),
        hidden_channels=cfg.ml.rgcn.hidden_channels,
        num_layers=cfg.ml.rgcn.num_layers
    ).to(device)
    rgcn_model.load_state_dict(torch.load(ml_dir / "rgcn_best.pt", weights_only=True, map_location=device))
    rgcn_model.eval()
    
    with torch.no_grad():
        rgcn_preds_raw = np.expm1(rgcn_model(batch.x_dict, batch.edge_index_dict).cpu().numpy())
        
    # 5. XGBoost Inference
    log.info("Running XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(ml_dir / "xgboost_best.json")
    
    # Filter tabular data strictly for the selected post_ids to maintain alignment
    feature_cols = [c for c in features_df.columns if c != "post_id"]
    selected_test_df = test_df.set_index("post_id").loc[post_ids].reset_index()
    X_test_selected = selected_test_df[feature_cols]
    
    xgb_preds_raw = np.expm1(xgb_model.predict(X_test_selected))
    
    # 6. Display Comparison Table
    print("\n" + "="*85)
    print(f"{'Post ID':<15} | {'Ground Truth':<12} | {'HGT Pred':<12} | {'R-GCN Pred':<12} | {'XGBoost Pred':<12}")
    print("-" * 85)
    
    for i in range(num_samples):
        pid = post_ids[i]
        gt = ground_truth_raw[i] if ground_truth_raw.ndim > 0 else ground_truth_raw
        h_p = hgt_preds_raw[i] if hgt_preds_raw.ndim > 0 else hgt_preds_raw
        r_p = rgcn_preds_raw[i] if rgcn_preds_raw.ndim > 0 else rgcn_preds_raw
        x_p = xgb_preds_raw[i] if xgb_preds_raw.ndim > 0 else xgb_preds_raw
        
        print(f"{pid:<15} | {float(gt):<12.1f} | {float(h_p):<12.1f} | {float(r_p):<12.1f} | {float(x_p):<12.1f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    compare_models()