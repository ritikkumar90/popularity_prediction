"""Evaluate and compare XGBoost, R-GCN, and HGT models on the most popular test posts.

Usage:
    python -m src.ml.compare --dataset flickr
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch_geometric.data import Batch

# Import data loaders and architecture definitions
try:
    from src.ml.dataset import get_dataloaders, find_latest_hash_dir
except ImportError:
    from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders
    from src.ml.dataset_ram import find_latest_hash_dir
    
from src.ml.models.rgcn import PopularityRGCN
from src.ml.models.hgt import PopularityHGT

log = logging.getLogger(__name__)

def compare_models(
    base_path: str = ".",
    dataset_name: str = "flickr",
    num_samples: int = 5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(base_path)
    ml_dir = base_dir / "outputs" / dataset_name / "ml"
    samples_dir = base_dir / "outputs" / dataset_name / "samples"
    splits_dir = base_dir / "data" / "splits" / dataset_name
    config_hash = find_latest_hash_dir(samples_dir)
    
    # 1. Identify the top most popular posts in the test set
    log.info(f"Identifying the top {num_samples} most popular test posts...")
    meta = pd.read_parquet(splits_dir / "split_meta.parquet", columns=["post_id", "split"])
    labels = pd.read_parquet(splits_dir / "labels.parquet", columns=["post_id", "label"])
    
    # Merge and find the largest labels in the test split
    test_df = meta[meta["split"] == "test"].merge(labels, on="post_id")
    top_posts = test_df.nlargest(num_samples, "label")
    
    post_ids = top_posts["post_id"].to_numpy()
    y_true_raw = top_posts["label"].to_numpy()
    
    # 2. Get the PyG Dataset object
    _, _, test_loader = get_dataloaders(
        base_path=base_path,
        dataset_name=dataset_name,
        batch_size=32, # Batch size doesn't matter here since we bypass the loader
        num_workers=0
    )
    test_dataset = test_loader.dataset
    
    # Find the internal dataset indices for these specific post_ids
    idx_df = pd.read_parquet(samples_dir / config_hash / "index.parquet", columns=["post_id"])
    test_post_ids = idx_df[idx_df["post_id"].isin(set(test_df["post_id"]))]["post_id"].to_numpy()
    
    # Get the exact row index for each target post_id
    target_indices = [np.where(test_post_ids == pid)[0][0] for pid in post_ids]
    
    # 3. Manually construct a PyG batch containing ONLY these top posts
    log.info("Constructing graph batch for top posts...")
    data_list = [test_dataset.get(i) for i in target_indices]
    batch = Batch.from_data_list(data_list).to(device)

    # 4. XGBoost Inference
    log.info("Running XGBoost inference...")
    tabular_df = pd.read_parquet(samples_dir / config_hash / "tabular_features.parquet")
    xgb_features = tabular_df[tabular_df["post_id"].isin(post_ids)].set_index("post_id").loc[post_ids]
    
    xgb_model = xgb.XGBRegressor()
    xgb_model_path = ml_dir / "xgboost_best.json" 
    
    if xgb_model_path.exists():
        xgb_model.load_model(xgb_model_path)
        xgb_preds = xgb_model.predict(xgb_features)
        xgb_preds_raw = np.expm1(xgb_preds)
    else:
        log.warning(f"XGBoost model not found at {xgb_model_path}. Skipping.")
        xgb_preds_raw = np.full(num_samples, np.nan)

    # 5. R-GCN Inference
    log.info("Running R-GCN inference...")
    rgcn_model_path = ml_dir / "rgcn_best.pt"
    if rgcn_model_path.exists():
        # INITIALIZED WITH YOUR TRAINED HYPERPARAMS
        rgcn = PopularityRGCN(
            metadata=batch.metadata(),
            hidden_channels=128, 
            num_layers=3
        ).to(device)
        with torch.no_grad():
            rgcn(batch.x_dict, batch.edge_index_dict)
        rgcn.load_state_dict(torch.load(rgcn_model_path, map_location=device, weights_only=True))
        rgcn.eval()
        
        with torch.no_grad():
            rgcn_preds = rgcn(batch.x_dict, batch.edge_index_dict)
            rgcn_preds_raw = torch.expm1(rgcn_preds).cpu().numpy()
    else:
        log.warning(f"R-GCN model not found at {rgcn_model_path}. Skipping.")
        rgcn_preds_raw = np.full(num_samples, np.nan)

    # 6. HGT Inference
    log.info("Running HGT inference...")
    hgt_model_path = ml_dir / "hgt_best.pt"
    if hgt_model_path.exists():
        # NOTE: If you changed HGT hyperparams during training, update them here too!
        hgt = PopularityHGT(metadata=batch.metadata()).to(device)
        with torch.no_grad():
            hgt(batch.x_dict, batch.edge_index_dict)
        hgt.load_state_dict(torch.load(hgt_model_path, map_location=device, weights_only=True))
        hgt.eval()
        
        with torch.no_grad():
            hgt_preds = hgt(batch.x_dict, batch.edge_index_dict)
            hgt_preds_raw = torch.expm1(hgt_preds).cpu().numpy()
    else:
        log.warning(f"HGT model not found at {hgt_model_path}. Skipping.")
        hgt_preds_raw = np.full(num_samples, np.nan)

    # 7. Print Comparison Table
    print("\n" + "="*80)
    print(f"{'Post ID':<15} | {'Ground Truth':<12} | {'XGBoost':<12} | {'R-GCN':<12} | {'HGT':<12}")
    print("-" * 80)
    
    for i in range(num_samples):
        pid = str(post_ids[i])
        truth = f"{y_true_raw[i]:.1f}"
        xgb_val = f"{xgb_preds_raw[i]:.1f}" if not np.isnan(xgb_preds_raw[i]) else "N/A"
        rgcn_val = f"{rgcn_preds_raw[i]:.1f}" if not np.isnan(rgcn_preds_raw[i]) else "N/A"
        hgt_val = f"{hgt_preds_raw[i]:.1f}" if not np.isnan(hgt_preds_raw[i]) else "N/A"
        
        print(f"{pid:<15} | {truth:<12} | {xgb_val:<12} | {rgcn_val:<12} | {hgt_val:<12}")
    print("="*80 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compare Models on top test samples")
    parser.add_argument("--dataset", type=str, default="flickr", help="Target dataset name")
    parser.add_argument("--base-path", type=str, default=".", help="Base path for the project")
    args = parser.parse_args()
    
    compare_models(base_path=args.base_path, dataset_name=args.dataset)