import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import load_config

log = logging.getLogger(__name__)

def find_latest_hash_dir(samples_dir: Path) -> str:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory {samples_dir}. Run features.py first.")
    
    subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hashed feature directories found in {samples_dir}")
    
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest.name

def train_xgboost():
    parser = argparse.ArgumentParser(description="Train XGBoost Baseline for Popularity Prediction")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    cfg = load_config()
    base_dir = cfg.run.base_path
    
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    splits_dir = base_dir / "data" / "splits" / cfg.dataset_name
    
    config_hash = find_latest_hash_dir(samples_dir)
    ml_dir = samples_dir / config_hash
    
    log.info(f"Loading tabular data from {ml_dir}")
    
    # 1. Load tabular features, labels, and splits
    features_df = pd.read_parquet(ml_dir / "tabular_features.parquet")
    labels_df = pd.read_parquet(splits_dir / "labels.parquet")
    meta_df = pd.read_parquet(splits_dir / "split_meta.parquet")
    
    # Merge everything on post_id
    df = features_df.merge(labels_df, on="post_id").merge(meta_df[["post_id", "split"]], on="post_id")
    
    # 2. Target transformation (log1p to optimize MSLE)
    df["target"] = np.log1p(df["label"])
    
    feature_cols = [c for c in features_df.columns if c != "post_id"]
    
    # 3. Split data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]
    
    log.info(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    
    # 4. Initialize and Train Model
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="reg:squarederror",
        random_state=cfg.run.seed,
        early_stopping_rounds=10
    )
    
    log.info("Starting XGBoost training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 5. Evaluate validation
    val_preds_log = model.predict(X_val)
    val_loss = mean_squared_error(y_val, val_preds_log)
    
    # Convert back from log scale for MAE
    val_preds_raw = np.expm1(val_preds_log)
    val_true_raw = np.expm1(y_val)
    val_mae = mean_absolute_error(val_true_raw, val_preds_raw)
    
    log.info(f"Validation | MSLE: {val_loss:.4f} | MAE: {val_mae:.2f}")
    
    # 6. Evaluate test
    test_preds_log = model.predict(X_test)
    test_loss = mean_squared_error(y_test, test_preds_log)
    
    test_preds_raw = np.expm1(test_preds_log)
    test_true_raw = np.expm1(y_test)
    test_mae = mean_absolute_error(test_true_raw, test_preds_raw)
    
    log.info("--- Training Complete. Evaluating on Test Set ---")
    log.info(f"Final Test Loss (MSLE): {test_loss:.4f}")
    log.info(f"Final Test MAE (Absolute Count): {test_mae:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_xgboost()