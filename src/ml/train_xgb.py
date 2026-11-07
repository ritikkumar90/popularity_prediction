import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

log = logging.getLogger(__name__)

# --- Default Configurations ---
DEFAULT_BASE_PATH = "."
DEFAULT_DATASET_NAME = "flickr"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.1


def find_latest_hash_dir(samples_dir: Path) -> str:
    """Find the most recently modified feature hash directory."""
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory {samples_dir}. Run features.py first.")
    
    subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hashed feature directories found in {samples_dir}")
    
    # Check for index.parquet to ensure it's a finished run if possible
    finished = [d for d in subdirs if (d / "index.parquet").exists()]
    if finished:
        latest = max(finished, key=lambda d: (d / "index.parquet").stat().st_mtime)
    else:
        latest = max(subdirs, key=lambda d: d.stat().st_mtime)
        
    return latest.name


def train_xgb():
    parser = argparse.ArgumentParser(description="Train XGBoost Baseline for Popularity Prediction")
    
    # Environment Arguments
    parser.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH, help="Base path for the project")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, help="Target dataset name (e.g., flickr, digg)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory inside base_path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    
    # XGBoost Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="Number of boosting rounds")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    
    args = parser.parse_args()

    # Setup directories
    base_dir = Path(args.base_path)
    samples_dir = base_dir / args.output_dir / args.dataset_name / "samples"
    splits_dir = base_dir / "data" / "splits" / args.dataset_name
    ml_dir = base_dir / args.output_dir / args.dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = find_latest_hash_dir(samples_dir)
    features_dir = samples_dir / config_hash
    
    log.info(f"Loading tabular data from {features_dir}")
    
    # 1. Load tabular features, labels, and splits
    features_df = pd.read_parquet(features_dir / "tabular_features.parquet")
    labels_df = pd.read_parquet(splits_dir / "labels.parquet")
    meta_df = pd.read_parquet(splits_dir / "split_meta.parquet")
    
    # Merge everything on post_id
    df = features_df.merge(labels_df, on="post_id").merge(meta_df[["post_id", "split"]], on="post_id")
    
    # 2. Target transformation (log1p to optimize MSLE natively via MSE)
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
    log.info(f"Initializing XGBoost with hyperparams: n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate}")
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="reg:squarederror",
        random_state=args.seed,
        early_stopping_rounds=10
    )
    
    log.info("Starting XGBoost training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Print early stopping and training summary
    best_iteration = model.best_iteration
    log.info(f"Training completed. Best iteration: {best_iteration}")
    
    # 5. Save Model Weights
    model_save_path = ml_dir / "xgboost_best.json"
    model.save_model(model_save_path)
    log.info(f" -> Saved best XGBoost model to {model_save_path}")
    
    # 6. Evaluate validation
    val_preds_log = model.predict(X_val)
    val_loss = mean_squared_error(y_val, val_preds_log)
    
    # Convert back from log scale for MAE
    val_preds_raw = np.expm1(val_preds_log)
    val_true_raw = np.expm1(y_val)
    val_mae = mean_absolute_error(val_true_raw, val_preds_raw)
    
    log.info(f"Validation | MSLE: {val_loss:.4f} | MAE: {val_mae:.2f}")
    
    # 7. Evaluate test
    test_preds_log = model.predict(X_test)
    test_loss = mean_squared_error(y_test, test_preds_log)
    
    test_preds_raw = np.expm1(test_preds_log)
    test_true_raw = np.expm1(y_test)
    test_mae = mean_absolute_error(test_true_raw, test_preds_raw)
    
    print(f"\n=== FINAL LOSS STATS ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: XGBoost (n_est={args.n_estimators}, depth={args.max_depth}, lr={args.learning_rate})")
    print(f"Validation Loss (MSLE): {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Test Loss (MSLE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"========================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_xgb()