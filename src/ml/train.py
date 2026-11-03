import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.config import load_config
from src.ml.dataset import get_dataloaders
from src.ml.models.hgt import PopularityHGT

log = logging.getLogger(__name__)

def find_latest_hash_dir(samples_dir: Path) -> str:
    """Finds the generated hash directory from features.py automatically."""
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory {samples_dir}. Run features.py first.")
    
    subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hashed feature directories found in {samples_dir}")
    
    # Sort by modified time to get the most recent extraction
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest.name

def train_hgt():
    parser = argparse.ArgumentParser(description="Train HGT for Popularity Prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels size")
    parser.add_argument("--heads", type=int, default=4, help="Number of HGT attention heads")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    base_dir = cfg.run.base_path
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    ml_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = find_latest_hash_dir(samples_dir)
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, config_hash=config_hash, batch_size=args.batch_size
    )

    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check your splits and extraction.")

    # 2. Initialize Model using metadata from the first batch
    sample_batch = next(iter(train_loader))
    model = PopularityHGT(
        metadata=sample_batch.metadata(),
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=2
    ).to(device)

    # 3. Setup Optimizer and Loss
    # Since targets are log1p(y), MSE effectively optimizes MSLE on original scale
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    model_save_path = ml_dir / "hgt_best.pt"

    # 4. Training Loop
    log.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            preds = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(preds, batch.y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(train_loader.dataset)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x_dict, batch.edge_index_dict)
                loss = criterion(preds, batch.y)
                val_loss += loss.item() * batch.num_graphs
                
                # Compute MAE on original scale (expm1 reverses log1p)
                true_vals = torch.expm1(batch.y)
                pred_vals = torch.expm1(preds)
                val_mae += torch.abs(true_vals - pred_vals).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        log.info(f"Epoch {epoch:03d} | Train Loss (MSLE): {train_loss:.4f} | Val Loss (MSLE): {val_loss:.4f} | Val MAE: {val_mae:.2f}")

        # 6. Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            log.info(f" -> Saved new best model to {model_save_path}")

    # 7. Test Evaluation
    log.info("--- Training Complete. Evaluating on Test Set ---")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict)
            test_loss += criterion(preds, batch.y).item() * batch.num_graphs
            
            true_vals = torch.expm1(batch.y)
            pred_vals = torch.expm1(preds)
            test_mae += torch.abs(true_vals - pred_vals).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    
    log.info(f"Final Test Loss (MSLE): {test_loss:.4f}")
    log.info(f"Final Test MAE (Absolute Count): {test_mae:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_hgt()