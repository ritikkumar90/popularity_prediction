import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.config import load_config
from src.ml.dataset import get_dataloaders
from src.ml.models.hgt import PopularityHGT
from src.ml.train import find_latest_hash_dir

log = logging.getLogger(__name__)

def test_hgt():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = cfg.run.base_path
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    ml_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "ml"
    model_save_path = ml_dir / "hgt_best.pt"
    
    if not model_save_path.exists():
        raise FileNotFoundError(f"Saved model weights not found at {model_save_path}")

    config_hash = find_latest_hash_dir(samples_dir)
    
    # 1. Load Dataloaders (We only need the test_loader, but get_dataloaders returns all three)
    _, _, test_loader = get_dataloaders(
        cfg, config_hash=config_hash, batch_size=cfg.ml.batch_size
    )

    if len(test_loader) == 0:
        raise ValueError("Test loader is empty!")

    # 2. Initialize Model Architecture
    sample_batch = next(iter(test_loader))
    model = PopularityHGT(
        metadata=sample_batch.metadata(),
        hidden_channels=cfg.ml.hgt.hidden_channels,
        num_heads=cfg.ml.hgt.num_heads,
        num_layers=cfg.ml.hgt.num_layers
    ).to(device)

    # 3. Load Saved Weights
    log.info(f"Loading saved weights from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))
    model.eval()

    # 4. Evaluation Loop
    criterion = nn.MSELoss()
    test_loss = 0.0
    test_mae = 0.0
    
    log.info("Starting test evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            preds = model(batch.x_dict, batch.edge_index_dict)
            
            # Squeeze batch.y to prevent broadcasting mismatches
            loss = criterion(preds, batch.y.squeeze())
            test_loss += loss.item() * batch.num_graphs
            
            # Revert log1p transformation for absolute error
            true_vals = torch.expm1(batch.y.squeeze())
            pred_vals = torch.expm1(preds)
            test_mae += torch.abs(true_vals - pred_vals).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    
    log.info(f"Final Test Loss (MSLE): {test_loss:.4f}")
    log.info(f"Final Test MAE (Absolute Count): {test_mae:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_hgt()