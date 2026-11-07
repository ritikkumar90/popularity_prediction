"""Train the PopularityHGT model on the sharded subgraphs. No config files.

Hyperparameters live in the DEFAULTS block below; every one of them can also be set
from the command line (run with --help to see the flags).

Pipeline (run these first, in order):
    1. splitter.py     -> data/splits/<dataset>/{split_meta,labels}.parquet
    2. features.py     -> outputs/<dataset>/samples/<hash>/shard_*.pt
    3. this file       -> outputs/<dataset>/ml/hgt_best.pt

Usage
-----
    python -m src.ml.train --dataset flickr --base-path . --epochs 30
    python -m src.ml.train --help        # list all tunable flags

Loss is MSE on log1p(label) ("MSLE"); MAE is reported on the original counts.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.ml.dataset import get_dataloaders          # no config object needed
from src.ml.models.hgt import PopularityHGT, DROPOUT, HIDDEN_CHANNELS, NUM_HEADS, NUM_LAYERS

log = logging.getLogger(__name__)

# ============================================================================ #
# HYPERPARAMETERS / DEFAULTS  (override with CLI flags, e.g. --lr 1e-4)
# ============================================================================ #
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0          # dataloader workers (0 = load in the main process)


def train_hgt(
    base_path: str | Path = ".",
    dataset_name: str = "flickr",
    *,
    output_dir: str = "outputs",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    hidden_channels: int = HIDDEN_CHANNELS,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    num_workers: int = NUM_WORKERS,
    config_hash: str | None = None,   # None = latest folder in samples/
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s | Batch Size: %d", device, batch_size)

    base_path = Path(base_path)
    ml_dir = base_path / output_dir / dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data (config_hash=None -> the most recent features folder)
    train_loader, val_loader, test_loader = get_dataloaders(
        base_path,
        dataset_name,
        config_hash=config_hash,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
    )

    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check your splits and extraction.")

    # 2. Initialize model
    sample_batch = next(iter(train_loader))
    model = PopularityHGT(
        metadata=sample_batch.metadata(),
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Dummy forward pass to initialize lazy Linear layers
    log.info("Initializing lazy layers...")
    with torch.no_grad():
        sample_batch = sample_batch.to(device)
        _ = model(sample_batch.x_dict, sample_batch.edge_index_dict)

    # 3. Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    model_save_path = ml_dir / "hgt_best.pt"

    # 4. Training loop
    log.info("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(preds, batch.y)

            loss.backward()
            optimizer.step()

            if batch_id % 50 == 0:
              log.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # 5. Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x_dict, batch.edge_index_dict)
                loss = criterion(preds, batch.y)
                val_loss += loss.item() * batch.num_graphs

                true_vals = torch.expm1(batch.y)
                pred_vals = torch.expm1(preds)
                val_mae += torch.abs(true_vals - pred_vals).sum().item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        log.info(
            "Epoch %03d | Train Loss (MSLE): %.4f | Val Loss (MSLE): %.4f | Val MAE: %.2f",
            epoch, train_loss, val_loss, val_mae,
        )

        # 6. Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            log.info(" -> Saved new best model to %s", model_save_path)

    # 7. Test evaluation
    log.info("--- Training complete. Evaluating on test set ---")
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

    log.info("Final Test Loss (MSLE): %.4f", test_loss)
    log.info("Final Test MAE (Absolute Count): %.2f", test_mae)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PopularityHGT on the extracted subgraph features.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--hidden-channels", type=int, default=HIDDEN_CHANNELS)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--config-hash", default=None,
                   help="features folder to use (default: most recently modified)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = _parse_args()
    train_hgt(
        base_path=a.base_path,
        dataset_name=a.dataset,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        hidden_channels=a.hidden_channels,
        num_heads=a.num_heads,
        num_layers=a.num_layers,
        dropout=a.dropout,
        num_workers=a.num_workers,
        config_hash=a.config_hash,
    )