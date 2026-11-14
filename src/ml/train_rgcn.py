import argparse
import json
import logging
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# from src.ml.dataset import get_dataloaders
from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders       # no config object needed

from src.ml.models.rgcn import PopularityRGCN, infer_schema

log = logging.getLogger(__name__)

# --- Default Configurations ---
DEFAULT_BASE_PATH = "."
DEFAULT_DATASET_NAME = "flickr"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.3
DEFAULT_EDGE_DROPOUT = 0.1
DEFAULT_PATIENCE = 15
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_METADATA_BATCHES = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """No weight decay on biases and LayerNorm parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def forward_batch(model, batch):
    batch_dict = getattr(batch, "batch_dict", None) or None
    preds = model(batch.x_dict, batch.edge_index_dict, batch_dict)
    y = batch.y.view(-1).float()
    if preds.shape != y.shape:
        raise ValueError(
            f"Prediction/target shape mismatch {tuple(preds.shape)} vs {tuple(y.shape)}. "
            "The model expects exactly one target node per graph."
        )
    return preds, y


def check_schema(model, loader, name: str) -> None:
    """Warn if a loader contains relations the model was not built with (they would be ignored)."""
    batch = next(iter(loader))
    unknown = [tuple(et) for et in batch.edge_types if tuple(et) not in set(model.edge_types)]
    if unknown:
        log.warning(f"[{name}] edge types not in model schema and therefore ignored: {unknown}")


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds, y = forward_batch(model, batch)
        loss = criterion(preds, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * y.numel()
        n += y.numel()
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, baseline_mean: float):
    """MSLE (log scale), MAE (count scale), Spearman, and the predict-the-train-mean MSLE."""
    model.eval()
    preds_all, y_all = [], []
    for batch in loader:
        batch = batch.to(device)
        preds, y = forward_batch(model, batch)
        preds_all.append(preds.cpu())
        y_all.append(y.cpu())
    p, y = torch.cat(preds_all), torch.cat(y_all)

    msle = F.mse_loss(p, y).item()
    mae = (torch.expm1(y) - torch.expm1(p.clamp(0, 25))).abs().mean().item()
    baseline_msle = F.mse_loss(torch.full_like(y, baseline_mean), y).item()
    rho = spearmanr(p.numpy(), y.numpy())[0] if len(y) > 2 else float("nan")
    rho = 0.0 if np.isnan(rho) else float(rho)
    return {"msle": msle, "mae": mae, "spearman": rho, "baseline_msle": baseline_msle}


def train_rgcn():
    parser = argparse.ArgumentParser(description="Train R-GCN for Popularity Prediction")

    # Environment Arguments
    parser.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH, help="Base path for the project")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, help="Target dataset name (e.g., flickr, digg)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory inside base_path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    # Model Hyperparameters
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Hidden dimension of the network")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of message passing layers")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability")
    parser.add_argument("--edge_dropout", type=float, default=DEFAULT_EDGE_DROPOUT, help="DropEdge probability (train only)")
    parser.add_argument("--no_degree_features", action="store_true", help="Disable log-degree input features")
    parser.add_argument("--no_pooling", action="store_true", help="Disable pooled readout of non-target nodes")

    # Optimisation Hyperparameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience (epochs)")
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP, help="Max grad norm (0 disables)")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Training loss on log1p target")
    parser.add_argument("--huber_delta", type=float, default=DEFAULT_HUBER_DELTA, help="Huber delta (loss=huber)")
    parser.add_argument("--metadata_batches", type=int, default=DEFAULT_METADATA_BATCHES, help="Batches used to infer the graph schema")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Setup directories
    base_dir = Path(args.base_path)
    ml_dir = base_dir / args.output_dir / args.dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        base_path=args.base_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check your splits and extraction.")

    # 2. Build model from the UNION schema of several batches (no lazy init needed)
    schema = infer_schema(islice(train_loader, args.metadata_batches))
    log.info(f"Schema | nodes: {schema['in_channels']} | edge types: {schema['edge_types']}")

    train_targets = torch.cat([b.y.view(-1).float() for b in train_loader])
    train_mean = train_targets.mean().item()
    log.info(f"Train target (log1p) mean={train_mean:.4f} std={train_targets.std().item():.4f} n={len(train_targets)}")

    model_kwargs = dict(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        use_degree_features=not args.no_degree_features,
        use_pooling=not args.no_pooling,
    )
    model = PopularityRGCN(
        metadata=(schema["node_types"], schema["edge_types"]),
        in_channels=schema["in_channels"],
        **model_kwargs,
    ).to(device)
    model.init_output_bias(train_mean)
    check_schema(model, val_loader, "val")
    check_schema(model, test_loader, "test")
    log.info(f"Initialized RGCN ({sum(p.numel() for p in model.parameters()):,} params): {model_kwargs}")

    # 3. Optimizer, scheduler, loss
    # Model selection always uses MSLE so runs stay comparable with XGBoost, whatever the training loss.
    criterion = nn.MSELoss() if args.loss == "mse" else nn.HuberLoss(delta=args.huber_delta)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3)
    )
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    model_save_path = ml_dir / "rgcn_best.pt"

    # 4. Training Loop
    log.info("Starting R-GCN training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val = evaluate(model, val_loader, device, train_mean)
        scheduler.step(val["msle"])

        log.info(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val MSLE: {val['msle']:.4f} (mean-baseline {val['baseline_msle']:.4f}) | "
            f"Val MAE: {val['mae']:.2f} | Val Spearman: {val['spearman']:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val["msle"] < best_val:
            best_val, best_epoch, bad_epochs = val["msle"], epoch, 0
            torch.save(
                {"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "schema": schema},
                model_save_path,
            )
            log.info(f" -> Saved new best model to {model_save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    # 5. Test Evaluation with the best checkpoint
    log.info("--- Training Complete. Evaluating on Test Set ---")
    ckpt = torch.load(model_save_path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    val = evaluate(model, val_loader, device, train_mean)
    test = evaluate(model, test_loader, device, train_mean)

    results = {
        "dataset": args.dataset_name,
        "model": "rgcn",
        "best_epoch": best_epoch,
        "config": vars(args),
        "val": val,
        "test": test,
    }
    with open(ml_dir / "rgcn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL LOSS STATS ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: R-GCN (hidden={args.hidden_channels}, layers={args.num_layers}, lr={args.lr}, "
          f"dropout={args.dropout}, edge_dropout={args.edge_dropout}, loss={args.loss})")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation Loss (MSLE): {val['msle']:.4f} | MAE: {val['mae']:.2f} | Spearman: {val['spearman']:.3f}")
    print(f"Test Loss (MSLE): {test['msle']:.4f} | MAE: {test['mae']:.2f} | Spearman: {test['spearman']:.3f}")
    print(f"Test MSLE of predicting the train mean: {test['baseline_msle']:.4f}")
    print("========================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_rgcn()