"""Train PopularityHGT (HGT or RGCN backbone) on the sharded subgraphs. No config files.

Hyperparameters live in the DEFAULTS block below and in models/hgt.py; every one of them
can also be set from the command line (run with --help to see the flags).

Pipeline (run these first, in order):
    1. splitter.py     -> data/splits/<dataset>/{split_meta,labels}.parquet
    2. features.py     -> outputs/<dataset>/samples/<hash>/shard_*.pt
    3. this file       -> outputs/<dataset>/ml/<conv><tag>_best.pt (+ _config.json)

Usage
-----
    python -m src.ml.train --dataset flickr --epochs 30
    python -m src.ml.train --dataset digg --conv rgcn --num-bases 2 --edge-dropout 0.2 \
        --conv-dropout 0.3 --weight-decay 1e-2 --patience 8 --loss huber
    python -m src.ml.train --help        # list all tunable flags

Loss is MSE (or Huber) on log1p(label). Best epoch is chosen by validation MSLE, whatever
the training loss is, so runs with different --loss values stay comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# from src.ml.dataset import get_dataloaders
from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders       # no config object needed
from src.ml.models.hgt import (
    PopularityHGT, get_batch_dict, DEFAULT_METADATA,
    DROPOUT, HIDDEN_CHANNELS, NUM_HEADS, NUM_LAYERS,
    CONV, NUM_BASES, INPUT_DROPOUT, CONV_DROPOUT, EDGE_DROPOUT, NORM, ACT, READOUT,
    HEAD_LAYERS, HEAD_HIDDEN, OUTPUT_ACT,
    CONV_CHOICES, NORM_CHOICES, ACT_CHOICES, READOUT_CHOICES, OUTPUT_ACT_CHOICES,
)

log = logging.getLogger(__name__)

# ============================================================================ #
# TRAINING DEFAULTS  (override with CLI flags, e.g. --lr 1e-4)
# ============================================================================ #
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0          # dataloader workers (0 = load in the main process)
SEED = 42
CLAMP_MAX_LOG = 20.0     # expm1 is applied to predictions clamped to [0, this] -> MAE can't overflow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, wd_split: bool):
    params = [p for p in model.parameters() if p.requires_grad]
    if not wd_split:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # no weight decay on biases / norm parameters (1-D tensors)
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    r = torch.empty(x.numel(), dtype=torch.float)
    r[order] = torch.arange(x.numel(), dtype=torch.float)
    return r


@torch.no_grad()
def evaluate(model, loader, criterion, device, clamp_max_log: float) -> dict:
    """Loss (training criterion), MSLE, MAE on counts, R2 and Spearman on log1p scale."""
    model.eval()
    n, loss_sum, se_sum, abs_sum = 0, 0.0, 0.0, 0.0
    all_p, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x_dict, batch.edge_index_dict, get_batch_dict(batch))
        g = batch.num_graphs
        n += g
        loss_sum += criterion(preds, batch.y).item() * g
        se_sum += ((preds - batch.y) ** 2).sum().item()
        pred_counts = torch.expm1(preds.clamp(0.0, clamp_max_log))
        abs_sum += torch.abs(torch.expm1(batch.y) - pred_counts).sum().item()
        all_p.append(preds.detach().cpu())
        all_y.append(batch.y.detach().cpu())
    p, y = torch.cat(all_p), torch.cat(all_y)
    sst = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ((p - y) ** 2).sum().item() / sst if sst > 0 else float("nan")
    rho = torch.corrcoef(torch.stack([_rank(p), _rank(y)]))[0, 1].item() if n > 1 else float("nan")
    return {"loss": loss_sum / n, "msle": se_sum / n, "mae": abs_sum / n, "r2": r2, "spearman": rho}


@torch.no_grad()
def export_embeddings(model, loaders: dict, device, path: Path) -> None:
    """Save pre-head graph embeddings + predictions + targets per split (for XGBoost stacking)."""
    model.eval()
    out = {}
    for split, loader in loaders.items():
        embs, preds, ys = [], [], []
        for batch in loader:
            batch = batch.to(device)
            pred, emb = model(batch.x_dict, batch.edge_index_dict, get_batch_dict(batch), return_embedding=True)
            embs.append(emb.cpu())
            preds.append(pred.cpu())
            ys.append(batch.y.cpu())
        out[split] = {"emb": torch.cat(embs), "pred_log1p": torch.cat(preds), "y_log1p": torch.cat(ys)}
    torch.save(out, path)
    log.info("Saved embeddings to %s", path)


def train_hgt(
    base_path: str | Path = ".",
    dataset_name: str = "flickr",
    *,
    output_dir: str = "outputs",
    # --- optimisation ---
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    wd_split: bool = False,
    loss: str = "mse",
    huber_delta: float = 1.0,
    scheduler: str = "none",
    grad_clip: float = 0.0,
    patience: int = 0,
    seed: int = SEED,
    clamp_max_log: float = CLAMP_MAX_LOG,
    # --- model ---
    conv: str = CONV,
    hidden_channels: int = HIDDEN_CHANNELS,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    num_bases: int | None = NUM_BASES,
    norm: str = NORM,
    act: str = ACT,
    residual: bool = True,
    readout: str = READOUT,
    dropout: float = DROPOUT,
    input_dropout: float = INPUT_DROPOUT,
    conv_dropout: float = CONV_DROPOUT,
    edge_dropout: float = EDGE_DROPOUT,
    head_layers: int = HEAD_LAYERS,
    head_hidden: int | None = HEAD_HIDDEN,
    output_act: str = OUTPUT_ACT,
    schema: str = "batch",
    # --- data / io ---
    num_workers: int = NUM_WORKERS,
    config_hash: str | None = None,   # None = latest folder in samples/
    tag: str = "",
    export_emb: bool = False,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s | Batch Size: %d | conv=%s | seed=%d", device, batch_size, conv, seed)

    base_path = Path(base_path)
    ml_dir = base_path / output_dir / dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{conv}{tag}"
    model_save_path = ml_dir / f"{run_name}_best.pt"

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
    metadata = DEFAULT_METADATA if schema == "fixed" else sample_batch.metadata()
    model = PopularityHGT(
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        conv=conv,
        num_bases=num_bases or None,
        input_dropout=input_dropout,
        conv_dropout=conv_dropout,
        edge_dropout=edge_dropout,
        norm=norm,
        act=act,
        residual=residual,
        readout=readout,
        head_layers=head_layers,
        head_hidden=head_hidden or None,
        output_act=output_act,
    ).to(device)

    # Dummy forward pass to initialize lazy Linear layers (eval mode: no edge dropout)
    log.info("Initializing lazy layers...")
    model.eval()
    with torch.no_grad():
        sample_batch = sample_batch.to(device)
        _ = model(sample_batch.x_dict, sample_batch.edge_index_dict, get_batch_dict(sample_batch))
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    # 3. Optimizer, loss, scheduler (created after lazy init so all parameters exist)
    criterion = nn.HuberLoss(delta=huber_delta) if loss == "huber" else nn.MSELoss()
    optimizer = build_optimizer(model, lr, weight_decay, wd_split)
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.01)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        sched = None

    best_val, best_epoch, bad_epochs = float("inf"), 0, 0

    # 4. Training loop
    log.info("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            preds = model(batch.x_dict, batch.edge_index_dict, get_batch_dict(batch))
            loss_val = criterion(preds, batch.y)

            loss_val.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if batch_idx % 50 == 0:
                log.info("Epoch %d | Batch %d/%d | Loss: %.4f", epoch, batch_idx, len(train_loader), loss_val.item())
            train_loss += loss_val.item() * batch.num_graphs
            seen += batch.num_graphs
        train_loss /= max(1, seen)

        # 5. Validation
        val = evaluate(model, val_loader, criterion, device, clamp_max_log)
        log.info(
            "Epoch %03d | Train Loss: %.4f | Val MSLE: %.4f | Val MAE: %.2f | Val R2: %.3f | Val Spearman: %.3f | LR: %.2e",
            epoch, train_loss, val["msle"], val["mae"], val["r2"], val["spearman"], optimizer.param_groups[0]["lr"],
        )

        if sched is not None:
            sched.step(val["msle"]) if scheduler == "plateau" else sched.step()

        # 6. Checkpoint best model + early stopping (both on validation MSLE)
        if val["msle"] < best_val:
            best_val, best_epoch, bad_epochs = val["msle"], epoch, 0
            torch.save(model.state_dict(), model_save_path)
            log.info(" -> Saved new best model to %s", model_save_path)
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    # 7. Test evaluation
    log.info("--- Training complete. Evaluating on test set ---")
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, criterion, device, clamp_max_log)
    log.info("Final Test MSLE: %.4f", test["msle"])
    log.info("Final Test MAE (Absolute Count): %.2f", test["mae"])
    log.info("Final Test R2 (log1p): %.3f | Spearman: %.3f", test["r2"], test["spearman"])

    results = {
        "dataset": dataset_name, "run_name": run_name, "n_params": n_params,
        "best_epoch": best_epoch, "best_val_msle": best_val,
        "test": test,
        "hparams": {
            "conv": conv, "hidden_channels": hidden_channels, "num_heads": num_heads, "num_layers": num_layers,
            "num_bases": num_bases, "norm": norm, "act": act, "residual": residual, "readout": readout,
            "dropout": dropout, "input_dropout": input_dropout, "conv_dropout": conv_dropout,
            "edge_dropout": edge_dropout, "head_layers": head_layers, "head_hidden": head_hidden,
            "output_act": output_act, "schema": schema, "lr": lr, "weight_decay": weight_decay,
            "wd_split": wd_split, "loss": loss, "huber_delta": huber_delta, "scheduler": scheduler,
            "grad_clip": grad_clip, "patience": patience, "batch_size": batch_size, "epochs": epochs,
            "seed": seed,
        },
    }
    with open(ml_dir / f"{run_name}_config.json", "w") as f:
        json.dump(results, f, indent=2)

    if export_emb:
        export_embeddings(
            model, {"train": train_loader, "val": val_loader, "test": test_loader},
            device, ml_dir / f"{run_name}_embeddings.pt",
        )
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PopularityHGT / RGCN on the extracted subgraph features.")
    # data / io
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--config-hash", default=None,
                   help="features folder to use (default: most recently modified)")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--tag", default="", help="suffix for checkpoint/config names, e.g. _trial3")
    p.add_argument("--export-embeddings", dest="export_emb", action="store_true",
                   help="save graph embeddings for train/val/test after training (XGBoost stacking)")
    p.add_argument("--seed", type=int, default=SEED)
    # optimisation
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--wd-split", action="store_true", help="no weight decay on biases / norm params")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off)")
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--clamp-max-log", type=float, default=CLAMP_MAX_LOG)
    # model
    p.add_argument("--conv", choices=list(CONV_CHOICES), default=CONV)
    p.add_argument("--hidden-channels", type=int, default=HIDDEN_CHANNELS)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS, help="HGT only")
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--num-bases", type=int, default=None, help="RGCN basis decomposition (omit = full weights)")
    p.add_argument("--norm", choices=list(NORM_CHOICES), default=NORM)
    p.add_argument("--act", choices=list(ACT_CHOICES), default=ACT)
    p.add_argument("--no-residual", dest="residual", action="store_false")
    p.add_argument("--readout", choices=list(READOUT_CHOICES), default=READOUT)
    p.add_argument("--dropout", "--head-dropout", dest="dropout", type=float, default=DROPOUT,
                   help="dropout inside the MLP head")
    p.add_argument("--input-dropout", type=float, default=INPUT_DROPOUT)
    p.add_argument("--conv-dropout", type=float, default=CONV_DROPOUT)
    p.add_argument("--edge-dropout", type=float, default=EDGE_DROPOUT)
    p.add_argument("--head-layers", type=int, default=HEAD_LAYERS)
    p.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN, help="default: hidden_channels // 2")
    p.add_argument("--output-act", choices=list(OUTPUT_ACT_CHOICES), default=OUTPUT_ACT)
    p.add_argument("--schema", choices=["batch", "fixed"], default="batch",
                   help="'fixed' builds the model from the shared 2-node/4-edge schema instead of the first batch")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    kwargs = vars(_parse_args())
    base_path = kwargs.pop("base_path")
    dataset = kwargs.pop("dataset")
    train_hgt(base_path=base_path, dataset_name=dataset, **kwargs)