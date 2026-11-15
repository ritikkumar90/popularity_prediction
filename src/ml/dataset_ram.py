"""In-RAM dataset wrapper for HGT / R-GCN training, compatible with src/ml/dataset.py (v5).

In the updated src/ml/dataset.py, PostGraphDataset already preloads all shards into RAM
in a single pass. This module provides the `PostGraphDatasetRAM` and `get_dataloaders_ram`
entry points expected by train.py / test_hgt.py / compare_models.py, defaulting
`keep_edge_attr=False` (since PyG HGTConv ignores edge_attr) while forwarding all v5
features, targets ("label" or "growth"), and `log_obs` / `post_id` attributes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.ml.dataset import (
    OUTPUT_DIR,
    SPLITS,
    TARGETS,
    PostGraphDataset,
    _resolve_run,
    find_latest_hash_dir,
    load_tabular,
    preload_datasets,
    to_target,
)

log = logging.getLogger(__name__)


class PostGraphDatasetRAM(PostGraphDataset):
    """Subclass of PostGraphDataset with HGT-friendly defaults (keep_edge_attr=False)."""

    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        split: str,
        config_hash: str | None = None,
        *,
        output_dir: str = OUTPUT_DIR,
        target: str = "label",
        keep_edge_attr: bool = False,
        preload: bool = True,
        **kwargs,
    ):
        super().__init__(
            base_path=base_path,
            dataset_name=dataset_name,
            split=split,
            config_hash=config_hash,
            output_dir=output_dir,
            target=target,
            keep_edge_attr=keep_edge_attr,
            preload=preload,
        )


def get_dataloaders_ram(
    base_path: str | Path,
    dataset_name: str,
    *,
    config_hash: str | None = None,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    output_dir: str = OUTPUT_DIR,
    target: str = "label",
    keep_edge_attr: bool = False,
    pin_memory: bool | None = None,
):
    """Drop-in replacement for get_dataloaders with keep_edge_attr=False by default."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_batch_size = eval_batch_size or batch_size
    run = _resolve_run(base_path, dataset_name, config_hash, output_dir).name

    log.info("Loading in-RAM datasets for %s (Hash: %s, target=%s)...", dataset_name, run, target)
    datasets = [
        PostGraphDatasetRAM(
            base_path,
            dataset_name,
            split,
            run,
            output_dir=output_dir,
            target=target,
            keep_edge_attr=keep_edge_attr,
            preload=False,
        )
        for split in SPLITS
    ]
    preload_datasets(datasets)
    train_ds, val_ds, test_ds = datasets

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common)

    log.info(
        "Batches -> Train: %d (bs %d), Val: %d, Test: %d (bs %d)",
        len(train_loader),
        batch_size,
        len(val_loader),
        len(test_loader),
        eval_batch_size,
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "OUTPUT_DIR",
    "SPLITS",
    "TARGETS",
    "PostGraphDatasetRAM",
    "find_latest_hash_dir",
    "get_dataloaders_ram",
    "load_tabular",
    "preload_datasets",
    "to_target",
]