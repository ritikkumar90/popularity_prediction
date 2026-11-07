"""In-RAM variant of PostGraphDataset (Plan A), used by train.py for the HGT model.

Why this file exists
--------------------
dataset.PostGraphDataset reloads a whole shard (1000 posts) for almost every sample when
shuffle=True. This module keeps that class untouched (so XGBoost / RGCN code that imports
dataset.py is not affected) and adds a subclass that:

  * reads every shard ONCE at startup (all three splits in a single pass),
  * keeps only the posts of each split, already converted to HeteroData, in CPU RAM,
  * makes get() a plain list lookup, so no disk access happens during training.

Put this file next to dataset.py (src/ml/dataset_ram.py).
Nothing in dataset.py or features.py needs to change.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.ml.dataset import OUTPUT_DIR, PostGraphDataset

log = logging.getLogger(__name__)


class PostGraphDatasetRAM(PostGraphDataset):
    def __init__(
        self,
        base_path,
        dataset_name: str,
        split: str,
        config_hash: str | None = None,
        *,
        output_dir: str = OUTPUT_DIR,
        keep_edge_attr: bool = False,
        preload: bool = True,
        **kwargs,
    ):
        """Same arguments as PostGraphDataset, plus:

        keep_edge_attr: also load `interacts_attr` as edge_attr. HGTConv ignores it, so the
                        default is False (smaller and faster). Set True for parity with the
                        original dataset.
        preload: load the shards right now. get_dataloaders_ram() sets this to False and
                 loads all three splits together so each shard is read only once.
        """
        super().__init__(base_path, dataset_name, split, config_hash, output_dir=output_dir, **kwargs)
        self.keep_edge_attr = keep_edge_attr

        # Arrays prepared once (replaces the slow per-sample pandas .iloc / .loc lookups).
        self._post_ids = self.index["post_id"].to_numpy()
        self._shard_ids = self.index["shard"].to_numpy()
        labels = self.labels["label"].reindex(self._post_ids).to_numpy(dtype=np.float64)
        if np.isnan(labels).any():
            raise ValueError(f"[{split}] some posts have no label in labels.parquet")
        self._ys = np.log1p(labels).astype(np.float32)   # same transform as dataset.to_target

        self.data_list: list = [None] * len(self._post_ids)
        self.shard_cache = {}   # not used here; frees the memory the base class would hold
        if preload:
            preload_datasets([self])

    # ------------------------------------------------------------------ #
    def _build(self, rec, y: float) -> HeteroData:
        data = HeteroData()
        data.y = torch.tensor([y], dtype=torch.float32)

        if rec is None:
            # Same fallback as the original dataset: empty graph with one post node.
            data["post"].x = torch.zeros((1, self.post_feature_dim), dtype=torch.float32)
            data["user"].x = torch.zeros((0, self.user_feature_dim), dtype=torch.float32)
            data["post"].num_nodes = 1
            data["user"].num_nodes = 0
            for et in (("user", "follows", "user"), ("user", "interacts", "post"),
                       ("post", "rev_interacts", "user"), ("user", "followed_by", "user")):
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
            return data

        user_x = torch.as_tensor(rec["user_x"], dtype=torch.float32)
        post_x = torch.as_tensor(rec["post_x"], dtype=torch.float32)
        data["user"].x = user_x
        data["post"].x = post_x
        data["user"].num_nodes = user_x.shape[0]
        data["post"].num_nodes = post_x.shape[0]

        follows = torch.as_tensor(rec["follows"], dtype=torch.long)
        interacts = torch.as_tensor(rec["interacts"], dtype=torch.long)
        data["user", "follows", "user"].edge_index = follows
        data["user", "interacts", "post"].edge_index = interacts
        data["post", "rev_interacts", "user"].edge_index = interacts.flip(0)
        data["user", "followed_by", "user"].edge_index = follows.flip(0)

        if self.keep_edge_attr and "interacts_attr" in rec and len(rec["interacts_attr"]) > 0:
            data["user", "interacts", "post"].edge_attr = torch.as_tensor(
                rec["interacts_attr"], dtype=torch.float32
            )
        return data

    def get(self, idx: int) -> HeteroData:
        return self.data_list[idx]


def preload_datasets(datasets: list[PostGraphDatasetRAM]) -> None:
    """Read every needed shard exactly once and fill all given datasets from it."""
    ml_dir = datasets[0].ml_dir
    if any(d.ml_dir != ml_dir for d in datasets):
        raise ValueError("All datasets must read from the same features folder")

    # For each dataset: positions sorted by shard, so one shard's posts are a contiguous slice.
    lookups = []
    for d in datasets:
        order = np.argsort(d._shard_ids, kind="stable")
        lookups.append((order, d._shard_ids[order]))

    shard_ids = np.unique(np.concatenate([d._shard_ids for d in datasets]))
    log.info("Preloading %d shards from %s into RAM (one pass)...", len(shard_ids), ml_dir)

    for n, sid in enumerate(shard_ids, start=1):
        f = ml_dir / f"shard_{sid}.pt"
        shard = torch.load(f, weights_only=False) if f.exists() else {}
        for d, (order, sorted_sh) in zip(datasets, lookups):
            lo = np.searchsorted(sorted_sh, sid, side="left")
            hi = np.searchsorted(sorted_sh, sid, side="right")
            for i in order[lo:hi]:
                d.data_list[i] = d._build(shard.get(int(d._post_ids[i])), float(d._ys[i]))
        del shard   # drop the unused (other-split) posts before the next shard
        if n % 10 == 0 or n == len(shard_ids):
            log.info("  loaded %d/%d shards", n, len(shard_ids))

    for d in datasets:
        log.info("  %s: %d posts in RAM", d.split, len(d.data_list))


def get_dataloaders_ram(
    base_path,
    dataset_name: str,
    *,
    config_hash: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    output_dir: str = OUTPUT_DIR,
    eval_batch_size: int | None = None,
    pin_memory: bool | None = None,
    keep_edge_attr: bool = False,
):
    """Drop-in replacement for dataset.get_dataloaders (same arguments, same return value)."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_batch_size = eval_batch_size or batch_size

    log.info("Loading in-RAM datasets for %s (Hash: %s)...", dataset_name, config_hash or "<latest>")
    datasets = [
        PostGraphDatasetRAM(base_path, dataset_name, split, config_hash, output_dir=output_dir,
                            keep_edge_attr=keep_edge_attr, preload=False)
        for split in ("train", "val", "test")
    ]
    preload_datasets(datasets)
    train_ds, val_ds, test_ds = datasets

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common)

    log.info("Batches -> Train: %d (bs %d), Val: %d, Test: %d (bs %d)",
             len(train_loader), batch_size, len(val_loader), len(test_loader), eval_batch_size)
    return train_loader, val_loader, test_loader