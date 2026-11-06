"""PyG dataset reading the sharded per-post subgraphs produced by src/data/features.py.

No config files: everything this module needs is passed in as plain arguments
(base_path, dataset_name, config_hash). Hyperparameters live in the block below.

Assumes features.py already ran:
    data/splits/<dataset>/split_meta.parquet   (columns: post_id, split, ...)
    data/splits/<dataset>/labels.parquet       (columns: post_id, label)
    <base_path>/<OUTPUT_DIR>/<dataset>/samples/<config_hash>/shard_*.pt
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)

# ============================================================================ #
# HYPERPARAMETERS / SETTINGS  (change here, or pass arguments to get_dataloaders)
# ============================================================================ #
OUTPUT_DIR = "outputs"   # must match features.OUTPUT_DIR
# post_x has 3 + n_time_bins columns (see features.post_feature_names);
# with the features.py default n_time_bins=4 this is 8.
POST_FEATURE_DIM = 8
USER_FEATURE_DIM = 6   # = len(features.USER_FEATURE_NAMES)


def to_target(y: float) -> float:
    """Log-transform the ground truth for stable MSLE/MAPE training."""
    return float(np.log1p(y))


def find_latest_hash_dir(samples_dir: Path) -> str:
    """Return the name of the most recently written feature-hash folder."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"Missing samples directory {samples_dir}. Run features.py (extract_all) first."
        )
    subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hashed feature directories found in {samples_dir}")
    return max(subdirs, key=lambda d: d.stat().st_mtime).name


class PostGraphDataset(Dataset):
    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        split: str,
        config_hash: str | None = None,
        *,
        output_dir: str = OUTPUT_DIR,
        post_feature_dim: int = POST_FEATURE_DIM,
        user_feature_dim: int = USER_FEATURE_DIM,
    ):
        """
        Args:
            base_path: project root (the same base_path passed to features.extract_all).
            dataset_name: dataset key, e.g. "flickr".
            split: "train", "val" or "test" (column `split` in split_meta.parquet).
            config_hash: folder name under samples/ produced by extract_all.
                         None -> the most recently modified folder is used.
            output_dir: must match features.OUTPUT_DIR.
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.split = split
        self.post_feature_dim = post_feature_dim
        self.user_feature_dim = user_feature_dim

        self.splits_dir = self.base_path / "data" / "splits" / dataset_name
        samples_dir = self.base_path / output_dir / dataset_name / "samples"
        if config_hash is None:
            config_hash = find_latest_hash_dir(samples_dir)
        self.ml_dir = samples_dir / config_hash

        if not self.ml_dir.exists():
            raise FileNotFoundError(
                f"Missing extracted features at {self.ml_dir}. Run features.py first."
            )
        if not (self.ml_dir / "index.parquet").exists():
            raise RuntimeError(
                f"Feature extraction at {self.ml_dir} did not finish "
                f"(index.parquet is missing). Re-run features.py - finished shards "
                f"are skipped, so it will resume where it stopped."
            )

        # Filter posts strictly by the requested split
        meta = pd.read_parquet(self.splits_dir / "split_meta.parquet")
        self.split_posts = set(meta[meta["split"] == split]["post_id"])

        self.labels = pd.read_parquet(self.splits_dir / "labels.parquet").set_index("post_id")

        idx_df = pd.read_parquet(self.ml_dir / "index.parquet")
        self.index = idx_df[idx_df["post_id"].isin(self.split_posts)].reset_index(drop=True)

        # Memory-safe shard caching (one shard held in RAM at a time)
        self.shard_cache = {}
        self.current_shard_id = None

    def len(self) -> int:
        return len(self.index)

    def get(self, idx: int) -> HeteroData:
        row = self.index.iloc[idx]
        post_id = row["post_id"]
        shard_id = row["shard"]

        # Load shard on demand to save RAM
        if shard_id != self.current_shard_id:
            shard_file = self.ml_dir / f"shard_{shard_id}.pt"
            self.shard_cache = torch.load(shard_file, weights_only=False) if shard_file.exists() else {}
            self.current_shard_id = shard_id

        rec = self.shard_cache.get(post_id, None)
        y_val = self.labels.loc[post_id, "label"]

        data = HeteroData()
        data.y = torch.tensor([to_target(y_val)], dtype=torch.float32)

        if rec is not None:
            data["user"].x = torch.tensor(rec["user_x"], dtype=torch.float32)
            data["post"].x = torch.tensor(rec["post_x"], dtype=torch.float32)

            # Explicitly tell PyG the node counts to prevent batching inference errors
            data["user"].num_nodes = rec["user_x"].shape[0]
            data["post"].num_nodes = rec["post_x"].shape[0]

            # Forward edges
            data["user", "follows", "user"].edge_index = torch.tensor(rec["follows"], dtype=torch.long)
            data["user", "interacts", "post"].edge_index = torch.tensor(rec["interacts"], dtype=torch.long)

            if "interacts_attr" in rec and len(rec["interacts_attr"]) > 0:
                data["user", "interacts", "post"].edge_attr = torch.tensor(
                    rec["interacts_attr"], dtype=torch.float32
                )

            # Reverse edges (critical for HGT message passing from post -> user)
            data["post", "rev_interacts", "user"].edge_index = torch.tensor(
                np.flip(rec["interacts"], axis=0).copy(), dtype=torch.long
            )
            data["user", "followed_by", "user"].edge_index = torch.tensor(
                np.flip(rec["follows"], axis=0).copy(), dtype=torch.long
            )
        else:
            # Fallback must explicitly define missing nodes and all edge permutations
            data["post"].x = torch.zeros((1, self.post_feature_dim), dtype=torch.float32)
            data["user"].x = torch.zeros((0, self.user_feature_dim), dtype=torch.float32)

            data["post"].num_nodes = 1
            data["user"].num_nodes = 0

            data["user", "follows", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["user", "interacts", "post"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["post", "rev_interacts", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["user", "followed_by", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data


def get_dataloaders(
    base_path: str | Path,
    dataset_name: str,
    *,
    config_hash: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    output_dir: str = OUTPUT_DIR,
):
    """PyG DataLoaders with automatic HeteroData batching. No config object needed."""
    log.info("Loading sharded datasets for %s (Hash: %s)...", dataset_name, config_hash or "<latest>")

    train_dataset = PostGraphDataset(base_path, dataset_name, "train", config_hash, output_dir=output_dir)
    val_dataset = PostGraphDataset(base_path, dataset_name, "val", config_hash, output_dir=output_dir)
    test_dataset = PostGraphDataset(base_path, dataset_name, "test", config_hash, output_dir=output_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    log.info(
        "Batches (size %d) -> Train: %d, Val: %d, Test: %d",
        batch_size, len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader