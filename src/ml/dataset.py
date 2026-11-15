"""Datasets for the three models, reading what src/ml/features.py wrote.

    HGT / R-GCN   get_dataloaders(...)  -> PyG loaders of HeteroData (one graph per story)
    XGBoost       load_tabular(...)     -> (X, y, post_ids) from tabular_features.parquet

Usage
-----
    from src.ml.dataset import get_dataloaders, load_tabular
    train_loader, val_loader, test_loader = get_dataloaders(".", "digg", batch_size=32)
    X_tr, y_tr, ids_tr = load_tabular(".", "digg", "train")

Needs (written by the splitter and by features.py)
    data/splits/<dataset>/split_meta.parquet     post_id, split, observed_count, ...
    data/splits/<dataset>/labels.parquet         post_id, label, growth
    <output_dir>/<dataset>/samples/<hash>/       shard_*.pt, index.parquet, tabular_features.parquet,
                                                 sample_config.json

Everything is read ONCE at start-up and kept in RAM (Digg is small), so there is no disk access
while training and no separate "RAM" variant is needed.

What one sample (HeteroData) contains
-------------------------------------
    user.x            [n_users, N_USER_FEATURES]   see features.USER_FEATURE_NAMES
    post.x            [1, n_post_features]         see features.post_feature_names
    (user, follows, user)           information flows source -> target (see features.py)
    (user, followed_by, user)       the same edges reversed
    (user, interacts, post)         voter -> story; edge_attr = adoption_time_frac (if keep_edge_attr)
    (post, rev_interacts, user)     reversed, so the story can send messages back to its voters
    y        [1]  target, log1p of `label` (total votes at the horizon) or of `growth`, see `target`
    log_obs  [1]  log1p(votes observed at t_obs). Use it as an offset: predict y - log_obs, or add it
                  to the output, so the model only has to learn the growth.
    post_id  [1]  to map predictions back to stories

Every edge type is present in every sample (possibly empty), so batches always have the same metadata.
Node features are not standardised here; fit a scaler on the train split if a model needs one.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

from src.ml.features import N_USER_FEATURES, OUTPUT_DIR, SampleConfig, n_post_features, post_feature_names

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
TARGETS = ("label", "growth")


def to_target(y: float) -> float:
    """Log-transform of a vote count (stable for MSE / MSLE training)."""
    return float(np.log1p(y))


# --------------------------------------------------------------------------- #
# Finding the feature run
# --------------------------------------------------------------------------- #

def find_latest_hash_dir(samples_dir: Path) -> str:
    """Name of the most recently written FINISHED feature run (one that has index.parquet)."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing {samples_dir}. Run features.py (extract_all) first.")
    runs = [d for d in samples_dir.iterdir() if d.is_dir() and (d / "index.parquet").exists()]
    if not runs:
        raise FileNotFoundError(f"No finished feature run in {samples_dir}. Run features.py first.")
    return max(runs, key=lambda d: d.stat().st_mtime).name


def _resolve_run(base_path, dataset_name, config_hash, output_dir) -> Path:
    samples_dir = Path(base_path) / output_dir / dataset_name / "samples"
    run = samples_dir / (config_hash or find_latest_hash_dir(samples_dir))
    if not (run / "index.parquet").exists():
        raise RuntimeError(f"Feature run {run} is missing or unfinished (no index.parquet). Re-run features.py.")
    cfg = json.loads((run / "sample_config.json").read_text())
    if cfg.get("version") != SampleConfig().version:
        raise RuntimeError(
            f"{run} was made by another version of features.py (version {cfg.get('version')}, "
            f"expected {SampleConfig().version}). Re-run features.py."
        )
    return run


def _split_tables(base_path, dataset_name, split: str):
    """(meta of this split, labels indexed by post_id). Order of meta = order of the dataset."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    splits_dir = Path(base_path) / "data" / "splits" / dataset_name
    meta = pd.read_parquet(splits_dir / "split_meta.parquet")
    meta = meta[meta["split"] == split].reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"Split '{split}' is empty in split_meta.parquet")
    labels = pd.read_parquet(splits_dir / "labels.parquet").set_index("post_id")
    return meta, labels


def _targets(meta: pd.DataFrame, labels: pd.DataFrame, target: str, split: str):
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    lab = labels.reindex(meta["post_id"])
    if lab[target].isna().any():
        raise ValueError(f"[{split}] some stories have no entry in labels.parquet")
    y = np.log1p(lab[target].to_numpy(dtype=np.float64)).astype(np.float32)
    log_obs = np.log1p(meta["observed_count"].to_numpy(dtype=np.float64)).astype(np.float32)
    return y, log_obs


# --------------------------------------------------------------------------- #
# Graph dataset (HGT, R-GCN)
# --------------------------------------------------------------------------- #

class PostGraphDataset(Dataset):
    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        split: str,
        config_hash: str | None = None,
        *,
        output_dir: str = OUTPUT_DIR,
        target: str = "label",
        keep_edge_attr: bool = True,
        preload: bool = True,
    ):
        """
        split           "train", "val" or "test"
        config_hash     feature run folder under samples/; None = the latest finished run
        target          "label": y = log1p(votes at the horizon);  "growth": y = log1p(votes after t_obs)
        keep_edge_attr  put adoption_time_frac on the (user, interacts, post) edges
        preload         read the shards now. get_dataloaders() passes False and loads all three
                        splits together so every shard is read only once.
        """
        super().__init__()
        self.base_path, self.dataset_name, self.split = Path(base_path), dataset_name, split
        self.target, self.keep_edge_attr = target, bool(keep_edge_attr)
        self.ml_dir = _resolve_run(base_path, dataset_name, config_hash, output_dir)

        cfg = json.loads((self.ml_dir / "sample_config.json").read_text())
        self.post_feature_dim = len(post_feature_names(cfg["n_time_bins"]))
        self.user_feature_dim = N_USER_FEATURES

        meta, self.labels = _split_tables(base_path, dataset_name, split)
        index = pd.read_parquet(self.ml_dir / "index.parquet").set_index("post_id")
        missing = ~meta["post_id"].isin(index.index)
        if missing.any():
            raise RuntimeError(
                f"[{split}] {int(missing.sum())} stories of split_meta.parquet are not in the feature run "
                f"{self.ml_dir.name}. The splits changed after features.py ran: re-run features.py."
            )
        self.index = index.loc[meta["post_id"], ["shard"]].reset_index()
        self._post_ids = self.index["post_id"].to_numpy()
        self._shard_ids = self.index["shard"].to_numpy()
        self._y, self._log_obs = _targets(meta, self.labels, target, split)

        self.data_list: list[HeteroData | None] = [None] * len(self._post_ids)
        if preload:
            preload_datasets([self])

    def len(self) -> int:
        return len(self._post_ids)

    def get(self, idx: int) -> HeteroData:
        data = self.data_list[idx]
        if data is None:
            raise RuntimeError("Dataset not loaded yet (created with preload=False); call preload_datasets().")
        return data

    def _build(self, rec: dict, i: int) -> HeteroData:
        user_x = np.asarray(rec["user_x"], dtype=np.float32)
        post_x = np.asarray(rec["post_x"], dtype=np.float32)
        if user_x.ndim != 2 or user_x.shape[1] != self.user_feature_dim or post_x.shape != (1, self.post_feature_dim):
            raise RuntimeError(
                f"Feature shapes {user_x.shape} / {post_x.shape} do not match the expected "
                f"(n, {self.user_feature_dim}) / (1, {self.post_feature_dim}). Re-run features.py."
            )
        follows = torch.from_numpy(np.asarray(rec["follows"], dtype=np.int64))
        interacts = torch.from_numpy(np.asarray(rec["interacts"], dtype=np.int64))

        data = HeteroData()
        data.y = torch.tensor([self._y[i]], dtype=torch.float32)
        data.log_obs = torch.tensor([self._log_obs[i]], dtype=torch.float32)
        data.post_id = torch.tensor([int(self._post_ids[i])], dtype=torch.long)

        data["user"].x = torch.from_numpy(user_x)
        data["post"].x = torch.from_numpy(post_x)
        data["user"].num_nodes = user_x.shape[0]
        data["post"].num_nodes = 1

        data["user", "follows", "user"].edge_index = follows
        data["user", "followed_by", "user"].edge_index = follows.flip(0)
        data["user", "interacts", "post"].edge_index = interacts
        data["post", "rev_interacts", "user"].edge_index = interacts.flip(0)
        if self.keep_edge_attr:
            attr = np.asarray(rec["interacts_attr"], dtype=np.float32).reshape(-1, 1)
            data["user", "interacts", "post"].edge_attr = torch.from_numpy(attr)
        return data


def preload_datasets(datasets: list[PostGraphDataset]) -> None:
    """Read every needed shard exactly once and fill all given datasets from it."""
    ml_dir = datasets[0].ml_dir
    if any(d.ml_dir != ml_dir for d in datasets):
        raise ValueError("All datasets must read from the same feature run")
    shard_ids = np.unique(np.concatenate([d._shard_ids for d in datasets]))
    log.info("Loading %d shard(s) from %s into RAM ...", len(shard_ids), ml_dir)

    for sid in shard_ids:
        f = ml_dir / f"shard_{sid}.pt"
        if not f.exists():
            raise FileNotFoundError(f"Missing shard file {f}. Re-run features.py.")
        shard = torch.load(f, weights_only=False)
        for d in datasets:
            for i in np.nonzero(d._shard_ids == sid)[0]:
                rec = shard.get(int(d._post_ids[i]))
                if rec is None:
                    raise KeyError(f"Story {d._post_ids[i]} is listed in index.parquet but missing from {f.name}")
                d.data_list[int(i)] = d._build(rec, int(i))
        del shard

    for d in datasets:
        log.info("  %s: %d stories in RAM", d.split, len(d.data_list))


def get_dataloaders(
    base_path: str | Path,
    dataset_name: str,
    *,
    config_hash: str | None = None,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    output_dir: str = OUTPUT_DIR,
    target: str = "label",
    keep_edge_attr: bool = True,
    pin_memory: bool | None = None,
):
    """PyG DataLoaders (train shuffled, val/test not) with automatic HeteroData batching."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_batch_size = eval_batch_size or batch_size
    run = _resolve_run(base_path, dataset_name, config_hash, output_dir).name   # same run for all splits

    log.info("Loading datasets for %s (run %s, target=%s) ...", dataset_name, run, target)
    datasets = [
        PostGraphDataset(base_path, dataset_name, s, run, output_dir=output_dir, target=target,
                         keep_edge_attr=keep_edge_attr, preload=False)
        for s in SPLITS
    ]
    preload_datasets(datasets)
    train_ds, val_ds, test_ds = datasets

    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common)
    log.info("Batches -> train: %d (bs %d), val: %d, test: %d (bs %d)",
             len(train_loader), batch_size, len(val_loader), len(test_loader), eval_batch_size)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Tabular data (XGBoost)
# --------------------------------------------------------------------------- #

def load_tabular(
    base_path: str | Path,
    dataset_name: str,
    split: str,
    config_hash: str | None = None,
    *,
    output_dir: str = OUTPUT_DIR,
    target: str = "label",
):
    """Flat features of one split: (X DataFrame, y = log1p(target) as float32 array, post_ids array).

    Same stories, same order and same target as the graph dataset. `log_obs_count` is a column of X.
    """
    run = _resolve_run(base_path, dataset_name, config_hash, output_dir)
    meta, labels = _split_tables(base_path, dataset_name, split)
    tab = pd.read_parquet(run / "tabular_features.parquet").set_index("post_id")
    missing = ~meta["post_id"].isin(tab.index)
    if missing.any():
        raise RuntimeError(f"[{split}] {int(missing.sum())} stories are not in {run.name}: re-run features.py.")
    X = tab.loc[meta["post_id"]].reset_index(drop=True)
    y, _ = _targets(meta, labels, target, split)
    return X, y, meta["post_id"].to_numpy()