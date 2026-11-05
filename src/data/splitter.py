"""Time-aware train/val/test splitter, shared by all datasets.

Reads the canonical tables written by BaseAdapter (events, posts, edges) and writes
  split_meta.parquet   post_id, author, t0, t_obs, t_end, observed_count, split
  labels.parquet       post_id, label, growth
  split_manifest.json  counts, split boundaries, config hash

Definitions (per post)
----------------------
t_obs          = t0 + observation window     (what a model may see)
t_end          = t0 + horizon                (what is predicted)
observed_count = events with time <= t_obs
label          = events with time <= t_end   (total popularity at the horizon)
growth         = label - observed_count      (new events after t_obs)

Labels live in their own file so models that only need the post metadata
(e.g. HGT / RGCN graph builders, IC simulation) cannot read them by accident.

Config used: time.observation_window, time.horizon, time.split, time.min_observed_events,
time.drop_if_horizon_exceeds_data_end, time.purge_straddling_posts,
active.edge_time_available. For observation_window.mode == "fraction_of_duration" the
window is value * horizon.
"""
from __future__ import annotations

import json
import logging
from functools import cached_property

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)


class TimeSplitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.time_cfg = cfg.time
        self.edge_time_available = cfg.active.edge_time_available

        base = cfg.run.base_path
        self.processed_dir = base / "data" / "processed" / self.dataset_name
        self.splits_dir = base / "data" / "splits" / self.dataset_name
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self._obs_seconds, self._horizon_seconds = self._window_seconds()

        self._meta: pd.DataFrame | None = None
        self._ev_index = None
        self._warned_static = False

    # ------------------------------------------------------------------
    # Lazy loading (edges are only needed for snapshots, so they are not loaded to build splits)
    # ------------------------------------------------------------------
    @cached_property
    def posts(self) -> pd.DataFrame:
        p = pd.read_parquet(self.processed_dir / "posts.parquet")
        return p.sort_values(["t0", "post_id"], kind="stable").reset_index(drop=True)

    @cached_property
    def events(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.processed_dir / "events.parquet")
        if not ev["time"].is_monotonic_increasing:
            ev = ev.sort_values("time", kind="stable").reset_index(drop=True)
        return ev

    @cached_property
    def edges(self) -> pd.DataFrame:
        e = pd.read_parquet(self.processed_dir / "edges.parquet")
        if self.edge_time_available and not e["created_at"].is_monotonic_increasing:
            e = e.sort_values("created_at", kind="stable").reset_index(drop=True)
        return e

    def _window_seconds(self) -> tuple[int, int]:
        w = self.time_cfg.observation_window
        horizon = int(round(self.time_cfg.horizon.seconds))
        if w.mode == "fixed":
            obs = int(round(w.seconds))
        else:  # fraction_of_duration: a fraction of the prediction horizon
            obs = int(round(w.value * horizon))
        if not 0 < obs < horizon:
            raise ValueError(f"Observation window ({obs}s) must be positive and shorter than the horizon ({horizon}s)")
        return obs, horizon

    # ------------------------------------------------------------------
    # Building the splits
    # ------------------------------------------------------------------
    def create_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tc, sp = self.time_cfg, self.time_cfg.split
        log.info("--- Creating time-aware splits for %s ---", self.dataset_name)
        log.info("Observation window: %d s, horizon: %d s", self._obs_seconds, self._horizon_seconds)

        df = self.posts.copy()
        n_initial = len(df)
        log.info("Initial posts: %d", n_initial)
        df["t_obs"] = df["t0"] + self._obs_seconds
        df["t_end"] = df["t0"] + self._horizon_seconds

        # Posts whose horizon runs past the end of the data would get truncated labels.
        data_end = int(self.events["time"].max())
        if tc.drop_if_horizon_exceeds_data_end:
            df = df[df["t_end"] <= data_end]
            log.info("Posts whose horizon ends before the data end: %d", len(df))
        df = df.reset_index(drop=True)
        self._require_nonempty(df, "after dropping posts whose horizon exceeds the data end "
                                   "(the horizon may be too long for this dataset)")

        df = self._add_counts(df)
        df = df[df["observed_count"] >= tc.min_observed_events].reset_index(drop=True)
        log.info("Posts with >= %d events at t_obs: %d", tc.min_observed_events, len(df))
        self._require_nonempty(df, f"after requiring >= {tc.min_observed_events} observed events")

        # Chronological split on t0.
        train_t = float(df["t0"].quantile(sp.train))
        val_t = float(df["t0"].quantile(min(sp.train + sp.val, 1.0)))
        df["split"] = np.where(df["t0"] < train_t, "train", np.where(df["t0"] < val_t, "val", "test"))
        raw_counts = df["split"].value_counts().to_dict()
        log.info("Raw splits - %s", {k: int(raw_counts.get(k, 0)) for k in ("train", "val", "test")})

        # Purge posts whose label window reaches into the next split.
        if tc.purge_straddling_posts:
            keep = (
                ((df["split"] == "train") & (df["t_end"] < train_t))
                | ((df["split"] == "val") & (df["t_end"] < val_t))
                | (df["split"] == "test")
            )
            df = df[keep].reset_index(drop=True)
        counts = df["split"].value_counts().to_dict()
        log.info("Final splits - %s", {k: int(counts.get(k, 0)) for k in ("train", "val", "test")})

        for name in ("train", "val", "test"):
            if counts.get(name, 0) == 0:
                raise ValueError(
                    f"Split '{name}' is empty after filtering (raw={raw_counts}, final={counts}). "
                    f"Shorten the horizon / observation window, lower min_observed_events, or change split fractions."
                )
        self._check_no_leakage(df)

        n_zero = float((df["growth"] == 0).mean())
        q = df["label"].quantile([0.5, 0.9, 0.99]).tolist()
        log.info("label median/p90/p99: %.0f / %.0f / %.0f; share with zero growth: %.1f%%", *q, 100 * n_zero)

        # Labels are written separately so graph / simulation code never sees them by accident.
        labels_df = df[["post_id", "label", "growth"]].copy()
        meta_df = df.drop(columns=["label", "growth"])
        meta_df.to_parquet(self.splits_dir / "split_meta.parquet", index=False)
        labels_df.to_parquet(self.splits_dir / "labels.parquet", index=False)

        manifest = {
            "dataset": self.dataset_name,
            "config_hash": self.cfg.section_hash("time"),
            "observation_seconds": self._obs_seconds,
            "horizon_seconds": self._horizon_seconds,
            "data_end": data_end,
            "train_boundary_t0": train_t,
            "val_boundary_t0": val_t,
            "n_initial_posts": int(n_initial),
            "raw_split_counts": {k: int(v) for k, v in raw_counts.items()},
            "final_split_counts": {k: int(v) for k, v in counts.items()},
        }
        with open(self.splits_dir / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._meta = None  # force get_snapshot to reload the new split_meta
        self._ev_index = None
        log.info("SUCCESS: generated %d posts.", len(df))
        return meta_df, labels_df

    def _add_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """observed_count, label and growth per post, without a big merge."""
        ev = self.events[self.events["post_id"].isin(df["post_id"])]
        codes = pd.Index(df["post_id"]).get_indexer(ev["post_id"])  # position of each event's post in df
        rel = ev["time"].to_numpy() - df["t0"].to_numpy()[codes]    # seconds since the post's t0
        n = len(df)
        df["observed_count"] = np.bincount(codes[rel <= self._obs_seconds], minlength=n)
        df["label"] = np.bincount(codes[rel <= self._horizon_seconds], minlength=n)
        df["growth"] = df["label"] - df["observed_count"]
        return df

    @staticmethod
    def _require_nonempty(df: pd.DataFrame, when: str) -> None:
        if df.empty:
            raise ValueError(f"No posts left {when}.")

    @staticmethod
    def _check_no_leakage(df: pd.DataFrame) -> None:
        """Hard checks (not `assert`, so they survive python -O)."""
        tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
        if not (set(tr["post_id"]).isdisjoint(va["post_id"]) and set(tr["post_id"]).isdisjoint(te["post_id"])
                and set(va["post_id"]).isdisjoint(te["post_id"])):
            raise RuntimeError("CRITICAL LEAK: a post appears in more than one split")
        if tr["t_end"].max() >= va["t0"].min():
            raise RuntimeError("CRITICAL LEAK: train label window reaches val")
        if va["t_end"].max() >= te["t0"].min():
            raise RuntimeError("CRITICAL LEAK: val label window reaches test")
        if tr["t_end"].max() >= te["t0"].min():
            raise RuntimeError("CRITICAL LEAK: train label window reaches test")

    # ------------------------------------------------------------------
    # Snapshot at t_obs
    # ------------------------------------------------------------------
    def _load_meta(self) -> pd.DataFrame:
        if self._meta is None:
            m = pd.read_parquet(self.splits_dir / "split_meta.parquet")
            self._meta = m.set_index("post_id", drop=False)
        return self._meta

    def _event_index(self):
        """Events of the split posts, sorted by (post_id, time), with per-post slice offsets."""
        if self._ev_index is None:
            meta = self._load_meta()
            ev = self.events[self.events["post_id"].isin(meta["post_id"])]
            ev = ev.sort_values(["post_id", "time"], kind="stable").reset_index(drop=True)
            uniq, start, counts = np.unique(ev["post_id"].to_numpy(), return_index=True, return_counts=True)
            self._ev_index = (ev, uniq, start, start + counts)
        return self._ev_index

    def get_snapshot(self, post_id):
        """Everything knowable at t_obs for one post: (edges, events, post_meta).

        Edges and events with time > t_obs are never returned. Treat the returned
        edges as read-only (it is a view of the full edge table).
        """
        meta = self._load_meta()
        try:
            post_meta = meta.loc[post_id]
        except KeyError:
            raise KeyError(f"post_id {post_id!r} is not in split_meta (unknown or purged)") from None
        t_obs = int(post_meta["t_obs"])

        if self.edge_time_available:
            # created_at already includes BaseAdapter's lag, so it is the first moment the edge is visible.
            idx = np.searchsorted(self.edges["created_at"].to_numpy(), t_obs, side="right")
            valid_edges = self.edges.iloc[:idx]
        else:
            if not self._warned_static:
                log.warning("%s has a static graph: edges are NOT time-filtered, so the snapshot "
                            "may contain edges created after t_obs.", self.dataset_name)
                self._warned_static = True
            valid_edges = self.edges

        ev, uniq, start, end = self._event_index()
        i = np.searchsorted(uniq, post_id)
        if i < len(uniq) and uniq[i] == post_id:
            post_events = ev.iloc[start[i]:end[i]]
        else:
            post_events = ev.iloc[0:0]
        k = np.searchsorted(post_events["time"].to_numpy(), t_obs, side="right")
        valid_events = post_events.iloc[:k].copy()

        return valid_edges, valid_events, post_meta


if __name__ == "__main__":
    from src.config import load_config
    logging.basicConfig(level=logging.INFO)

    c = load_config()
    TimeSplitter(c).create_splits()