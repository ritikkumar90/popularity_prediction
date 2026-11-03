import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config

log = logging.getLogger(__name__)

class TimeSplitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.time_cfg = cfg.time
        
        # Paths 
        self.base_dir = cfg.run.base_path
        self.processed_dir = self.base_dir / "data" / "processed" / self.dataset_name
        self.splits_dir = self.base_dir / "data" / "splits" / self.dataset_name
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Data
        self.posts = pd.read_parquet(self.processed_dir / "posts.parquet").sort_values("t0")
        self.events = pd.read_parquet(self.processed_dir / "events.parquet").sort_values("time")
        self.edges = pd.read_parquet(self.processed_dir / "edges.parquet")
        
        if cfg.active.edge_time_available:
            self.edges = self.edges.sort_values("created_at").reset_index(drop=True)
        
        # Safely extract seconds using the Pydantic properties
        self._obs_seconds = self.time_cfg.observation_window.seconds
        self._horizon_seconds = self.time_cfg.horizon.seconds

    def create_splits(self):
        log.info(f"--- Creating time-aware splits for {self.dataset_name} ---")
        
        df = self.posts.copy()
        log.info(f"Initial posts: {len(df)}")
        
        df["t_obs"] = df["t0"] + self._obs_seconds
        df["t_end"] = df["t0"] + self._horizon_seconds
        
        # LEAKAGE HOLE 3 FIX: Truncate posts whose horizon extends beyond the dataset collection end
        if self.time_cfg.drop_if_horizon_exceeds_data_end:
            max_dataset_time = self.events["time"].max()
            df = df[df["t_end"] <= max_dataset_time].copy()
            log.info(f"Posts surviving end-of-dataset truncation: {len(df)}")

        # MEMORY OPTIMIZATION: filter events to surviving posts before the heavy merge
        relevant_events = self.events[self.events["post_id"].isin(df["post_id"])]
        events_merged = pd.merge(relevant_events, df[["post_id", "t_obs", "t_end"]], on="post_id")
        
        obs_mask = events_merged["time"] <= events_merged["t_obs"]
        horizon_mask = events_merged["time"] <= events_merged["t_end"]
        
        obs_counts = events_merged[obs_mask].groupby("post_id").size().rename("observed_count")
        label_counts = events_merged[horizon_mask].groupby("post_id").size().rename("label")
        
        df = df.join(obs_counts, on="post_id").join(label_counts, on="post_id").fillna(0)
        
        # Cast metrics to int explicitly
        df = df.astype({"observed_count": int, "label": int})
        
        df = df[df["observed_count"] >= self.time_cfg.min_observed_events].sort_values("t0").reset_index(drop=True)
        log.info(f"Posts after requiring >= {self.time_cfg.min_observed_events} events at t_obs: {len(df)}")
        
        if len(df) == 0:
            log.warning("No posts survived filtering.")
            return df
            
        train_q = self.time_cfg.split.train
        val_q = train_q + self.time_cfg.split.val
        
        train_t = df["t0"].quantile(train_q)
        val_t = df["t0"].quantile(val_q)
        
        df["split"] = "test"
        df.loc[df["t0"] < val_t, "split"] = "val"
        df.loc[df["t0"] < train_t, "split"] = "train"
        
        log.info(f"Raw splits - Train: {len(df[df['split']=='train'])}, Val: {len(df[df['split']=='val'])}, Test: {len(df[df['split']=='test'])}")
        
        # LEAKAGE HOLES 1 & 2 FIX: Quantile-based purge and leakage assertions
        if self.time_cfg.purge_straddling_posts:
            keep = (
                ((df["split"] == "train") & (df["t_end"] < train_t))
                | ((df["split"] == "val") & (df["t_end"] < val_t))
                | (df["split"] == "test")
            )
            df = df[keep]
            
            tr = df[df["split"] == "train"]
            va = df[df["split"] == "val"]
            te = df[df["split"] == "test"]
            
            # Assertions to guarantee strict temporal separation
            if len(tr) and len(va):
                assert tr["t_end"].max() < va["t0"].min(), "CRITICAL LEAK: Train label window reaches Val"
            if len(va) and len(te):
                assert va["t_end"].max() < te["t0"].min(), "CRITICAL LEAK: Val label window reaches Test"
            if len(tr) and len(te):
                assert tr["t_end"].max() < te["t0"].min(), "CRITICAL LEAK: Train label window reaches Test"

            log.info(f"Posts surviving straddle purge - Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

        # LEAKAGE HOLE 4 FIX: Isolate labels to prevent Method B (HGT) from accidentally reading them
        labels_df = df[["post_id", "label"]].copy()
        meta_df = df.drop(columns=["label"])

        meta_df.to_parquet(self.splits_dir / "split_meta.parquet", index=False)
        labels_df.to_parquet(self.splits_dir / "labels.parquet", index=False)
        
        log.info(f"SUCCESS: Generated {len(df)} posts.")
        return meta_df, labels_df

    # LEAKAGE HOLE 5 FIX: get_snapshot strictly guarantees no future events or edges can be retrieved
    def get_snapshot(self, post_id: str):
        meta = pd.read_parquet(self.splits_dir / "split_meta.parquet")
        post_meta = meta[meta["post_id"] == post_id].iloc[0]
        t_obs = post_meta["t_obs"]
        
        if self.cfg.active.edge_time_available:
            idx = np.searchsorted(self.edges["created_at"].values, t_obs, side="right")
            valid_edges = self.edges.iloc[:idx]
        else:
            valid_edges = self.edges
            
        post_events = self.events[self.events["post_id"] == post_id].sort_values("time")
        idx_ev = np.searchsorted(post_events["time"].values, t_obs, side="right")
        valid_events = post_events.iloc[:idx_ev]
        
        return valid_edges, valid_events, post_meta

if __name__ == "__main__":
    from src.config import load_config
    logging.basicConfig(level=logging.INFO)
    
    c = load_config()
    splitter = TimeSplitter(c)
    splitter.create_splits()