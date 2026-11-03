import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from src.config import Config, load_config

log = logging.getLogger(__name__)

class DiffusionFitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.base_dir = cfg.run.base_path
        
        self.processed_dir = self.base_dir / "data" / "processed" / self.dataset_name
        self.splits_dir = self.base_dir / "data" / "splits" / self.dataset_name
        self.output_dir = self.base_dir / cfg.run.output_dir / self.dataset_name / "fit"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tables and merge isolated labels safely
        meta = pd.read_parquet(self.splits_dir / "split_meta.parquet")
        labels = pd.read_parquet(self.splits_dir / "labels.parquet")
        self.meta = pd.merge(meta, labels, on="post_id")
        
        self.events = pd.read_parquet(self.processed_dir / "events.parquet")
        self.edges = pd.read_parquet(self.processed_dir / "edges.parquet")

    def fit(self):
        log.info(f"--- Fitting diffusion parameters for {self.dataset_name} (Train Split Only) ---")
        
        train_meta = self.meta[self.meta["split"] == "train"]
        if len(train_meta) == 0:
            raise ValueError("Train split is empty! Adjust time windows or min_observed_events in config.yaml.")

        # Strict leak-prevention: cap train events at each post's horizon t_end
        train_events = pd.merge(
            self.events, 
            train_meta[["post_id", "t_end"]], 
            on="post_id"
        )
        train_events = train_events[train_events["time"] <= train_events["t_end"]].sort_values(["post_id", "time"])

        # Precompute static/dynamic out-degrees and incoming adjacency lists for active train users
        train_users = set(train_events["user"].unique())
        out_degrees = self.edges.groupby("source").size().to_dict()
        
        relevant_edges = self.edges[
            self.edges["target"].isin(train_users) & 
            self.edges["source"].isin(train_users)
        ]
        
        has_edge_time = self.cfg.active.edge_time_available
        target_to_sources = defaultdict(list)
        
        if has_edge_time:
            for row in relevant_edges.itertuples(index=False):
                target_to_sources[row.target].append((row.source, row.created_at))
        else:
            for row in relevant_edges.itertuples(index=False):
                target_to_sources[row.target].append((row.source, 0))

        user_attempts = defaultdict(int)
        user_successes = defaultdict(int)
        edge_successes = defaultdict(int)
        delays = []

        # Fast linear scan over train cascades
        for _, group in train_events.groupby("post_id", sort=False):
            users = group["user"].to_numpy()
            times = group["time"].to_numpy()
            active_time = {}
            
            for i in range(len(users)):
                v = int(users[i])
                tv = float(times[i])
                if v in active_time:
                    continue
                
                for u, edge_t in target_to_sources.get(v, ()):
                    if u in active_time:
                        tu = active_time[u]
                        if tu < tv and edge_t <= tv:
                            edge_successes[(u, v)] += 1
                            user_successes[u] += 1
                            delays.append(tv - tu)
                            
                active_time[v] = tv
                user_attempts[v] += 1

        # Compute hierarchical backoff probabilities with Laplace smoothing
        alpha = self.cfg.fit.smoothing.alpha
        beta = self.cfg.fit.smoothing.beta

        total_attempts = sum(count * out_degrees.get(u, 0) for u, count in user_attempts.items())
        total_successes = len(delays)
        global_p = float((total_successes + alpha) / (total_attempts + alpha + beta)) if total_attempts > 0 else 0.01

        user_p = {}
        for u, att_cascades in user_attempts.items():
            deg = out_degrees.get(u, 0)
            if deg > 0:
                u_att = att_cascades * deg
                u_succ = user_successes.get(u, 0)
                if u_succ > 0:
                    user_p[int(u)] = float((u_succ + alpha) / (u_att + alpha + beta))
        edge_p = {}
        for (u, v), succ in edge_successes.items():
            att = user_attempts[u]
            edge_p[f"{u}_{v}"] = float((succ + alpha) / (att + alpha + beta))

        # Fit continuous-time delay rate (lambda = 1 / mean_delay)
        delay_rate = 1.0 / float(np.mean(delays)) if len(delays) > 0 and np.mean(delays) > 0 else 1.0 / 3600.0

        # Fit validation scale factor
        scale_factor = 1.0
        if self.cfg.fit.scale_factor.enabled:
            target_split = self.cfg.fit.scale_factor.fit_on
            val_meta = self.meta[self.meta["split"] == target_split]
            if len(val_meta) > 0:
                obs_mean = val_meta["observed_count"].mean()
                label_mean = val_meta["label"].mean()
                if obs_mean > 0:
                    scale_factor = float(label_mean / obs_mean)

        params = {
            "global_p": global_p,
            "delay_rate": delay_rate,
            "scale_factor": scale_factor,
            "p_estimator": self.cfg.fit.p_estimator,
            "user_p": user_p,
            "edge_p": edge_p
        }

        out_file = self.output_dir / "fitted_params.json"
        with open(out_file, "w") as f:
            json.dump(params, f)

        log.info(
            f"Fit complete -> Global P: {global_p:.6e} | Delay Rate: {delay_rate:.6e} "
            f"| Scale Factor: {scale_factor:.3f} | Saved {len(edge_p)} edge probs"
        )
        return params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = load_config()
    DiffusionFitter(c).fit()