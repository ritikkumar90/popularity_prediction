import json
import heapq
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import Config, load_config
from src.data.splitter import TimeSplitter
from src.im.selectors import SeedSelectors

log = logging.getLogger(__name__)

class FastPostCascadeSimulator:
    """Compiled per-post graph representation for ultra-fast Monte Carlo rollouts."""
    def __init__(
        self, 
        adj: dict[int, tuple[np.ndarray, np.ndarray]], 
        initial_active: dict[int, float], 
        t_obs: float, 
        t_end: float, 
        delay_rate: float, 
        scale_factor: float, 
        max_size: int, 
        base_seed: int
    ):
        self.adj = adj
        self.initial_active = initial_active
        self.t_obs = t_obs
        self.t_end = t_end
        self.mean_delay = 1.0 / delay_rate
        self.scale_factor = scale_factor
        self.max_size = max_size
        self.base_seed = base_seed

    def evaluate_spread(self, extra_seeds: list[int], n_runs: int) -> float:
        total_size = 0
        
        for r in range(n_runs):
            rng = np.random.default_rng(self.base_seed + r)
            active = self.initial_active.copy()
            pq = []

            # Only nodes with outgoing edges can infect others
            for u, t in active.items():
                if u in self.adj:
                    heapq.heappush(pq, (t, u))

            for s in extra_seeds:
                if s not in active:
                    active[s] = self.t_obs
                    if s in self.adj:
                        heapq.heappush(pq, (self.t_obs, s))

            count = len(active)

            while pq and count < self.max_size:
                tu, u = heapq.heappop(pq)
                if tu > self.t_end:
                    break
                if tu > active[u]:
                    continue

                targets, probs = self.adj[u]
                mask = rng.random(len(targets)) < probs
                if not np.any(mask):
                    continue

                succ_targets = targets[mask]
                delays = rng.exponential(self.mean_delay, size=len(succ_targets))
                arr_times = tu + delays

                for v, tv in zip(succ_targets, arr_times):
                    v_int = int(v)
                    if tv <= self.t_end:
                        prev_t = active.get(v_int)
                        if prev_t is None or tv < prev_t:
                            if prev_t is None:
                                count += 1
                            active[v_int] = float(tv)
                            if v_int in self.adj:
                                heapq.heappush(pq, (float(tv), v_int))

            total_size += count

        return (total_size / n_runs) * self.scale_factor


class DiffusionRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.base_dir = cfg.run.base_path
        
        fit_file = self.base_dir / cfg.run.output_dir / self.dataset_name / "fit" / "fitted_params.json"
        if not fit_file.exists():
            raise FileNotFoundError(f"Missing {fit_file}. Run `python -m src.diffusion.fit` first.")
            
        with open(fit_file, "r") as f:
            self.params = json.load(f)

        self.global_p = float(self.params["global_p"])
        self.delay_rate = float(self.params["delay_rate"])
        self.scale_factor = float(self.params["scale_factor"]) if cfg.fit.scale_factor.enabled else 1.0
        self.user_p = {int(k): float(v) for k, v in self.params.get("user_p", {}).items()}
        self.edge_p = self.params.get("edge_p", {})
        self.estimator = self.params.get("p_estimator", "backoff")

        self.splitter = TimeSplitter(cfg)
        self.results_dir = self.base_dir / cfg.run.output_dir / self.dataset_name / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_prob(self, u: int, v: int) -> float:
        if self.estimator == "global":
            return self.global_p
        if self.estimator in ("edge", "backoff"):
            ep = self.edge_p.get(f"{u}_{v}")
            if ep is not None:
                return float(ep)
        if self.estimator in ("user", "backoff"):
            up = self.user_p.get(u)
            if up is not None:
                return float(up)
        return self.global_p

    def _prepare_post_subgraph(self, post_id: int | str):
        """Extracts frozen t_obs snapshot, selects top-N candidates, and compiles local NumPy adjacency."""
        edges_df, events_df, post_meta = self.splitter.get_snapshot(post_id)
        t0 = float(post_meta["t0"])
        t_obs = float(post_meta["t_obs"])
        t_end = float(post_meta["t_end"])

        initial_active = {}
        if self.cfg.simulate.start_active_set == "observed":
            for u, t in zip(events_df["user"].to_numpy(), events_df["time"].to_numpy()):
                u_int = int(u)
                if u_int not in initial_active:
                    initial_active[u_int] = float(t)
        else:
            initial_active[int(post_meta["author"])] = t0

        active_set = set(initial_active.keys())

        # Extract local multi-hop neighbourhood around active set so we don't scan millions of irrelevant edges
        frontier = set(active_set)
        visited_sources = set()
        sub_edges_list = []

        max_hops = max(self.cfg.im.candidate_hops, 2) + 1
        for _ in range(max_hops):
            new_sources = frontier - visited_sources
            if not new_sources:
                break
            hop_df = edges_df[edges_df["source"].isin(new_sources)]
            if hop_df.empty:
                break
            sub_edges_list.append(hop_df)
            visited_sources.update(new_sources)
            frontier = set(hop_df["target"].unique())

        if sub_edges_list:
            sub_edges = pd.concat(sub_edges_list, ignore_index=True).drop_duplicates(subset=["source", "target"])
        else:
            sub_edges = edges_df.iloc[0:0]

        # Candidate Pool: inactive nodes in the neighbourhood ranked by out-degree (influence potential)
        if not sub_edges.empty:
            out_deg = sub_edges.groupby("source").size()
            inactive_deg = out_deg[~out_deg.index.isin(active_set)]
            candidates = inactive_deg.nlargest(self.cfg.im.candidate_top_n).index.astype(int).tolist()
        else:
            candidates = []

        # Compile fast NumPy adjacency lists
        raw_adj = defaultdict(list)
        for u, v in zip(sub_edges["source"].to_numpy(), sub_edges["target"].to_numpy()):
            u_int, v_int = int(u), int(v)
            raw_adj[u_int].append((v_int, self._get_prob(u_int, v_int)))

        compiled_adj = {}
        for u, pairs in raw_adj.items():
            targets = np.array([p[0] for p in pairs], dtype=np.int64)
            probs = np.array([p[1] for p in pairs], dtype=np.float64)
            compiled_adj[u] = (targets, probs)

        sim = FastPostCascadeSimulator(
            adj=compiled_adj,
            initial_active=initial_active,
            t_obs=t_obs,
            t_end=t_end,
            delay_rate=self.delay_rate,
            scale_factor=self.scale_factor,
            max_size=self.cfg.simulate.max_cascade_size,
            base_seed=self.cfg.run.seed
        )
        return sim, candidates

    def run_all(self):
        meta = pd.read_parquet(self.splitter.splits_dir / "split_meta.parquet")
        labels = pd.read_parquet(self.splitter.splits_dir / "labels.parquet")
        
        test_posts = meta[meta["split"] == "test"]["post_id"].tolist()
        if not test_posts:
            log.warning("No posts in 'test' split; falling back to all available split posts.")
            test_posts = meta["post_id"].tolist()

        if self.cfg.sampling.n_posts:
            test_posts = test_posts[:self.cfg.sampling.n_posts]

        k_list = sorted(self.cfg.im.k_list)
        k_max = max(k_list)
        sel_runs = self.cfg.im.selection_runs

        rows = []
        log.info(f"Running Greedy & CELF (k={k_list}) across {len(test_posts)} posts...")

        for idx, post_id in enumerate(test_posts, 1):
            # Fetch Ground Truth
            true_spread = int(labels.loc[labels["post_id"] == post_id, "label"].iloc[0])
            
            sim, candidates = self._prepare_post_subgraph(post_id)
            spread_fn = lambda seeds: sim.evaluate_spread(seeds, n_runs=sel_runs)
            
            base_spread = spread_fn([])

            # Run Greedy and CELF for k=1, 2, 3
            _, greedy_spreads = SeedSelectors.greedy(candidates, k_max, spread_fn, base_spread)
            _, celf_spreads = SeedSelectors.celf(candidates, k_max, spread_fn, base_spread)

            g_vals = [greedy_spreads[k] for k in k_list]
            c_vals = [celf_spreads[k] for k in k_list]
            
            greedy_avg = float(np.mean(g_vals))
            celf_avg = float(np.mean(c_vals))
            all_seeds_avg = float(np.mean(g_vals + c_vals))

            row = {
                "post_id": post_id,
                "ground_truth": true_spread,
                "base_spread": round(base_spread, 2),
                "greedy_k1": round(greedy_spreads.get(1, base_spread), 2),
                "greedy_k2": round(greedy_spreads.get(2, base_spread), 2),
                "greedy_k3": round(greedy_spreads.get(3, base_spread), 2),
                "celf_k1": round(celf_spreads.get(1, base_spread), 2),
                "celf_k2": round(celf_spreads.get(2, base_spread), 2),
                "celf_k3": round(celf_spreads.get(3, base_spread), 2),
                "greedy_avg": round(greedy_avg, 2),
                "celf_avg": round(celf_avg, 2),
                "all_seeds_avg": round(all_seeds_avg, 2),
            }
            rows.append(row)
            log.info(
                f"[{idx}/{len(test_posts)}] Post {post_id} | True: {true_spread} | Base: {row['base_spread']} | "
                f"Greedy(k1,2,3)=({row['greedy_k1']}, {row['greedy_k2']}, {row['greedy_k3']})"
            )

        results_df = pd.DataFrame(rows)
        out_csv = self.results_dir / "spread_comparison.csv"
        results_df.to_csv(out_csv, index=False)
        
        print("\n=== Final Spread Comparison ===")
        print(results_df.to_string(index=False))
        log.info(f"Saved results table to {out_csv}")
        return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = load_config()
    DiffusionRunner(c).run_all()