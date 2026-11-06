"""Time-aware train/val/test splitter with leak-proof snapshots, shared by all datasets.

Reads the canonical tables written by BaseAdapter (events, posts, edges) and writes
  split_meta.parquet   post_id, author, t0, t_obs, t_end, observed_count, split
  labels.parquet       post_id, label, growth
  split_manifest.json  parameters, counts, split boundaries

Definitions (per post)
----------------------
t_obs          = t0 + observation window     (what a model may see)
t_end          = t0 + horizon                (what is predicted)
observed_count = events with time <= t_obs
label          = events with time <= t_end   (total popularity at the horizon)
growth         = label - observed_count      (new events after t_obs)

Usage (no config files)
-----------------------
    splitter = TimeSplitter(".", "flickr",
                            horizon_seconds=7 * DAY,
                            observation_fraction=0.25)      # or observation_seconds=...
    splitter.create_splits()
    splitter.audit()                                         # optional leak check on samples
    snap = splitter.get_snapshot(post_id)                    # edges, events, nodes at t_obs

Leak protection (none of these can be switched off)
---------------------------------------------------
1. Labels live in their own file; split_meta and snapshots never contain them.
2. Posts whose horizon runs past the end of the data are dropped (truncated labels).
3. Chronological split on t0, and a post is purged if its label window (t0 .. t_end)
   reaches into the next split. So no train/val label is computed from a period in
   which later splits start. Hard checks (RuntimeError) verify this after every split.
4. Snapshots contain only events with time <= t_obs and edges with created_at <= t_obs.
   Timed edges already include BaseAdapter's lag, so created_at is the first moment an
   edge is visible.
5. Nodes are managed by first appearance. Every user gets a node index = its rank by
   first_seen (earliest edge, event or authored post), ties broken by user id. At time T
   the visible nodes are therefore exactly indices 0..k-1 (a prefix), and indices never
   change, so embedding tables / adjacency matrices sized to k are always consistent and
   never contain a user the model could not know yet.
6. A static graph (no edge times) cannot be time-filtered. It is refused unless you pass
   allow_static_graph=True, and then results may be optimistic.
7. audit() re-checks sampled snapshots and recomputes labels from the raw events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400
SPLITS = ("train", "val", "test")
_ALWAYS_KNOWN = np.iinfo(np.int64).min  # first_seen of users in a static graph


@dataclass(frozen=True)
class Snapshot:
    """Everything knowable at t_obs for one post. Treat all frames as read-only (views of big tables).

    post_meta  Series: post_id, author, t0, t_obs, t_end, observed_count, split (no labels)
    events     this post's events with time <= t_obs; extra column user_idx = node index
    edges      edges visible at t_obs; extra columns src_idx, dst_idx = node indices
    nodes      users visible at t_obs; the row position is the node index (0..num_nodes-1)
    """
    post_id: object
    post_meta: pd.Series
    t_obs: int
    events: pd.DataFrame
    edges: pd.DataFrame
    nodes: pd.DataFrame

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def as_tuple(self):
        """Old-style (edges, events, post_meta)."""
        return self.edges, self.events, self.post_meta


class TimeSplitter:
    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        *,
        horizon_seconds: int,
        observation_seconds: int | None = None,
        observation_fraction: float | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        min_observed_events: int = 1,
        edge_time_available: bool | None = None,
        allow_static_graph: bool = False,
    ):
        """
        horizon_seconds        prediction horizon after t0
        observation_seconds    observation window in seconds ...
        observation_fraction   ... or as a fraction of the horizon (give exactly one)
        train_frac, val_frac   chronological fractions of posts by t0; test is the rest
        min_observed_events    keep posts with at least this many events at t_obs
        edge_time_available    None = read it from the adapter's manifest.json (recommended)
        allow_static_graph     allow a graph without edge times (NOT leak-proof, see docstring)
        """
        self.dataset_name = dataset_name
        base = Path(base_path)
        self.processed_dir = base / "data" / "processed" / dataset_name
        self.splits_dir = base / "data" / "splits" / dataset_name
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.horizon_seconds = int(round(horizon_seconds))
        if (observation_seconds is None) == (observation_fraction is None):
            raise ValueError("Give exactly one of observation_seconds or observation_fraction")
        if observation_seconds is not None:
            self.observation_seconds = int(round(observation_seconds))
        else:
            self.observation_seconds = int(round(observation_fraction * self.horizon_seconds))
        if not 0 < self.observation_seconds < self.horizon_seconds:
            raise ValueError(f"Observation window ({self.observation_seconds}s) must be positive and "
                             f"shorter than the horizon ({self.horizon_seconds}s)")

        if not (0 < train_frac and 0 <= val_frac and train_frac + val_frac < 1):
            raise ValueError("Need train_frac > 0, val_frac >= 0 and train_frac + val_frac < 1 (test gets the rest)")
        self.train_frac, self.val_frac = float(train_frac), float(val_frac)
        self.min_observed_events = int(min_observed_events)
        self.allow_static_graph = bool(allow_static_graph)
        self._edge_time_override = edge_time_available

        self._meta: pd.DataFrame | None = None
        self._ev_index = None
        self._warned_static = False

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    @property
    def params(self) -> dict:
        return {
            "horizon_seconds": self.horizon_seconds,
            "observation_seconds": self.observation_seconds,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "min_observed_events": self.min_observed_events,
        }

    @cached_property
    def edge_time_available(self) -> bool:
        if self._edge_time_override is not None:
            return bool(self._edge_time_override)
        manifest = self.processed_dir / "manifest.json"
        if manifest.exists():
            return bool(json.loads(manifest.read_text()).get("edge_time_available", True))
        return True

    # ------------------------------------------------------------------
    # Lazy loading (edges and nodes are only needed for snapshots)
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
    def _raw_edges(self) -> pd.DataFrame:
        e = pd.read_parquet(self.processed_dir / "edges.parquet")
        has_time = "created_at" in e.columns
        if self.edge_time_available != has_time:
            raise ValueError(f"edge_time_available={self.edge_time_available} but edges.parquet "
                             f"{'has' if has_time else 'has no'} a created_at column")
        if has_time and not e["created_at"].is_monotonic_increasing:
            e = e.sort_values("created_at", kind="stable").reset_index(drop=True)
        return e

    @cached_property
    def nodes(self) -> pd.DataFrame:
        """All users, ordered by first appearance: columns user, first_seen. Row position = node index.

        first_seen = earliest of: an edge touching the user (created_at), an event by the user,
        a post authored by the user (t0). Each of these is information that exists from that
        moment on, so the set of nodes visible at time T is {first_seen <= T}, a prefix.
        """
        e, ev, p = self._raw_edges, self.events, self.posts
        parts = [
            pd.DataFrame({"user": ev["user"].to_numpy(), "first_seen": ev["time"].to_numpy()}),
            pd.DataFrame({"user": p["author"].to_numpy(), "first_seen": p["t0"].to_numpy()}),
        ]
        for col in ("source", "target"):
            seen = e["created_at"].to_numpy() if self.edge_time_available else _ALWAYS_KNOWN
            parts.append(pd.DataFrame({"user": e[col].to_numpy(), "first_seen": seen}))
        allp = pd.concat(parts, ignore_index=True)
        allp = allp.sort_values(["first_seen", "user"], kind="stable")
        nodes = allp.drop_duplicates("user", keep="first").reset_index(drop=True)
        log.info("Node table: %d users", len(nodes))
        return nodes

    @cached_property
    def _node_index(self) -> pd.Index:
        return pd.Index(self.nodes["user"])

    @cached_property
    def _node_first_seen(self) -> np.ndarray:
        return self.nodes["first_seen"].to_numpy()

    @cached_property
    def edges(self) -> pd.DataFrame:
        """Edge table sorted by created_at, with node indices src_idx / dst_idx added."""
        raw = self._raw_edges
        idx = self._node_index
        out = raw.assign(
            src_idx=idx.get_indexer(raw["source"]).astype(np.int64),
            dst_idx=idx.get_indexer(raw["target"]).astype(np.int64),
        )
        self.__dict__.pop("_raw_edges", None)  # free the duplicate
        return out

    @cached_property
    def _edge_times(self) -> np.ndarray:
        return self.edges["created_at"].to_numpy()

    # ------------------------------------------------------------------
    # Building the splits
    # ------------------------------------------------------------------
    def create_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        obs, hor = self.observation_seconds, self.horizon_seconds
        log.info("--- Creating time-aware splits for %s ---", self.dataset_name)
        log.info("Observation window: %d s, horizon: %d s", obs, hor)

        df = self.posts.copy()
        n_initial = len(df)
        log.info("Initial posts: %d", n_initial)
        df["t_obs"] = df["t0"] + obs
        df["t_end"] = df["t0"] + hor

        # Posts whose horizon runs past the end of the data would get truncated labels.
        data_end = int(self.events["time"].max())
        df = df[df["t_end"] <= data_end].reset_index(drop=True)
        log.info("Posts whose horizon ends before the data end: %d of %d", len(df), n_initial)
        self._require_nonempty(df, "after dropping posts whose horizon exceeds the data end "
                                   "(the horizon may be too long for this dataset)")

        df = self._add_counts(df)
        df = df[df["observed_count"] >= self.min_observed_events].reset_index(drop=True)
        log.info("Posts with >= %d events at t_obs: %d", self.min_observed_events, len(df))
        self._require_nonempty(df, f"after requiring >= {self.min_observed_events} observed events")

        # Chronological split on t0.
        train_t = float(df["t0"].quantile(self.train_frac))
        val_t = float(df["t0"].quantile(min(self.train_frac + self.val_frac, 1.0)))
        df["split"] = np.where(df["t0"] < train_t, "train", np.where(df["t0"] < val_t, "val", "test"))
        raw_counts = df["split"].value_counts().to_dict()
        log.info("Raw splits - %s", {k: int(raw_counts.get(k, 0)) for k in SPLITS})

        # Purge posts whose label window reaches into the next split (always on).
        keep = (
            ((df["split"] == "train") & (df["t_end"] < train_t))
            | ((df["split"] == "val") & (df["t_end"] < val_t))
            | (df["split"] == "test")
        )
        df = df[keep].reset_index(drop=True)
        counts = df["split"].value_counts().to_dict()
        log.info("Final splits - %s", {k: int(counts.get(k, 0)) for k in SPLITS})

        for name in SPLITS:
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

        per_split = {
            s: {
                "n_posts": int(counts[s]),
                "t0_min": int(df.loc[df["split"] == s, "t0"].min()),
                "t0_max": int(df.loc[df["split"] == s, "t0"].max()),
                "t_end_max": int(df.loc[df["split"] == s, "t_end"].max()),
            }
            for s in SPLITS
        }
        manifest = {
            "dataset": self.dataset_name,
            "params": self.params,
            "params_hash": hashlib.sha1(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:12],
            "observation_seconds": obs,
            "horizon_seconds": hor,
            "data_end": data_end,
            "train_boundary_t0": train_t,
            "val_boundary_t0": val_t,
            "n_initial_posts": int(n_initial),
            "raw_split_counts": {k: int(v) for k, v in raw_counts.items()},
            "final_split_counts": {k: int(v) for k, v in counts.items()},
            "splits": per_split,
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
        df["observed_count"] = np.bincount(codes[rel <= self.observation_seconds], minlength=n)
        df["label"] = np.bincount(codes[rel <= self.horizon_seconds], minlength=n)
        df["growth"] = df["label"] - df["observed_count"]
        return df

    @staticmethod
    def _require_nonempty(df: pd.DataFrame, when: str) -> None:
        if df.empty:
            raise ValueError(f"No posts left {when}.")

    @staticmethod
    def _check_no_leakage(df: pd.DataFrame) -> None:
        """Hard checks (not `assert`, so they survive python -O)."""
        parts = {s: df[df["split"] == s] for s in SPLITS}
        ids = {s: set(p["post_id"]) for s, p in parts.items()}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                if not ids[a].isdisjoint(ids[b]):
                    raise RuntimeError(f"CRITICAL LEAK: a post appears in both {a} and {b}")
                if parts[a]["t_end"].max() >= parts[b]["t0"].min():
                    raise RuntimeError(f"CRITICAL LEAK: {a} label window reaches {b}")
        if not ((df["t0"] < df["t_obs"]) & (df["t_obs"] < df["t_end"])).all():
            raise RuntimeError("CRITICAL: expected t0 < t_obs < t_end for every post")
        if (df["growth"] < 0).any():
            raise RuntimeError("CRITICAL: negative growth (label < observed_count)")

    # ------------------------------------------------------------------
    # Reading the written splits (metadata and labels are separate on purpose)
    # ------------------------------------------------------------------
    def _load_meta(self) -> pd.DataFrame:
        if self._meta is None:
            m = pd.read_parquet(self.splits_dir / "split_meta.parquet")
            if {"label", "growth"} & set(m.columns):
                raise RuntimeError("split_meta.parquet contains label columns; regenerate the splits")
            self._meta = m.set_index("post_id", drop=False)
        return self._meta

    def load_meta(self, split: str | None = None) -> pd.DataFrame:
        """Post metadata (no labels), optionally for one split."""
        m = self._load_meta()
        if split is None:
            return m.reset_index(drop=True)
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return m[m["split"] == split].reset_index(drop=True)

    def load_labels(self, split: str | None = None) -> pd.DataFrame:
        """post_id, label, growth. Call this only where the labels are really needed."""
        lab = pd.read_parquet(self.splits_dir / "labels.parquet")
        if split is None:
            return lab
        return lab[lab["post_id"].isin(self.load_meta(split)["post_id"])].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Snapshot at t_obs
    # ------------------------------------------------------------------
    def _event_index(self):
        """Events of the split posts, sorted by (post_id, time), with per-post slice offsets."""
        if self._ev_index is None:
            meta = self._load_meta()
            ev = self.events[self.events["post_id"].isin(meta["post_id"])]
            ev = ev.sort_values(["post_id", "time"], kind="stable").reset_index(drop=True)
            uniq, start, counts = np.unique(ev["post_id"].to_numpy(), return_index=True, return_counts=True)
            self._ev_index = (ev, uniq, start, start + counts)
        return self._ev_index

    def _post_events(self, post_id) -> pd.DataFrame:
        """ALL events of one post (including after t_obs). Internal: never returned to models."""
        ev, uniq, start, end = self._event_index()
        i = np.searchsorted(uniq, post_id)
        if i < len(uniq) and uniq[i] == post_id:
            return ev.iloc[start[i]:end[i]]
        return ev.iloc[0:0]

    def get_snapshot(self, post_id) -> Snapshot:
        """Everything knowable at t_obs for one post (events, edges, nodes, post_meta).

        Events with time > t_obs and edges with created_at > t_obs are never returned,
        and nodes are the users first seen at or before t_obs. See Snapshot for details.
        """
        meta = self._load_meta()
        try:
            post_meta = meta.loc[post_id]
        except KeyError:
            raise KeyError(f"post_id {post_id!r} is not in split_meta (unknown or purged)") from None
        t_obs = int(post_meta["t_obs"])

        if self.edge_time_available:
            n_edges = int(np.searchsorted(self._edge_times, t_obs, side="right"))
            valid_edges = self.edges.iloc[:n_edges]
        else:
            if not self.allow_static_graph:
                raise ValueError(
                    f"{self.dataset_name} has a static graph (no edge times), so a snapshot cannot be "
                    f"limited to what existed at t_obs and would leak future edges. Pass "
                    f"allow_static_graph=True to accept this explicitly."
                )
            if not self._warned_static:
                log.warning("%s has a static graph: edges are NOT time-filtered, results may be optimistic.",
                            self.dataset_name)
                self._warned_static = True
            valid_edges = self.edges

        n_nodes = int(np.searchsorted(self._node_first_seen, t_obs, side="right"))
        valid_nodes = self.nodes.iloc[:n_nodes]

        post_events = self._post_events(post_id)
        k = int(np.searchsorted(post_events["time"].to_numpy(), t_obs, side="right"))
        valid_events = post_events.iloc[:k].copy()
        valid_events["user_idx"] = self._node_index.get_indexer(valid_events["user"]).astype(np.int64)

        return Snapshot(post_id, post_meta, t_obs, valid_events, valid_edges, valid_nodes)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def audit(self, n_per_split: int = 100, seed: int = 0) -> dict:
        """Re-check sampled snapshots and recompute labels from raw events. Raises on any violation."""
        meta = self._load_meta()
        labels = self.load_labels().set_index("post_id")
        rng = np.random.default_rng(seed)
        problems: list[str] = []
        checked = 0

        for s in SPLITS:
            ids = meta.index[meta["split"] == s].to_numpy()
            for pid in rng.choice(ids, size=min(n_per_split, len(ids)), replace=False):
                snap = self.get_snapshot(pid)
                pm, t_obs, n = snap.post_meta, snap.t_obs, snap.num_nodes
                tag = f"[{s} post {pid}]"
                ev, ed = snap.events, snap.edges

                if len(ev) and (ev["time"].max() > t_obs or (ev["post_id"] != pid).any()):
                    problems.append(f"{tag} events after t_obs or from another post")
                if len(ev) != int(pm["observed_count"]):
                    problems.append(f"{tag} snapshot has {len(ev)} events, split_meta says {int(pm['observed_count'])}")
                if len(ev) and (ev["user_idx"].min() < 0 or ev["user_idx"].max() >= n):
                    problems.append(f"{tag} event user outside the visible nodes")
                if self.edge_time_available:
                    if len(ed) and ed["created_at"].max() > t_obs:
                        problems.append(f"{tag} edge created after t_obs")
                    if len(snap.nodes) and snap.nodes["first_seen"].max() > t_obs:
                        problems.append(f"{tag} node first seen after t_obs")
                if len(ed) and (min(ed["src_idx"].min(), ed["dst_idx"].min()) < 0
                                or max(ed["src_idx"].max(), ed["dst_idx"].max()) >= n):
                    problems.append(f"{tag} edge endpoint outside the visible nodes")
                a_idx = self._node_index.get_indexer([pm["author"]])[0]
                if not 0 <= a_idx < n:
                    problems.append(f"{tag} author is not a visible node")

                times = self._post_events(pid)["time"].to_numpy()
                label = int((times <= int(pm["t_end"])).sum())
                if label != int(labels.loc[pid, "label"]):
                    problems.append(f"{tag} stored label {int(labels.loc[pid, 'label'])} != recomputed {label}")
                checked += 1

        if problems:
            raise RuntimeError(f"Audit found {len(problems)} problem(s):\n - " + "\n - ".join(problems[:10]))
        log.info("Audit passed on %d sampled snapshots", checked)
        return {"checked": checked, "problems": 0}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create time-aware splits for a processed dataset.")
    p.add_argument("--dataset", default="flickr")
    p.add_argument("--base-path", default=".")
    p.add_argument("--horizon-hours", type=float, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--observation-hours", type=float)
    g.add_argument("--observation-fraction", type=float, help="fraction of the horizon")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--min-observed-events", type=int, default=1)
    p.add_argument("--allow-static-graph", action="store_true")
    p.add_argument("--audit", action="store_true", help="re-check sampled snapshots (loads edges, slower)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    a = _parse_args()
    splitter = TimeSplitter(
        a.base_path, a.dataset,
        horizon_seconds=a.horizon_hours * HOUR,
        observation_seconds=None if a.observation_hours is None else a.observation_hours * HOUR,
        observation_fraction=a.observation_fraction,
        train_frac=a.train_frac,
        val_frac=a.val_frac,
        min_observed_events=a.min_observed_events,
        allow_static_graph=a.allow_static_graph,
    )
    splitter.create_splits()
    if a.audit:
        splitter.audit()