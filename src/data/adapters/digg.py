"""Digg adapter (Lerman et al., Digg 2009 data).

Raw files (header-less, comma separated), all inside `raw_dir`:
  votes   digg_votes1.csv (+ digg_votes2.csv, optional) : vote_time, user, post
  edges   digg_friends.csv                              : mutual, created_at, user, friend

Both files are header-less. (In the schema printout the first data row shows up as the
header, which is why the loader passes header=None.)

Time
----
Vote times and friendship times are both plain Unix seconds, so time_format is
"unix_seconds" and no convert_*_time() override is needed. Numeric edge times carry no
visibility lag (edge_time_lag_seconds == 0): an edge is visible from its own timestamp.
BaseAdapter does the rest (cleaning, de-duplication, sorting).

Edge direction
--------------
In digg_friends.csv, `user` follows `friend` and sees the friend's votes, so information
flows friend -> user:  source = friend, target = user.
Set reverse_edges=True to flip this.

`mutual` is 1 when the friendship is reciprocal. With expand_mutual=True (default) a
reverse edge (user -> friend, same timestamp) is added for every mutual row. If the file
already lists both directions this is harmless: BaseAdapter keeps the earliest copy of
each (source, target) pair.

Authors (author_rule)
---------------------
The Digg data has no story-owner file. The first vote on a story is the submitter's own
digg, so the author is the first voter: author_rule = "first_event" (the only option).
Ties on the exact same second are broken by user id, as in the Flickr adapter.

Dataset size (subset_mode / subset_value)
-----------------------------------------
Same modes as the Flickr adapter:

  "full"           everything (subset_value ignored)
  "post_fraction"  random fraction of posts, 0 < value <= 1 (e.g. 0.05). Each kept post
                   keeps its COMPLETE vote history, so labels stay correct. Recommended.
  "top_posts"      the `value` posts with the most votes. WARNING: selecting by total
                   popularity biases the label distribution (it uses future information
                   for selection). Fine for debugging, not for reported results.
  "first_rows"     first `value` rows of the votes file(s). The Digg votes file is ordered
                   by story, not by time, so only the last story at the cut is truncated.
                   Still not recommended for reported results.

With restrict_edges=True (default for any subset), edges are reduced to the subgraph
induced by the voters in the subset. That makes graph building much cheaper. Set it to
False to keep the full graph.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.adapters.base import BaseAdapter

log = logging.getLogger(__name__)

# Column order of the raw files.
VOTE_COLUMNS = ["time", "user", "post"]
FRIEND_COLUMNS = ["mutual", "created_at", "user", "friend"]

# Default raw file names inside raw_dir (override via constructor arguments).
DEFAULT_VOTES_FILES = ("digg_votes1.csv",)   # add "digg_votes2.csv" if you have the second part
DEFAULT_EDGES_FILE = "digg_friends.csv"

SUBSET_MODES = ("full", "post_fraction", "top_posts", "first_rows")
AUTHOR_RULES = ("first_event",)


class DiggAdapter(BaseAdapter):
    def __init__(
        self,
        base_path: str | Path,
        raw_dir: str | Path,
        *,
        votes_files: str | tuple[str, ...] | list[str] = DEFAULT_VOTES_FILES,
        edges_file: str = DEFAULT_EDGES_FILE,
        author_rule: str = "first_event",
        subset_mode: str = "full",
        subset_value: float | int | None = None,
        seed: int = 42,
        restrict_edges: bool = True,
        reverse_edges: bool = False,
        expand_mutual: bool = True,
    ):
        """
        base_path       project root; output goes to <base_path>/data/processed/digg
        raw_dir         folder containing the raw files
        votes_files     one file name or several (e.g. digg_votes1.csv, digg_votes2.csv)
        author_rule     only "first_event" (no owner file exists for Digg)
        subset_mode     "full" | "post_fraction" | "top_posts" | "first_rows" (see module docstring)
        subset_value    fraction (post_fraction) or count (top_posts, first_rows)
        seed            random seed for post_fraction (same seed -> same subset)
        restrict_edges  with a subset, keep only edges between users present in the subset
        reverse_edges   swap source/target (default direction is friend -> user)
        expand_mutual   add the reverse edge for rows with mutual == 1
        """
        if author_rule not in AUTHOR_RULES:
            raise ValueError(f"author_rule must be one of {AUTHOR_RULES} for Digg, got {author_rule!r}")
        if subset_mode not in SUBSET_MODES:
            raise ValueError(f"subset_mode must be one of {SUBSET_MODES}, got {subset_mode!r}")
        self._check_subset_value(subset_mode, subset_value)

        if isinstance(votes_files, str):
            votes_files = (votes_files,)
        if not votes_files:
            raise ValueError("votes_files must contain at least one file name")

        raw_dir = Path(raw_dir)
        vote_keys = [f"votes_{i}" for i in range(len(votes_files))]
        raw_paths = {"edges": raw_dir / edges_file}
        raw_paths.update({k: raw_dir / f for k, f in zip(vote_keys, votes_files)})

        super().__init__(
            dataset_name="digg",
            base_path=base_path,
            raw_paths=raw_paths,
            time_format="unix_seconds",  # votes and friendships are both Unix seconds
            edge_time_available=True,
            read_csv_kwargs={"sep": ",", "header": None},
        )
        self.votes_files = tuple(votes_files)
        self._vote_keys = vote_keys
        self.author_rule = author_rule
        self.subset_mode = subset_mode
        self.subset_value = subset_value
        self.seed = int(seed)
        self.restrict_edges = bool(restrict_edges)
        self.reverse_edges = bool(reverse_edges)
        self.expand_mutual = bool(expand_mutual)

        # The events/posts branch and the edges branch may run in different threads
        # (see main_loader), so the shared subset is prepared once, under a lock.
        self._lock = threading.RLock()
        self._votes: pd.DataFrame | None = None
        self._users: pd.Index | None = None

    @staticmethod
    def _check_subset_value(mode: str, value) -> None:
        if mode == "full":
            return
        if value is None:
            raise ValueError(f"subset_mode={mode!r} needs subset_value")
        if mode == "post_fraction" and not 0 < value <= 1:
            raise ValueError(f"post_fraction needs 0 < subset_value <= 1, got {value}")
        if mode in ("top_posts", "first_rows") and (int(value) != value or value < 1):
            raise ValueError(f"{mode} needs a positive integer subset_value, got {value}")

    # ------------------------------------------------------------------
    # Subset selection (shared by events, posts and edges)
    # ------------------------------------------------------------------
    def _load_votes(self) -> pd.DataFrame:
        """Read all vote files (stopping early for first_rows) into one table."""
        remaining = int(self.subset_value) if self.subset_mode == "first_rows" else None
        frames = []
        for key in self._vote_keys:
            kwargs = {} if remaining is None else {"nrows": remaining}
            df = self.load_raw_dataframe(key, VOTE_COLUMNS, **kwargs)
            frames.append(df)
            if remaining is not None:
                remaining -= len(df)
                if remaining <= 0:
                    break
        return pd.concat(frames, ignore_index=True)

    def _prepare_subset(self) -> None:
        """Load the votes once, apply the subset, and record which users are in it."""
        with self._lock:
            if self._votes is not None:
                return

            votes = self._load_votes()
            votes = votes.rename(columns={"post": "post_id"})[["post_id", "user", "time"]]

            n_posts_all = votes["post_id"].nunique()
            if self.subset_mode == "post_fraction":
                ids = votes["post_id"].unique()
                k = max(1, int(round(self.subset_value * len(ids))))
                chosen = np.random.default_rng(self.seed).choice(ids, size=k, replace=False)
                votes = votes[votes["post_id"].isin(chosen)]
            elif self.subset_mode == "top_posts":
                top = votes["post_id"].value_counts().nlargest(int(self.subset_value)).index
                votes = votes[votes["post_id"].isin(top)]
            votes = votes.reset_index(drop=True)

            if votes.empty:
                raise ValueError(f"Subset (mode={self.subset_mode}, value={self.subset_value}) left no events")

            self._users = pd.Index(pd.unique(votes["user"]))
            self._votes = votes

            log.info(
                "Subset '%s' (value=%s): %d of %d posts, %d events, %d users",
                self.subset_mode, self.subset_value, votes["post_id"].nunique(), n_posts_all,
                len(votes), len(self._users),
            )

    # ------------------------------------------------------------------
    # Canonical tables
    # ------------------------------------------------------------------
    def build_events(self) -> pd.DataFrame:
        self._prepare_subset()
        return self._votes[["post_id", "user", "time"]].copy()

    def build_posts(self, events: pd.DataFrame) -> pd.DataFrame:
        # `events` is cleaned and sorted by (time, post_id, user), so the first row of
        # each post is its earliest vote: the submitter's own digg. t0 = time of that vote.
        first = events.drop_duplicates("post_id", keep="first")
        return first.rename(columns={"user": "author", "time": "t0"})[["post_id", "author", "t0"]]

    def build_edges(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("edges", FRIEND_COLUMNS)

        # `user` follows `friend`, so information flows friend -> user.
        edges = df.rename(columns={"friend": "source", "user": "target"})[
            ["source", "target", "created_at", "mutual"]
        ]

        if self.expand_mutual:
            mutual = edges[edges["mutual"] == 1]
            reverse = mutual.rename(columns={"source": "target", "target": "source"})
            edges = pd.concat([edges, reverse], ignore_index=True)
            log.info("Mutual friendships: added %d reverse edges", len(reverse))

        if self.reverse_edges:
            edges = edges.rename(columns={"source": "target", "target": "source"})

        # created_at stays as raw Unix seconds; BaseAdapter parses and de-duplicates it.
        if self.subset_mode != "full" and self.restrict_edges:
            self._prepare_subset()
            n0 = len(edges)
            keep = edges["source"].isin(self._users) & edges["target"].isin(self._users)
            edges = edges[keep]
            log.info("Edges restricted to subset users: kept %d of %d", len(edges), n0)
        return edges[["source", "target", "created_at"]]

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def manifest_extra(self) -> dict:
        return {
            "author_rule": self.author_rule,
            "votes_files": list(self.votes_files),
            "subset_mode": self.subset_mode,
            "subset_value": self.subset_value,
            "subset_seed": self.seed if self.subset_mode == "post_fraction" else None,
            "restrict_edges": self.restrict_edges,
            "reverse_edges": self.reverse_edges,
            "expand_mutual": self.expand_mutual,
        }


if __name__ == "__main__":
    # Quick standalone run, e.g. a 5% random subset:
    #   python -m src.data.adapters.digg
    logging.basicConfig(level=logging.INFO)
    DiggAdapter(
        base_path=".",
        raw_dir="data/raw/digg",
        subset_mode="post_fraction",
        subset_value=0.05,
    ).process_and_save()