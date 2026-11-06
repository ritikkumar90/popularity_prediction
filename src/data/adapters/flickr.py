"""Flickr adapter (Mislove et al., WWW 2009 Flickr data).

Raw files (header-less, whitespace separated), all inside `raw_dir`:
  votes   flickr-all-photo-favorite-markings : user, post, time
  edges   flickr-growth                      : two user ids, date (YYYY-MM-DD)
  owners  flickr-all-photos (optional)       : photo_id, upload_time, owner,
                                               views, comments, favorites

Time
----
Edge dates are 'date:%Y-%m-%d'. Favorite times in this dataset are relative seconds
since the crawl started on 2006-11-02. convert_event_time() adds the Unix timestamp for
2006-11-02 (1162425600) to align them with the edges. BaseAdapter does the rest
(cleaning, de-duplication, sorting, edge lag).

Authors (author_rule)
---------------------
first_event : author = first user to favorite the photo (a proxy).
owner_file  : author = real photo owner from the owners file.

The owners file also holds views / comments / favorites. Those are totals at crawl
time and include events after t_obs, so they are never read into any output table
(they would leak the label).

Dataset size (subset_mode / subset_value)
-----------------------------------------
The full dataset is large. Use a subset for development and quick experiments:

  "full"           everything (subset_value ignored)
  "post_fraction"  random fraction of posts, 0 < value <= 1 (e.g. 0.05). Each kept post
                   keeps its COMPLETE event history, so labels stay correct. Recommended.
  "top_posts"      the `value` posts with the most favorites. WARNING: selecting by total
                   popularity biases the label distribution (it uses future information
                   for selection). Fine for debugging, not for reported results.
  "first_rows"     first `value` rows of the votes file. WARNING: posts near the cut can
                   have truncated histories, and the file may not be time-ordered.

With restrict_edges=True (default for any subset), edges are reduced to the subgraph
induced by the users in the subset (voters, plus photo owners when author_rule is
"owner_file"). That makes graph building much cheaper. Set it to False to keep the
full graph.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.adapters.base import BaseAdapter

log = logging.getLogger(__name__)

# Column order of the raw files, as documented by the dataset authors.
VOTE_COLUMNS = ["user", "post", "time"]
EDGE_COLUMNS = ["source", "target", "created_at"]
PHOTO_COLUMNS = ["post_id", "upload_time", "owner", "views", "comments", "favorites"]

CRAWL_START_UNIX = 1162425600.0  # Unix timestamp for 2006-11-02

# Default raw file names inside raw_dir (override via constructor arguments).
DEFAULT_VOTES_FILE  = "flickr-all-photo-favorite-markings.txt.gz"
DEFAULT_EDGES_FILE  = "flickr-growth.txt.gz"
DEFAULT_OWNERS_FILE = "flickr-all-photos.txt.gz"

SUBSET_MODES = ("full", "post_fraction", "top_posts", "first_rows")
AUTHOR_RULES = ("first_event", "owner_file")


class FlickrAdapter(BaseAdapter):
    def __init__(
        self,
        base_path: str | Path,
        raw_dir: str | Path,
        *,
        votes_file: str = DEFAULT_VOTES_FILE,
        edges_file: str = DEFAULT_EDGES_FILE,
        owners_file: str | None = DEFAULT_OWNERS_FILE,
        author_rule: str = "first_event",
        subset_mode: str = "full",
        subset_value: float | int | None = None,
        seed: int = 42,
        restrict_edges: bool = True,
        reverse_edges: bool = False,
    ):
        """
        base_path       project root; output goes to <base_path>/data/processed/flickr
        raw_dir         folder containing the raw files
        author_rule     "first_event" or "owner_file" (needs owners_file)
        subset_mode     "full" | "post_fraction" | "top_posts" | "first_rows" (see module docstring)
        subset_value    fraction (post_fraction) or count (top_posts, first_rows)
        seed            random seed for post_fraction (same seed -> same subset)
        restrict_edges  with a subset, keep only edges between users present in the subset
        reverse_edges   swap source/target if the raw edge file lists them the other way round
                        (information must flow source -> target)
        """
        if author_rule not in AUTHOR_RULES:
            raise ValueError(f"author_rule must be one of {AUTHOR_RULES}, got {author_rule!r}")
        if subset_mode not in SUBSET_MODES:
            raise ValueError(f"subset_mode must be one of {SUBSET_MODES}, got {subset_mode!r}")
        self._check_subset_value(subset_mode, subset_value)

        raw_dir = Path(raw_dir)
        raw_paths = {"votes": raw_dir / votes_file, "edges": raw_dir / edges_file}
        if author_rule == "owner_file":
            if not owners_file:
                raise ValueError("author_rule='owner_file' requires owners_file")
            raw_paths["owners"] = raw_dir / owners_file

        super().__init__(
            dataset_name="flickr",
            base_path=base_path,
            raw_paths=raw_paths,
            time_format="date:%Y-%m-%d",  # describes the EDGE dates
            edge_time_available=True,
            read_csv_kwargs={"sep": r"\s+", "header": None},
        )
        self.author_rule = author_rule
        self.subset_mode = subset_mode
        self.subset_value = subset_value
        self.seed = int(seed)
        self.restrict_edges = bool(restrict_edges)
        self.reverse_edges = bool(reverse_edges)

        # The events/posts branch and the edges branch may run in different threads
        # (see main_loader), so the shared subset is prepared once, under a lock.
        self._lock = threading.RLock()
        self._votes: pd.DataFrame | None = None
        self._users: pd.Index | None = None
        self._owners: pd.DataFrame | None = None

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
    # Time
    # ------------------------------------------------------------------
    def convert_event_time(self, values: pd.Series) -> pd.Series:
        # Shift relative elapsed seconds to standard Unix time starting 2006-11-02
        return pd.to_numeric(values, errors="coerce").astype("float64") + CRAWL_START_UNIX

    # ------------------------------------------------------------------
    # Subset selection (shared by events, posts and edges)
    # ------------------------------------------------------------------
    def _prepare_subset(self) -> None:
        """Load the votes once, apply the subset, and record which users are in it."""
        with self._lock:
            if self._votes is not None:
                return

            kwargs = {}
            if self.subset_mode == "first_rows":
                kwargs["nrows"] = int(self.subset_value)
            votes = self.load_raw_dataframe("votes", VOTE_COLUMNS, **kwargs)
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

            users = votes["user"]
            if self.author_rule == "owner_file":
                owners = self._load_photo_owners()
                owners = owners[owners["post_id"].isin(votes["post_id"])]
                users = pd.concat([users, owners["author"]], ignore_index=True)
            self._users = pd.Index(pd.unique(users))
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
        # `events` is cleaned and sorted by (time, post_id, user), so the first row
        # of each post is its earliest favorite. t0 = time of that first favorite.
        first = events.drop_duplicates("post_id", keep="first")
        posts = first.rename(columns={"user": "author", "time": "t0"})[["post_id", "author", "t0"]]

        if self.author_rule == "owner_file":
            owners = self._load_photo_owners()
            n0 = len(posts)
            # Inner merge keeps the id dtype intact; posts without a known owner are dropped.
            posts = posts.drop(columns="author").merge(owners, on="post_id", how="inner")
            log.info("Photo owners used as authors: %d of %d posts have a known owner", len(posts), n0)
            posts = posts[["post_id", "author", "t0"]]
        return posts

    def build_edges(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("edges", EDGE_COLUMNS)
        if self.reverse_edges:
            df = df.rename(columns={"source": "target", "target": "source"})

        # created_at stays as the raw date string; BaseAdapter parses it with
        # time_format and adds the 1-day lag so an edge only becomes visible after its day.
        if self.subset_mode != "full" and self.restrict_edges:
            self._prepare_subset()
            n0 = len(df)
            keep = df["source"].isin(self._users) & df["target"].isin(self._users)
            df = df[keep]
            log.info("Edges restricted to subset users: kept %d of %d", len(df), n0)
        return df[["source", "target", "created_at"]]

    # ------------------------------------------------------------------
    # Owners
    # ------------------------------------------------------------------
    def _load_photo_owners(self) -> pd.DataFrame:
        with self._lock:
            if self._owners is None:
                df = self.load_raw_dataframe("owners", PHOTO_COLUMNS)
                # Keep only id and owner; the count columns would leak future information.
                owners = df[["post_id", "owner"]].drop_duplicates("post_id", keep="first")
                self._owners = owners.rename(columns={"owner": "author"})
            return self._owners

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def manifest_extra(self) -> dict:
        return {
            "author_rule": self.author_rule,
            "subset_mode": self.subset_mode,
            "subset_value": self.subset_value,
            "subset_seed": self.seed if self.subset_mode == "post_fraction" else None,
            "restrict_edges": self.restrict_edges,
            "reverse_edges": self.reverse_edges,
        }


if __name__ == "__main__":
    # Quick standalone run, e.g. a 5% random subset:
    #   python -m src.data.adapters.flickr
    logging.basicConfig(level=logging.INFO)
    FlickrAdapter(
        base_path=".",
        raw_dir="data/raw/flickr",
        subset_mode="post_fraction",
        subset_value=0.05,
    ).process_and_save()