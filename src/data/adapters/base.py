"""Shared base class for all dataset adapters (Flickr, Digg, Higgs).

An adapter only maps a raw dataset onto three canonical tables. Everything
else (time conversion, cleaning, validation, saving) lives here, so every
dataset is treated identically.

Canonical tables
----------------
events : post_id, user, time          one row per (user, post) interaction
posts  : post_id, author, t0          one row per post
edges  : source, target[, created_at] information flows source -> target

What an adapter must return
---------------------------
build_events()        raw `time` values, as they appear in the file.
build_posts(events)   receives the CLEANED events (time already in unix seconds);
                      t0 must be returned in unix seconds.
build_edges()         raw `created_at` values, as they appear in the file. Leave
                      the column out for static graphs
                      (cfg.active.edge_time_available = False).

Time handling (cfg.active.time_format)
--------------------------------------
Supported: unix_seconds | unix_millis | date:<strptime format>.
Event times and edge times are both converted to unix seconds with this format.
If a dataset stores event times differently from edge times (Flickr: numeric
epoch-days for favorites, dates for edges), the adapter overrides
convert_event_time() and/or convert_edge_time().

Edge visibility lag is derived from the format, not configured: date-only edge
timestamps (e.g. date:%Y-%m-%d) get +86400 s, so an edge becomes visible only
after its day has ended (no same-day leakage). Adapters can override the
`edge_time_lag_seconds` property.
"""
from __future__ import annotations

import abc
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)

EVENT_COLS = ["post_id", "user", "time"]
POST_COLS = ["post_id", "author", "t0"]
EDGE_COLS_TIMED = ["source", "target", "created_at"]
EDGE_COLS_STATIC = ["source", "target"]

# strptime directives that mean a date format has time-of-day resolution
_SUBDAY_DIRECTIVES = ("%H", "%I", "%M", "%S", "%f", "%p", "%X", "%c")


def _parse_time_format(fmt: str) -> tuple[str, object]:
    """'unix_seconds' | 'unix_millis' | 'date:<fmt>' -> (kind, argument)."""
    if fmt == "unix_seconds":
        return "numeric", 1.0
    if fmt == "unix_millis":
        return "numeric", 0.001
    if fmt.startswith("date:") and len(fmt) > len("date:"):
        return "date", fmt[len("date:"):]
    raise ValueError(f"Unsupported time_format '{fmt}'. Use unix_seconds, unix_millis or date:<fmt>.")


class BaseAdapter(abc.ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.active_cfg = cfg.active

        self.time_format = self.active_cfg.time_format
        self._time_kind, self._time_arg = _parse_time_format(self.time_format)
        self.edge_time_available = bool(self.active_cfg.edge_time_available)

        self.processed_dir: Path = cfg.run.base_path / "data" / "processed" / self.dataset_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Adapter interface (dataset-specific)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def build_events(self) -> pd.DataFrame:
        """Return columns ['post_id', 'user', 'time'] with raw time values."""

    @abc.abstractmethod
    def build_posts(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return columns ['post_id', 'author', 't0'] with t0 in unix seconds.

        `events` is the cleaned events table (time in unix seconds), so there is
        no need to reload the raw votes file.
        """

    @abc.abstractmethod
    def build_edges(self) -> pd.DataFrame:
        """Return ['source', 'target', 'created_at'] with raw created_at values,
        or ['source', 'target'] for a static graph."""

    # ------------------------------------------------------------------
    # Time conversion (overridable per dataset)
    # ------------------------------------------------------------------
    def parse_times(self, values: pd.Series) -> pd.Series:
        """Raw values -> unix seconds (float, NaN if unparseable), per cfg time_format."""
        if self._time_kind == "numeric":
            return pd.to_numeric(values, errors="coerce").astype("float64") * self._time_arg
        dt = pd.to_datetime(values, format=self._time_arg, errors="coerce", utc=True)
        delta = dt - pd.Timestamp("1970-01-01", tz="UTC")
        return (delta // pd.Timedelta(seconds=1)).astype("float64")

    def convert_event_time(self, values: pd.Series) -> pd.Series:
        return self.parse_times(values)

    def convert_edge_time(self, values: pd.Series) -> pd.Series:
        return self.parse_times(values)

    @property
    def edge_time_lag_seconds(self) -> int:
        """Seconds added to created_at; 86400 when edge times are date-only."""
        if self._time_kind == "date" and not any(d in self._time_arg for d in _SUBDAY_DIRECTIVES):
            return 86400
        return 0

    # ------------------------------------------------------------------
    # Loading helper available to adapters
    # ------------------------------------------------------------------
    def load_raw_dataframe(self, path_key: str, column_names: list[str]) -> pd.DataFrame:
        """Load a raw file using the path resolver and pandas kwargs from config."""
        path = self.cfg.path(path_key)
        if path is None:
            raise ValueError(f"Path for '{path_key}' is not defined in the config.")

        column_names = list(column_names)
        kwargs = dict(self.active_cfg.read_csv_kwargs)
        log.info(f"Loading {path_key} from {path}")
        df = pd.read_csv(path, **kwargs)

        # Only treat the file as headerless when config says header=None explicitly.
        # (With the default header='infer' the first data row would be eaten as a header.)
        headerless = "header" in kwargs and kwargs["header"] is None
        if headerless:
            if df.shape[1] != len(column_names):
                raise ValueError(
                    f"'{path_key}' has {df.shape[1]} columns but config names "
                    f"{len(column_names)}: {column_names}"
                )
            df.columns = column_names
        else:
            missing = [c for c in column_names if c not in df.columns]
            if missing:
                raise ValueError(f"'{path_key}' is missing expected columns {missing}")
        return df

    # ------------------------------------------------------------------
    # Cleaning (shared by every dataset)
    # ------------------------------------------------------------------
    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} table is missing columns {missing}; got {list(df.columns)}")

    @staticmethod
    def _log_drop(table: str, before: int, after: int, reason: str) -> None:
        if before != after:
            log.info("%s: dropped %d rows (%s); %d remain", table, before - after, reason, after)

    def clean_events(self, events: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(events, EVENT_COLS, "events")
        ev = events[EVENT_COLS].copy()

        ev["time"] = self.convert_event_time(ev["time"])
        n0 = len(ev)
        ev = ev.dropna()
        self._log_drop("events", n0, len(ev), "null id or unparseable time")
        if ev.empty:
            raise ValueError(
                f"All {n0} events were dropped: event times could not be parsed with "
                f"time_format '{self.time_format}'. Override convert_event_time() if events "
                f"use a different time format than edges."
            )
        ev["time"] = ev["time"].round().astype("int64")

        # Deterministic order, then keep the earliest event per (post, user).
        ev = ev.sort_values(["time", "post_id", "user"], kind="stable")
        n1 = len(ev)
        ev = ev.drop_duplicates(["post_id", "user"], keep="first")
        self._log_drop("events", n1, len(ev), "duplicate (post, user)")
        return ev.reset_index(drop=True)

    def clean_posts(self, posts: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(posts, POST_COLS, "posts")
        p = posts[POST_COLS].copy()
        n0 = len(p)
        p = p.dropna()
        self._log_drop("posts", n0, len(p), "null post_id, author or t0")
        p["t0"] = p["t0"].round().astype("int64")

        dup = int(p["post_id"].duplicated().sum())
        if dup:
            raise ValueError(f"posts has {dup} duplicated post_id values; the adapter must return one row per post")
        return p.sort_values(["t0", "post_id"], kind="stable").reset_index(drop=True)

    def align_events_and_posts(self, events: pd.DataFrame, posts: pd.DataFrame):
        """Drop events whose post is not in posts; report posts without events."""
        known = events["post_id"].isin(posts["post_id"])
        if not known.all():
            log.warning("Dropping %d events whose post_id is not in posts", int((~known).sum()))
            events = events[known].reset_index(drop=True)
        no_events = int((~posts["post_id"].isin(events["post_id"])).sum())
        if no_events:
            log.info("%d posts have no events", no_events)
        return events, posts

    def clean_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        cols = EDGE_COLS_TIMED if self.edge_time_available else EDGE_COLS_STATIC
        self._require_columns(edges, cols, "edges")
        e = edges[cols].copy()

        if self.edge_time_available:
            e["created_at"] = self.convert_edge_time(e["created_at"])

        n0 = len(e)
        e = e.dropna()
        self._log_drop("edges", n0, len(e), "null endpoint or unparseable created_at")

        n1 = len(e)
        e = e[e["source"] != e["target"]]
        self._log_drop("edges", n1, len(e), "self-loops")

        if self.edge_time_available:
            e["created_at"] = e["created_at"].round().astype("int64") + self.edge_time_lag_seconds
            e = e.sort_values(["created_at", "source", "target"], kind="stable")
            n2 = len(e)
            e = e.drop_duplicates(["source", "target"], keep="first")  # keep earliest
            self._log_drop("edges", n2, len(e), "duplicate (source, target)")
        else:
            n2 = len(e)
            e = e.drop_duplicates(["source", "target"]).sort_values(["source", "target"], kind="stable")
            self._log_drop("edges", n2, len(e), "duplicate (source, target)")
        return e.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(t: int) -> str:
        """Unix seconds -> 'YYYY-MM-DD'; never raises (out-of-range values are shown as such)."""
        try:
            return pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%Y-%m-%d")
        except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime, NotImplementedError):
            return f"<out of range: {int(t)} s = {int(t) / 86400:.0f} days>"

    def check_time_ranges(self, events: pd.DataFrame, posts: pd.DataFrame) -> None:
        """Log event-time quantiles and warn about implausible values (outliers / wrong unit)."""
        q = events["time"].quantile([0.0, 0.01, 0.5, 0.99, 0.9999, 1.0])
        log.info("Event time quantiles (min, 1%%, median, 99%%, 99.99%%, max): %s",
                 " | ".join(self._fmt(v) for v in q))
        far = int((events["time"] > 4_102_444_800).sum())  # after year 2100
        early = int((events["time"] < 0).sum())            # before 1970
        if far or early:
            log.warning(
                "%d events are after year 2100 and %d before 1970 (of %d). These are almost "
                "certainly bad rows or a wrong time unit; they distort the 'end of data' used by the "
                "splitter. Inspect the raw time column before using the splits.",
                far, early, len(events),
            )

    def check_time_alignment(self, events: pd.DataFrame, edges: pd.DataFrame) -> None:
        """Warn if edge times and event times look like they are on different clocks."""
        if not self.edge_time_available or events.empty or edges.empty:
            return
        ev_lo, ev_hi = int(events["time"].min()), int(events["time"].max())
        ed_lo, ed_hi = int(edges["created_at"].min()), int(edges["created_at"].max())
        log.info("Event time range: %s -> %s", self._fmt(ev_lo), self._fmt(ev_hi))
        log.info("Edge  time range: %s -> %s", self._fmt(ed_lo), self._fmt(ed_hi))
        if ed_hi < ev_lo or ed_lo > ev_hi:
            log.warning(
                "Edge and event time ranges do NOT overlap. Their clocks or units probably "
                "differ, and the time filter on edges will be wrong. Check time_format."
            )

    @staticmethod
    def _id_kind(s: pd.Series) -> str:
        return "int" if s.dtype.kind in "iu" else str(s.dtype.kind)

    def validate(self, events: pd.DataFrame, posts: pd.DataFrame, edges: pd.DataFrame) -> None:
        if events.empty or posts.empty:
            raise ValueError(f"Empty table after cleaning: events={len(events)}, posts={len(posts)}")

        errors = []
        if not posts["post_id"].is_unique:
            errors.append("post_id is not unique in posts")
        if not events["time"].is_monotonic_increasing:
            errors.append("events are not sorted by time")
        if not posts["t0"].is_monotonic_increasing:
            errors.append("posts are not sorted by t0")
        if not events["post_id"].isin(posts["post_id"]).all():
            errors.append("some events reference a post_id missing from posts")
        if self.edge_time_available and not edges["created_at"].is_monotonic_increasing:
            errors.append("edges are not sorted by created_at")

        if len(edges):
            pairs = [
                ("events.post_id vs posts.post_id", events["post_id"], posts["post_id"]),
                ("events.user vs posts.author", events["user"], posts["author"]),
                ("events.user vs edges.source", events["user"], edges["source"]),
                ("edges.source vs edges.target", edges["source"], edges["target"]),
            ]
            for label, a, b in pairs:
                if self._id_kind(a) != self._id_kind(b):
                    errors.append(f"id dtype mismatch: {label} ({a.dtype} vs {b.dtype})")

        if errors:
            raise ValueError("Validation failed:\n - " + "\n - ".join(errors))

        # Warn-only: events that happen before their post's t0.
        t0_map = posts.set_index("post_id")["t0"]
        early = int((events["time"] < events["post_id"].map(t0_map)).sum())
        if early:
            log.warning(
                "%d events (%.2f%%) happen before their post's t0",
                early, 100.0 * early / len(events),
            )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(self, events: pd.DataFrame, posts: pd.DataFrame, edges: pd.DataFrame) -> None:
        events.to_parquet(self.processed_dir / "events.parquet", index=False)
        posts.to_parquet(self.processed_dir / "posts.parquet", index=False)
        edges.to_parquet(self.processed_dir / "edges.parquet", index=False)

        manifest = {
            "dataset": self.dataset_name,
            "profile": str(self.cfg.run.profile),
            "time_format": self.time_format,
            "edge_time_available": self.edge_time_available,
            "edge_time_lag_seconds": self.edge_time_lag_seconds,
            "n_events": int(len(events)),
            "n_posts": int(len(posts)),
            "n_edges": int(len(edges)),
            "event_time_range": [int(events["time"].min()), int(events["time"].max())],
            "post_t0_range": [int(posts["t0"].min()), int(posts["t0"].max())],
        }
        if self.edge_time_available and len(edges):
            manifest["edge_time_range"] = [int(edges["created_at"].min()), int(edges["created_at"].max())]
        with open(self.processed_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def process_and_save(self) -> None:
        log.info(f"Processing dataset: {self.dataset_name} (Profile: {self.cfg.run.profile})")

        events = self.clean_events(self.build_events())
        posts = self.clean_posts(self.build_posts(events))
        events, posts = self.align_events_and_posts(events, posts)
        edges = self.clean_edges(self.build_edges())

        self.check_time_ranges(events, posts)
        self.check_time_alignment(events, edges)
        self.validate(events, posts, edges)
        self.save(events, posts, edges)

        log.info(
            "Saved canonical tables to %s (events=%d, posts=%d, edges=%d)",
            self.processed_dir, len(events), len(posts), len(edges),
        )