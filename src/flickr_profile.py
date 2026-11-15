#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_flickr_profile.py
=========================
Flickr counterpart of extract_digg_profile.py.  Reads the REAL Flickr files and extracts every
statistic a synthetic-data generator needs to imitate them.

Same OUTPUT SCHEMA as the Digg script, so the generator can read either profile:
    "story" = photo,   "vote" = favorite marking (user favorited photo),
    "friend edge user_id -> friend_id" = user follows contact.

Input (auto-found in --data-dir, default data/raw/flickr):
    interaction file  (name contains fav/vote/like/mark):  a time column + a user column + a photo column
    graph file        (name contains growth/link/friend/contact/edge/graph/follow):
                      user, contact [, date] [, mutual-flag]      (2 to 4 columns)
  Files may be .txt/.csv/.tsv/.gz, space/tab/comma separated, with or without a header row,
  with '#' or '%' comment lines.  Timestamps may be unix seconds OR date strings (YYYY-MM-DD[ HH:MM:SS]).
  IDs must be integers.

Outputs (in --out-dir, default data/raw/flickr_profile):
    profile_report.txt        human-readable report  <- paste this back to Claude
    profile.json              machine-readable profile (the generator reads this)
    story_table.csv           one row per photo: final votes, window counts, peak, bucket
    user_table.csv            one row per voter: activity, degrees
    story_features_early_sample.csv   early-window features for the sampled photos used in section 11

Usage:
    python extract_flickr_profile.py
    python extract_flickr_profile.py --data-dir data/raw/flickr --out-dir data/raw/flickr_profile
Useful options:
    --skip-cv  --skip-svd  --skip-exposure       skip the slowest sections
    --sample-items N         keep only N random photos (all their votes) -> fast dry run
    --min-item-votes K       drop photos with fewer than K votes
    --early-sample N         photos used for early-signal models (default 20000)
    --max-exposure-edges N   cap on edges used in the exposure merge (default 8,000,000)
    --votes F [F ...]  --friends F          explicit file paths
    --vote-cols T,U,I        0-based file column index of time,user,photo (overrides auto-detection)
    --friend-cols U,F[,D[,M]]  0-based column index of user,contact[,date[,mutual]]; use -1 for "absent"

Design notes
  * Columns are mapped by value checks (date parsing, unix range, overlap with graph ids), never blindly;
    the mapping that was chosen is printed so you can verify it.  Override with --vote-cols/--friend-cols.
  * Every section runs inside try/except, so one failure does not kill the rest.
  * Nothing is assumed about the real data: every number is measured.
  * Flickr favorites live for months/years (Digg: hours/days), so windows, profiles and peaks are
    scaled up (windows up to 1 year, daily peak instead of hourly).
"""
import argparse
import gzip
import json
import os
import re
import time
import traceback
import warnings
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse import csgraph

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
DEFAULT_DATA_DIR = "data/raw/flickr"
VOTE_COLS = ["vote_date", "voter_id", "story_id"]
FRIEND_COLS = ["mutual", "friend_date", "user_id", "friend_id"]

WINDOWS_H = [1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 2160, 4320, 8760]   # hours after 1st vote
FEAT_W = [6, 24, 72, 168, 720]        # windows for per-story early features (hours)
EARLY_W = [24, 168, 720]              # windows used for the early-signal models (1 day, 1 week, 30 days)
ALL_W = sorted(set(float(w) for w in WINDOWS_H) | {w / 2 for w in FEAT_W})
KTH = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
QS = [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, .995, .999, 1.0]
BUCKET_EDGES = [0, .5, .8, .95, .99, 1.0]
BUCKET_NAMES = ["p00-50", "p50-80", "p80-95", "p95-99", "p99-100"]
ADOPT_SAMPLE = 60                      # photos sampled per size group in the adoption section
SKIP_EXT = (".py", ".json", ".md", ".pdf", ".png", ".jpg", ".ipynb", ".zip", ".tar", ".tar.gz", ".html", ".log", ".pkl")
FRIEND_KW = ("growth", "link", "friend", "contact", "edge", "graph", "follow", "network")
VOTE_KW = ("fav", "vote", "like", "mark", "interact", "action")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?(Z|[+-]\d{2}:?\d{2})?$")
TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

REPORT = []
PROFILE = OrderedDict()


# --------------------------------------------------------------------------- logging / helpers
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    REPORT.append(s)


def hdr(title):
    log("")
    log("=" * 8 + " " + title + " " + "=" * 8)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def section(name, title):
    def deco(fn):
        def wrapper(*a, **k):
            hdr(title)
            t0 = time.time()
            try:
                out = fn(*a, **k)
                if out is not None:
                    PROFILE[name] = out
                log(f"[{name}] done in {time.time() - t0:.1f}s")
                return out
            except Exception as e:  # keep going, record the failure
                log(f"[{name}] FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                PROFILE[name] = {"error": f"{type(e).__name__}: {e}"}
                return None
        return wrapper
    return deco


def qcurve(x, qs=QS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    return {f"{q:g}": float(np.quantile(x, q)) for q in qs}


def qline(d):
    return "  ".join(f"p{float(k) * 100:g}={val:,.4g}" for k, val in d.items())


def fmt_series(arr, maxlen=60):
    """Compact printing for long per-day arrays (JSON keeps the full array)."""
    arr = np.asarray(arr)
    if len(arr) <= maxlen:
        return arr.tolist()
    nw = int(np.ceil(len(arr) / 7))
    pad = nw * 7 - len(arr)
    w = np.r_[arr, np.zeros(pad, arr.dtype)].reshape(nw, 7).sum(axis=1)
    return f"[weekly sums of {len(arr)} daily values] {w.tolist()}"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    n, s = len(x), x.sum()
    if n == 0 or s == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def hill(x, xmin):
    x = np.asarray(x, dtype=float)
    x = x[x >= xmin]
    if len(x) < 20 or xmin < 1:
        return float("nan")
    return float(1 + len(x) / np.sum(np.log(x / (xmin - 0.5))))


def top_share(x, frac):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(np.ceil(frac * len(x))))
    return float(x[:k].sum() / x.sum())


def eta_sq(groups, y):
    df = pd.DataFrame({"g": groups, "y": y}).dropna()
    if df.empty:
        return float("nan")
    gm = df["y"].mean()
    ss_tot = ((df["y"] - gm) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    agg = df.groupby("g")["y"].agg(["count", "mean"])
    return float((agg["count"] * (agg["mean"] - gm) ** 2).sum() / ss_tot)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m])[0])


def auc_bal(x, y):
    """AUC (orientation as given) and best-threshold balanced accuracy (either direction)."""
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    r = stats.rankdata(x)
    auc = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    order = np.argsort(-x, kind="mergesort")
    xs, ys = x[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    last = np.r_[xs[1:] != xs[:-1], True]
    j = tp[last] / n1 - fp[last] / n0
    return float(auc), float(0.5 + 0.5 * np.max(np.abs(j)))


# --------------------------------------------------------------------------- file reading
def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def sniff(path):
    """Return (sep, has_header, n_leading_comment_lines)."""
    skip, first = 0, ""
    with _open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("%", "#")):
                skip += 1
                continue
            first = line.strip()
            break
    if "\t" in first:
        sep, parts = "\t", first.split("\t")
    elif "," in first:
        sep, parts = ",", first.split(",")
    else:
        sep, parts = r"\s+", first.split()
    header = not all(_isnum(p) or DATE_RE.match(p) or TIME_RE.match(p) for p in parts)
    return sep, header, skip


def merge_date_time_cols(df):
    """Whitespace-separated files split 'YYYY-MM-DD HH:MM:SS' into 2 columns: glue them back."""
    i = 0
    cols = list(df.columns)
    while i < len(cols) - 1:
        a, b = cols[i], cols[i + 1]
        if df[a].dtype == object and df[b].dtype == object:
            if DATE_RE.match(str(df[a].iloc[0])) and TIME_RE.match(str(df[b].iloc[0])):
                df[a] = df[a].astype(str) + " " + df[b].astype(str)
                df = df.drop(columns=[b])
                cols = list(df.columns)
                continue
        i += 1
    df.columns = range(df.shape[1])
    return df


def read_raw(path):
    """Read a headerless/headed delimited file -> (int64 DataFrame with columns 0..k-1, set of date-derived cols)."""
    sep, header, skip = sniff(path)
    df = pd.read_csv(path, sep=sep, header=0 if header else None, skiprows=skip,
                     low_memory=False, on_bad_lines="skip")
    df.columns = range(df.shape[1])
    if sep == r"\s+":
        df = merge_date_time_cols(df)
    datecols = set()
    for c in list(df.columns):
        if df[c].dtype == object:
            s = df[c]
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            if dt.notna().mean() > 0.9:
                df[c] = (dt - EPOCH) // pd.Timedelta(seconds=1)
                datecols.add(c)
            else:
                num = pd.to_numeric(s, errors="coerce")
                if num.isna().mean() > 0.01:
                    raise ValueError(f"{os.path.basename(path)} column {c} is neither numeric nor a date "
                                     f"(sample: {s.iloc[:3].tolist()}); IDs must be integers")
                df[c] = num
    n0 = len(df)
    df = df.dropna().astype(np.int64)
    log(f"  read {os.path.basename(path)}: rows={len(df):,} (dropped {n0 - len(df):,} unparsable) cols={df.shape[1]} "
        f"sep={sep!r} header_row_found={header} comment_lines_skipped={skip} date_string_columns={sorted(datecols)}")
    return df, datecols


def _time_scale(s):
    """Divisor that turns the column into unix seconds: 1 (seconds), 1000 (milliseconds), or 0 (not a timestamp).
    Quantile-based so a few outlier rows (e.g. 0 or far-future stamps) cannot hide a real timestamp column."""
    q1, med, q99 = (float(x) for x in np.quantile(s.values, [.01, .5, .99]))
    for div in (1, 1000):
        lo, m, hi = q1 / div, med / div, q99 / div
        if 9e8 < m < 2.1e9 and 9e8 < lo and hi < 2.1e9 and (hi - lo) < 4e8:
            return div
    return 0


def _unix_like(s):
    """True if a column looks like unix timestamps (seconds or milliseconds, 2000-2036, p1-p99 span < ~12 years)."""
    return _time_scale(s) > 0


def _describe_cols(df, n=3):
    rows = []
    for c in list(df.columns)[:n]:
        s = df[c]
        q = np.quantile(s.values, [0, .01, .5, .99, 1])
        rows.append(f"    col {c}: min={q[0]:,.0f} p1={q[1]:,.0f} median={q[2]:,.0f} p99={q[3]:,.0f} max={q[4]:,.0f} "
                    f"nunique(sample)={s.iloc[:500000].nunique():,}")
    return "\n".join(rows)


def _overlap(col, gids):
    u = np.unique(col.values[:2_000_000])
    return float(np.isin(u, gids).mean())


def map_vote_cols(df, datecols, gids, forced):
    """Return ({vote_date, voter_id, story_id} -> column idx, explanation)."""
    k = df.shape[1]
    if forced:
        idx = [int(x) for x in forced.split(",")]
        return {"vote_date": idx[0], "voter_id": idx[1], "story_id": idx[2]}, "forced by --vote-cols"
    if k < 3:
        raise ValueError(f"interaction file has {k} columns; need time + user + photo (3). "
                         "If it has no timestamp the temporal analysis is impossible.")
    cols = list(df.columns)[:3]
    if k > 3:
        log(f"  WARNING: interaction file has {k} columns; only the first 3 are considered (use --vote-cols to change)")
    # random sample (the file may be sorted by user/photo, so the head is not representative)
    head = df.iloc[:, :3].sample(n=min(len(df), 500000), random_state=0)
    log("  column diagnostics (full file):\n" + _describe_cols(df))
    cand = [c for c in cols if c in datecols] or [c for c in cols if _unix_like(head[c])]
    if not cand:
        raise ValueError("no timestamp-like column found in the interaction file (see column diagnostics above); "
                         "pass --vote-cols T,U,I  (0-based file column indices of time,user,photo)")
    tcol = cand[0]
    div = _time_scale(head[tcol]) if tcol not in datecols else 1
    how = "time=date/unix-range detection" + (f" (milliseconds, will divide by {div})" if div > 1 else "")
    if len(cand) > 1:
        how += f" (AMBIGUOUS: candidates {cand}, took first)"
    rest = [c for c in cols if c != tcol]
    ov = {c: _overlap(head[c], gids) for c in rest}
    if abs(ov[rest[0]] - ov[rest[1]]) > 0.2:
        user = max(rest, key=lambda c: ov[c])
        how += f"; user=column with higher overlap with graph ids {({c: round(x, 3) for c, x in ov.items()})}"
    else:
        user = rest[0]
        how += f"; user/photo split AMBIGUOUS (overlap with graph ids {({c: round(x, 3) for c, x in ov.items()})}) -> assumed file order"
    item = [c for c in rest if c != user][0]
    return {"vote_date": tcol, "voter_id": user, "story_id": item, "time_div": div}, how


def map_friend_cols(df, datecols, forced):
    """Return ({user_id, friend_id, friend_date|None, mutual|None}, explanation)."""
    k = df.shape[1]
    if forced:
        idx = [int(x) for x in forced.split(",")]
        m = {"user_id": idx[0], "friend_id": idx[1],
             "friend_date": idx[2] if len(idx) > 2 and idx[2] >= 0 else None,
             "mutual": idx[3] if len(idx) > 3 and idx[3] >= 0 else None}
        return m, "forced by --friend-cols"
    if k < 2:
        raise ValueError("graph file needs at least 2 columns")
    if k == 2:
        return {"user_id": 0, "friend_id": 1, "friend_date": None, "mutual": None}, "2 columns: user, contact"
    cols = list(df.columns)
    head = df.iloc[:200000]
    mut = [c for c in cols if set(pd.unique(head[c])) <= {0, 1}]
    ts = [c for c in cols if c in datecols] or [c for c in cols if _unix_like(head[c])]
    mutc = mut[0] if len(mut) == 1 else None
    tsc = ts[0] if ts else None
    rest = [c for c in cols if c not in (mutc, tsc)]
    if len(rest) < 2:
        log("  WARNING: could not isolate 2 id columns in graph file; assuming first two columns are user, contact")
        return {"user_id": 0, "friend_id": 1, "friend_date": None, "mutual": None}, "fallback file order"
    if len(rest) > 2:
        log(f"  WARNING: graph file has extra columns {rest[2:]}; ignoring them")
    return {"user_id": rest[0], "friend_id": rest[1], "friend_date": tsc, "mutual": mutc}, \
        "auto (id columns in file order; date by parse/unix range; mutual = binary column)"


def find_files(args):
    d = args.data_dir
    if not os.path.isdir(d):
        raise FileNotFoundError(f"data dir not found: {d}")
    allf = [n for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and not n.startswith(".") and not n.lower().endswith(SKIP_EXT)]
    log(f"  files in {d}: {allf}")
    low = {n: n.lower() for n in allf}
    if args.friends:
        fp = args.friends
    else:
        c = [n for n in allf if any(k in low[n] for k in FRIEND_KW) and not any(k in low[n] for k in VOTE_KW)]
        pref = [n for n in c if "growth" in low[n]]
        pick = pref[0] if pref else (c[0] if c else None)
        fp = os.path.join(d, pick) if pick else None
        if len(c) > 1:
            log(f"  graph-file candidates: {c}; using {pick} (pass --friends to change)")
    if args.votes:
        vp = args.votes
    else:
        c = [n for n in allf if any(k in low[n] for k in VOTE_KW)]
        if not c:
            c = [n for n in allf if os.path.join(d, n) != fp and not any(k in low[n] for k in FRIEND_KW)]
        vp = [os.path.join(d, n) for n in c]
    if not vp or not fp:
        raise FileNotFoundError(f"Need an interaction file and a graph file in {d} (found votes={vp}, friends={fp}). "
                                "Pass --votes/--friends explicitly.")
    return vp, fp


def load_friends(path, forced):
    raw, datecols = read_raw(path)
    m, how = map_friend_cols(raw, datecols, forced)
    log(f"  graph column mapping (0-based file col index): {m}  [{how}]")
    log("  NOTE: user_id -> friend_id direction (user follows contact) is taken from file column order, not detectable from values.")
    has_mut, has_date = m["mutual"] is not None, m["friend_date"] is not None
    e = pd.DataFrame({"user_id": raw[m["user_id"]].values, "friend_id": raw[m["friend_id"]].values})
    e["mutual"] = raw[m["mutual"]].values.astype(np.int8) if has_mut else np.int8(0)
    e["friend_date"] = raw[m["friend_date"]].values if has_date else np.int64(0)
    del raw
    info = {"file": os.path.basename(path), "rows_raw": int(len(e)), "mapping": m, "mapping_how": how,
            "has_mutual_column": has_mut, "has_date_column": has_date}
    n0 = len(e)
    e = e.drop_duplicates()
    info["exact_duplicate_rows_dropped"] = int(n0 - len(e))
    n0 = len(e)
    e = e.drop_duplicates(["user_id", "friend_id"], keep="first")
    info["duplicate_directed_pairs_dropped"] = int(n0 - len(e))
    n0 = len(e)
    e = e[e["user_id"] != e["friend_id"]]
    info["self_loops_dropped"] = int(n0 - len(e))
    if has_date:
        info["rows_with_nonpositive_date"] = int((e["friend_date"] <= 0).sum())
    return e.reset_index(drop=True), info


def load_votes(paths, gids, args):
    frames = []
    for p in paths:
        raw, datecols = read_raw(p)
        m, how = map_vote_cols(raw, datecols, gids, args.vote_cols)
        log(f"  vote column mapping (0-based file col index): {m}  [{how}]")
        fr = pd.DataFrame({k: raw[m[k]].values for k in VOTE_COLS})
        if m.get("time_div", 1) > 1:
            fr["vote_date"] = fr["vote_date"] // m["time_div"]
        frames.append(fr)
        del raw
    v = pd.concat(frames, ignore_index=True)
    info = {"rows_raw": int(len(v)), "files": [os.path.basename(p) for p in paths]}
    n0 = len(v)
    v = v.drop_duplicates()
    info["exact_duplicate_rows_dropped"] = int(n0 - len(v))
    n0 = len(v)
    v = v.sort_values(["vote_date", "voter_id"], kind="mergesort").drop_duplicates(["voter_id", "story_id"], keep="first")
    info["duplicate_user_item_pairs_dropped"] = int(n0 - len(v))
    # optional filtering / subsampling of photos
    vc = v["story_id"].value_counts()
    info["items_before_filter"] = int(len(vc))
    keep = vc.index[vc.values >= args.min_item_votes]
    if args.min_item_votes > 1:
        info["min_item_votes"] = int(args.min_item_votes)
    if args.sample_items and len(keep) > args.sample_items:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(keep.values, args.sample_items, replace=False)
        info["sampled_items"] = int(args.sample_items)
    if len(keep) < len(vc):
        v = v[v["story_id"].isin(keep)]
    info["items_after_filter"] = int(v["story_id"].nunique())
    v = v.reset_index(drop=True)
    for c in ("voter_id", "story_id"):
        if v[c].max() < 2 ** 31:
            v[c] = v[c].astype(np.int32)
    return v, info


# --------------------------------------------------------------------------- preparation
def prep(C):
    v = C.v
    v.sort_values(["story_id", "vote_date", "voter_id"], kind="mergesort", inplace=True)
    v.reset_index(drop=True, inplace=True)
    g = v.groupby("story_id", sort=True)
    v["t0"] = g["vote_date"].transform("min")
    v["age_h"] = (v["vote_date"] - v["t0"]) / 3600.0
    v["pos"] = g.cumcount() + 1
    v["n_story"] = g["vote_date"].transform("size")
    v["rank_frac"] = v["pos"] / v["n_story"]
    v["gap_s"] = g["vote_date"].diff()
    v["tie"] = v.duplicated(["story_id", "vote_date"], keep=False)
    v["log_story_n"] = np.log(v["n_story"])

    S = g.agg(n_votes=("vote_date", "size"), t_first=("vote_date", "min"), t_last=("vote_date", "max"))
    S["lifetime_h"] = (S["t_last"] - S["t_first"]) / 3600.0
    pct = S["n_votes"].rank(pct=True, method="average")
    S["bk"] = pd.cut(pct, BUCKET_EDGES, labels=False, include_lowest=True).astype(int)
    v["bk"] = S["bk"].reindex(v["story_id"]).values.astype(np.int8)
    C.starts = np.r_[0, np.cumsum(S["n_votes"].values)][:-1]
    age = v["age_h"].values
    for w in ALL_W:
        S[f"v_{w:g}h"] = np.add.reduceat((age <= w).astype(np.int64), C.starts)
    C.S = S

    U = v.groupby("voter_id").agg(n_votes=("story_id", "size"), t_first=("vote_date", "min"),
                                  t_last=("vote_date", "max"), mean_rank_frac=("rank_frac", "mean"),
                                  mean_log_story_n=("log_story_n", "mean"))
    C.U = U

    e = C.e
    u = e["user_id"].values.astype(np.int64)
    f = e["friend_id"].values.astype(np.int64)
    ids = C.gids
    gi = pd.Index(ids)
    C.gi_ids = ids
    C.aidx = gi.get_indexer(u)
    C.bidx = gi.get_indexer(f)
    del u, f
    C.N = len(ids)
    C.Adj = sparse.csr_matrix((np.ones(len(C.aidx), dtype=np.int8), (C.aidx, C.bidx)), shape=(C.N, C.N))
    C.out_deg = np.bincount(C.aidx, minlength=C.N)   # number of people this user follows
    C.in_deg = np.bincount(C.bidx, minlength=C.N)    # number of followers
    vgi = gi.get_indexer(v["voter_id"].values)
    C.v_gi = vgi
    ing = vgi >= 0
    safe = np.maximum(vgi, 0)
    v["in_graph"] = ing
    v["indeg"] = np.where(ing, C.in_deg[safe], 0)
    v["outdeg"] = np.where(ing, C.out_deg[safe], 0)
    ugi = gi.get_indexer(U.index.values)
    ui = ugi >= 0
    U["in_graph"] = ui
    U["in_deg"] = np.where(ui, C.in_deg[np.maximum(ugi, 0)], 0)
    U["out_deg"] = np.where(ui, C.out_deg[np.maximum(ugi, 0)], 0)
    pos_in = U.loc[U["in_deg"] > 0, "in_deg"]
    C.hub_thr = float(np.quantile(pos_in, 0.99)) if len(pos_in) else float("inf")
    v["is_hub"] = v["indeg"] >= C.hub_thr
    v["exposed"] = False   # filled by the exposure section


# --------------------------------------------------------------------------- sections
@section("overview", "1. LOADING & SANITY CHECKS")
def s_overview(C):
    v, S, U, e = C.v, C.S, C.U, C.e
    t0, t1 = int(v["vote_date"].min()), int(v["vote_date"].max())
    tt = v["vote_date"].values
    out = {
        "votes": int(len(v)), "stories": int(len(S)), "voters": int(len(U)),
        "vote_time_min": t0, "vote_time_max": t1,
        "vote_time_min_iso": str(pd.to_datetime(t0, unit="s")), "vote_time_max_iso": str(pd.to_datetime(t1, unit="s")),
        "span_days": (t1 - t0) / 86400.0,
        "votes_info": C.vinfo, "friends_info": C.finfo,
        "edges_after_cleaning": int(len(e)), "graph_users": int(C.N),
        "time_resolution": {
            "share_timestamps_at_utc_midnight": float((tt % 86400 == 0).mean()),
            "share_timestamps_on_whole_minute": float((tt % 60 == 0).mean()),
            "share_votes_with_same_second_as_previous_vote_in_story": float((v["gap_s"].values == 0).mean()),
            "n_distinct_vote_days": int(len(np.unique(tt // 86400))),
        },
    }
    log(f"votes={len(v):,}  photos={len(S):,}  voters={len(U):,}  edges={len(e):,}  graph users={C.N:,}")
    log(f"time range {out['vote_time_min_iso']} -> {out['vote_time_max_iso']}  ({out['span_days']:.1f} days)")
    log(f"votes cleaning: {C.vinfo}")
    log(f"friends cleaning: {C.finfo}")
    tr = out["time_resolution"]
    log(f"time resolution: {tr}")
    if tr["share_timestamps_at_utc_midnight"] > 0.9:
        log("  WARNING: >90% of timestamps are at UTC midnight -> DAY resolution only; hour-level windows/hour-of-day are meaningless.")
    checks = [
        ("timestamps look like unix seconds (2000-2036)", 9e8 < t0 and t1 < 2.1e9),
        (">100 distinct photos", len(S) > 100),
        (">100 distinct voters", len(U) > 100),
        ("graph has edges", len(e) > 0),
    ]
    for name, ok in checks:
        log(f"  [{'PASS' if ok else 'WARN'}] {name}")
    out["sanity_checks"] = {n: bool(ok) for n, ok in checks}
    return out


@section("story_popularity", "2. PHOTO POPULARITY (final votes per photo)")
def s_popularity(C):
    n = C.S["n_votes"].values.astype(float)
    out = {
        "n_stories": int(len(n)), "n_votes_total": int(n.sum()), "mean": float(n.mean()), "std": float(n.std()),
        "skew": float(stats.skew(n)), "skew_log": float(stats.skew(np.log(n))),
        "log_mu": float(np.log(n).mean()), "log_sigma": float(np.log(n).std()),
        "gini": gini(n), "top1pct_share": top_share(n, .01), "top10pct_share": top_share(n, .10),
        "top20pct_share": top_share(n, .20),
        "hill_alpha_xmin_p50": hill(n, max(1, np.quantile(n, .5))), "hill_alpha_xmin_p80": hill(n, max(1, np.quantile(n, .8))),
        "hill_alpha_xmin_p95": hill(n, max(1, np.quantile(n, .95))),
        "quantiles": qcurve(n),
        "share_single_vote": float((n == 1).mean()),
        "share_ge": {str(k): float((n >= k).mean()) for k in (2, 3, 5, 10, 20, 50, 100, 1000)},
        "count_value_histogram_1_to_20": np.bincount(np.minimum(n, 21).astype(int), minlength=22)[1:] / len(n),
    }
    labels = {}
    for nm, q in [("top50", .5), ("top20", .8), ("top10", .9)]:
        thr = float(np.quantile(n, q))
        labels[nm] = {"threshold_votes": thr, "positive_rate": float((n >= thr).mean())}
    for k in (2, 5, 10, 50, 100):
        labels[f"ge{k}"] = {"threshold_votes": float(k), "positive_rate": float((n >= k).mean())}
    out["label_rules"] = labels
    log("votes/photo quantiles:", qline(out["quantiles"]))
    log(f"mean={out['mean']:.2f} std={out['std']:.2f} log-normal fit mu={out['log_mu']:.3f} sigma={out['log_sigma']:.3f}  gini={out['gini']:.3f}")
    log(f"single-vote photos: {out['share_single_vote']:.1%};  share with >=k votes: {({k: round(x, 4) for k, x in out['share_ge'].items()})}")
    log(f"top1%/10%/20% of photos hold {out['top1pct_share']:.1%} / {out['top10pct_share']:.1%} / {out['top20pct_share']:.1%} of votes")
    log(f"Hill tail alpha (xmin=p50/p80/p95): {out['hill_alpha_xmin_p50']:.2f} / {out['hill_alpha_xmin_p80']:.2f} / {out['hill_alpha_xmin_p95']:.2f}")
    log("label rules:", labels)
    return out


@section("user_activity", "3. USER ACTIVITY (votes per user)")
def s_users(C):
    U = C.U
    n = U["n_votes"].values.astype(float)
    span = (U["t_last"] - U["t_first"]).values[n >= 2] / 3600.0
    out = {
        "n_voters": int(len(U)), "mean": float(n.mean()), "gini": gini(n),
        "quantiles": qcurve(n, [0, .1, .25, .5, .75, .9, .95, .99, .999, 1.0]),
        "share_one_vote": float((n == 1).mean()), "share_le_2": float((n <= 2).mean()),
        "share_ge_10": float((n >= 10).mean()), "share_ge_100": float((n >= 100).mean()),
        "share_ge_1000": float((n >= 1000).mean()),
        "top1pct_share": top_share(n, .01), "top10pct_share": top_share(n, .10),
        "hill_alpha_xmin_p50": hill(n, max(1, np.quantile(n, .5))), "hill_alpha_xmin_p90": hill(n, max(1, np.quantile(n, .9))),
        "active_span_h_quantiles_(users>=2votes)": qcurve(span, [.1, .25, .5, .75, .9]),
        "share_users_all_votes_within_24h_(users>=2votes)": float((span <= 24).mean()) if len(span) else None,
        "spearman_nvotes_vs_mean_rank_frac": spearman(n, U["mean_rank_frac"].values),
        "spearman_nvotes_vs_mean_log_story_size": spearman(n, U["mean_log_story_n"].values),
        "mean_rank_frac_overall": float(U["mean_rank_frac"].mean()),
    }
    log("votes/user quantiles:", qline(out["quantiles"]))
    log(f"mean={out['mean']:.2f} gini={out['gini']:.3f} one-vote users={out['share_one_vote']:.1%} (>=10 votes {out['share_ge_10']:.1%}, >=100 {out['share_ge_100']:.2%}, >=1000 {out['share_ge_1000']:.3%})")
    log(f"top1% of users cast {out['top1pct_share']:.1%} of votes; Hill alpha (p50/p90) {out['hill_alpha_xmin_p50']:.2f} / {out['hill_alpha_xmin_p90']:.2f}")
    log(f"active span (h), users with >=2 votes: {qline(out['active_span_h_quantiles_(users>=2votes)'])}")
    log(f"heavy voters vote earlier in cascades? Spearman(n_votes, mean cascade position)={out['spearman_nvotes_vs_mean_rank_frac']:.3f}")
    log(f"heavy voters pick bigger photos? Spearman(n_votes, mean log photo size)={out['spearman_nvotes_vs_mean_log_story_size']:.3f}")
    return out


@section("temporal", "4. TEMPORAL DYNAMICS (vote time vs. photo age; age = hours since the photo's FIRST observed vote)")
def s_temporal(C):
    v, S = C.v, C.S
    out = {}
    present = set(S["bk"].unique().tolist())
    names = [nm for i, nm in enumerate(BUCKET_NAMES) if i in present] + ["all"]   # skip empty size buckets

    def sub_of(nm):
        return S if nm == "all" else S[S["bk"] == BUCKET_NAMES.index(nm)]

    # --- cumulative votes at fixed windows, by size bucket
    casc = {}
    for nm in names:
        sub = sub_of(nm)
        casc[nm] = {"n_stories": int(len(sub))}
        for w in WINDOWS_H:
            a = sub[f"v_{w:g}h"].values.astype(float)
            sh = a / sub["n_votes"].values
            casc[nm][f"{w:g}"] = {
                "abs_mean": float(a.mean()), "abs_median": float(np.median(a)), "abs_p10": float(np.quantile(a, .1)),
                "abs_p90": float(np.quantile(a, .9)), "share_mean": float(sh.mean()), "share_median": float(np.median(sh)),
                "share_p10": float(np.quantile(sh, .1)), "share_p90": float(np.quantile(sh, .9)),
            }
    out["cascade_by_bucket"] = casc
    show = [1, 6, 24, 72, 168, 720, 2160, 8760]
    log("MEDIAN votes accumulated by age (rows=final-size bucket, cols=hours since first vote):")
    log(pd.DataFrame({f"{w:g}h": [casc[nm][f"{w:g}"]["abs_median"] for nm in names] for w in show}, index=names).round(1).to_string())
    log("MEDIAN share of final votes reached by age:")
    log(pd.DataFrame({f"{w:g}h": [casc[nm][f"{w:g}"]["share_median"] for nm in names] for w in show}, index=names).round(3).to_string())
    log("MEAN share of final votes reached by age:")
    log(pd.DataFrame({f"{w:g}h": [casc[nm][f"{w:g}"]["share_mean"] for nm in names] for w in show}, index=names).round(3).to_string())

    # --- time to k-th vote
    kth = {}
    bk_counts = np.bincount(S["bk"].values, minlength=len(BUCKET_NAMES))
    for k in KTH:
        m = v["pos"].values == k
        ages, bks = v["age_h"].values[m], v["bk"].values[m]
        row = {"reached_share_all": float(m.sum() / len(S)), "age_h_all": qcurve(ages, [.1, .5, .9])}
        for bi, nm in enumerate(BUCKET_NAMES):
            a = ages[bks == bi]
            row[nm] = {"reached_share": float(len(a) / max(1, bk_counts[bi])), "age_h": qcurve(a, [.1, .5, .9])}
        kth[str(k)] = row
    out["time_to_kth_vote"] = kth
    log("time to k-th vote (hours, median [p10-p90]) over photos that reach k:")
    for k in KTH:
        r = kth[str(k)]["age_h_all"]
        if r:
            log(f"  k={k:<5} reached by {kth[str(k)]['reached_share_all']:.1%}  median={r['0.5']:.2f}h  [{r['0.1']:.2f} - {r['0.9']:.2f}]")

    # --- time to reach X% of the photo's final votes
    tts = {}
    for qf in [.1, .25, .5, .75, .9]:
        target = np.ceil(qf * v["n_story"].values)
        m = v["pos"].values == target
        ages = v["age_h"].values[m]
        ages_story = pd.Series(ages, index=v["story_id"].values[m]).groupby(level=0).min()
        row = {"all": qcurve(ages_story.values, [.1, .5, .9])}
        bk_story = S["bk"].reindex(ages_story.index).values
        for bi, nm in enumerate(BUCKET_NAMES):
            row[nm] = qcurve(ages_story.values[bk_story == bi], [.1, .5, .9])
        tts[f"{qf:g}"] = row
    out["age_h_to_reach_share_of_final"] = tts
    log("age (h) at which a photo reaches X% of its final votes (median):")
    log(pd.DataFrame({f"{q}": [tts[q][nm].get("0.5", np.nan) for nm in names] for q in tts}, index=names).round(1).to_string())

    # --- vote-rate profile vs age
    edges_log = np.r_[0, np.geomspace(0.05, 50000, 61)]
    edges_d = np.arange(0, 366, 1.0)
    prof = {"edges_log_h": edges_log, "edges_daily_d": edges_d}
    decay = {}
    ages_all = v["age_h"].values
    for nm in names:
        m = np.ones(len(v), bool) if nm == "all" else (v["bk"].values == BUCKET_NAMES.index(nm))
        a = ages_all[m]
        cl, _ = np.histogram(a, bins=edges_log)
        ch, _ = np.histogram(a / 24.0, bins=edges_d)
        sl = cl / max(1, len(a))
        prof[nm] = {"share_per_log_bin": sl, "rate_per_h_log_bin": sl / np.diff(edges_log),
                    "share_per_day_0_365d": ch / max(1, len(a))}
        cen = np.sqrt(np.maximum(edges_log[:-1], 1e-9) * edges_log[1:])
        sel = (edges_log[:-1] >= 24) & (edges_log[1:] <= 5000) & (sl > 0)
        if sel.sum() >= 4:
            slope = np.polyfit(np.log(cen[sel]), np.log(sl[sel] / np.diff(edges_log)[sel]), 1)[0]
            decay[nm] = float(slope)
    out["age_profile"] = prof
    out["late_decay_loglog_slope_24h_to_5000h"] = decay
    log("late-life decay: log-log slope of vote rate vs age (24h-5000h):", {k: round(x, 2) for k, x in decay.items()})

    # --- peak DAY (Flickr photos live for months, so hourly peaks are replaced by daily peaks)
    ab = np.floor(v["age_h"].values / 24.0).astype(np.int64)
    cnt = pd.DataFrame({"story_id": v["story_id"].values, "ab": ab}).groupby(["story_id", "ab"]).size().reset_index(name="c")
    cnt = cnt.sort_values(["story_id", "c", "ab"], ascending=[True, False, True]).drop_duplicates("story_id").set_index("story_id")
    S["peak_age_d"] = cnt["ab"].reindex(S.index).values
    S["peak_votes_d"] = cnt["c"].reindex(S.index).values
    S["peak_share"] = S["peak_votes_d"] / S["n_votes"]
    pk = {}
    for nm in names:
        sub = sub_of(nm)
        pk[nm] = {"peak_age_d": qcurve(sub["peak_age_d"].values, [.1, .5, .9]),
                  "peak_votes_per_day": qcurve(sub["peak_votes_d"].values, [.1, .5, .9]),
                  "peak_share_of_final": qcurve(sub["peak_share"].values, [.1, .5, .9])}
    out["peak_day"] = pk
    log("peak vote-day age (days, median) by bucket:", {nm: round(pk[nm]["peak_age_d"].get("0.5", np.nan), 1) for nm in names})
    log("peak-day share of final votes (median) by bucket:", {nm: round(pk[nm]["peak_share_of_final"].get("0.5", np.nan), 3) for nm in names})

    # --- lifetime / censoring / first-vote-time distribution
    t_end, t_start = int(v["vote_date"].max()), int(v["vote_date"].min())
    left_h = (t_end - S["t_first"].values) / 3600.0
    out["lifetime_h"] = {nm: qcurve(sub_of(nm)["lifetime_h"].values, [.1, .5, .9]) for nm in names}
    out["share_stories_first_vote_within_48h_of_end"] = float((left_h < 48).mean())
    out["share_stories_first_vote_within_168h_of_end"] = float((left_h < 168).mean())
    out["share_stories_first_vote_within_720h_of_end"] = float((left_h < 720).mean())
    out["story_first_vote_per_day_since_start"] = np.bincount(((S["t_first"].values - t_start) // 86400).astype(int))
    log(f"lifetime (h) median by bucket: { {nm: round(out['lifetime_h'][nm].get('0.5', np.nan), 0) for nm in names} }")
    log(f"photos first voted <48h / <168h / <720h before dataset end (right-censored): "
        f"{out['share_stories_first_vote_within_48h_of_end']:.1%} / {out['share_stories_first_vote_within_168h_of_end']:.1%} / {out['share_stories_first_vote_within_720h_of_end']:.1%}")
    log("photos first-voted per day since start:", fmt_series(out["story_first_vote_per_day_since_start"]))
    return out


@section("calendar", "5. CALENDAR RHYTHMS & TIMESTAMP TIES")
def s_calendar(C):
    v, S = C.v, C.S
    dt = pd.to_datetime(v["vote_date"].values, unit="s", utc=True)
    hour, wd = np.asarray(dt.hour), np.asarray(dt.weekday)
    dt1 = pd.to_datetime(S["t_first"].values, unit="s", utc=True)
    h1, w1 = np.asarray(dt1.hour), np.asarray(dt1.weekday)
    t_start = int(v["vote_date"].min())
    day = ((v["vote_date"].values - t_start) // 86400).astype(int)
    logn = np.log(S["n_votes"].values)
    out = {
        "timezone": "UTC",
        "votes_share_by_hour": np.bincount(hour, minlength=24) / len(v),
        "votes_share_by_weekday_(0=Mon)": np.bincount(wd, minlength=7) / len(v),
        "votes_per_day_since_start": np.bincount(day),
        "story_first_vote_share_by_hour": np.bincount(h1, minlength=24) / len(S),
        "story_first_vote_share_by_weekday": np.bincount(w1, minlength=7) / len(S),
        "mean_log_final_votes_by_first_vote_hour": [float(logn[h1 == h].mean()) if (h1 == h).any() else None for h in range(24)],
        "eta2_first_vote_hour_vs_log_final": eta_sq(h1, logn),
        "eta2_first_vote_weekday_vs_log_final": eta_sq(w1, logn),
    }
    log("share of votes by hour of day (UTC):", np.round(out["votes_share_by_hour"], 3).tolist())
    log("share of votes by weekday (0=Mon):", np.round(out["votes_share_by_weekday_(0=Mon)"], 3).tolist())
    log("votes per day since start:", fmt_series(out["votes_per_day_since_start"]))
    log(f"eta^2 (first-vote hour / weekday vs log final votes): {out['eta2_first_vote_hour_vs_log_final']:.3f} / {out['eta2_first_vote_weekday_vs_log_final']:.3f}")

    tie = v["tie"].values
    ab = np.digitize(v["age_h"].values, [1, 24])
    out["tie_share_votes"] = float(tie.mean())
    out["tie_share_by_age"] = {n: float(tie[ab == i].mean()) if (ab == i).any() else None for i, n in enumerate(["<1h", "1-24h", ">24h"])}
    out["tie_share_by_bucket"] = {nm: (float(tie[v["bk"].values == i].mean()) if (v["bk"].values == i).any() else None) for i, nm in enumerate(BUCKET_NAMES)}
    gs = v.groupby(["story_id", "vote_date"]).size().values
    out["same_second_group_size_share_of_groups"] = {k: float(((gs == k) if k < 5 else (gs >= 5)).mean()) for k in range(1, 6)}
    t2 = v.loc[v["pos"] == 2].set_index("story_id")["vote_date"]
    out["share_stories_first_two_votes_same_second"] = float((t2.reindex(S.index).values == S["t_first"].values).mean())
    log(f"share of votes sharing a (photo, second) with another vote: {out['tie_share_votes']:.1%}")
    log("  by age:", {k: round(x, 3) if x is not None else None for k, x in out["tie_share_by_age"].items()},
        " by bucket:", {k: (round(x, 3) if x is not None else None) for k, x in out["tie_share_by_bucket"].items()})
    log("  same-second group sizes (share of groups; 5 = 5+):", {k: round(x, 3) for k, x in out["same_second_group_size_share_of_groups"].items()})
    log(f"  photos whose first two votes share a second: {out['share_stories_first_two_votes_same_second']:.1%}")
    return out


@section("friend_graph", "6. CONTACT GRAPH (edge user_id -> friend_id = 'user follows contact')")
def s_graph(C):
    e = C.e
    key = (C.aidx.astype(np.int64) << 32) | C.bidx.astype(np.int64)
    rkey = (C.bidx.astype(np.int64) << 32) | C.aidx.astype(np.int64)
    has_rev = np.isin(rkey, key)
    out = {
        "edges": int(len(e)), "graph_users": int(C.N),
        "has_mutual_flag_column": bool(C.has_mutual), "has_edge_date_column": bool(C.has_fdate),
        "reciprocity_edges_with_reverse": float(has_rev.mean()),
    }
    out["edges_if_mutual_pairs_collapsed"] = int((~has_rev).sum() + has_rev.sum() // 2)
    mut = None
    if C.has_mutual:
        mut = e["mutual"].values == 1
        out.update({
            "mutual_flag_share": float(mut.mean()),
            "mutual_rows_with_reverse_present": float(has_rev[mut].mean()) if mut.any() else None,
            "nonmutual_rows_with_reverse_present": float(has_rev[~mut].mean()) if (~mut).any() else None,
            "mutual_flag_agrees_with_reverse_presence": float((mut == has_rev).mean()),
        })
        out["mutual_edges_listed_in_both_directions"] = bool(out["mutual_rows_with_reverse_present"] is not None and out["mutual_rows_with_reverse_present"] > 0.95)
    # date relation inside reciprocal pairs
    if C.has_fdate:
        mm = (mut & has_rev) if mut is not None else has_rev
        idxs = np.flatnonzero(mm)
        if len(idxs) > 2_000_000:
            idxs = np.random.default_rng(C.seed).choice(idxs, 2_000_000, replace=False)
        if len(idxs):
            dates = e["friend_date"].values
            order = np.argsort(key)
            ks = key[order]
            p = np.searchsorted(ks, rkey[idxs])
            rd = dates[order][p]
            d = (dates[idxs] - rd) / 86400.0
            out["reciprocal_pair_date_gap_days_abs"] = qcurve(np.abs(d), [.1, .25, .5, .75, .9, .99])
            out["reciprocal_pair_share_identical_date"] = float((d == 0).mean())
    # degrees
    od, idg = C.out_deg.astype(float), C.in_deg.astype(float)
    for nm, d in [("out_degree_(people_followed)", od), ("in_degree_(followers)", idg)]:
        pos = d[d > 0]
        out[nm] = {
            "share_zero_among_graph_users": float((d == 0).mean()),
            "quantiles_all_graph_users": qcurve(d, [.5, .75, .9, .95, .99, .999, 1.0]),
            "quantiles_positive_only": qcurve(pos, [.1, .25, .5, .75, .9, .95, .99, .999, 1.0]),
            "mean_all": float(d.mean()), "gini_all": gini(d),
            "hill_alpha_xmin_p50_pos": hill(pos, max(1, np.quantile(pos, .5))) if len(pos) else None,
            "hill_alpha_xmin_p90_pos": hill(pos, max(1, np.quantile(pos, .9))) if len(pos) else None,
            "top_20_values": np.sort(d)[::-1][:20],
        }
    out["spearman_in_vs_out_degree"] = spearman(idg, od)
    # connected components
    ncomp, lab = csgraph.connected_components(C.Adj, directed=False)
    sz = np.bincount(lab)
    out["weak_components"] = int(ncomp)
    out["largest_component_share"] = float(sz.max() / C.N)
    t_start, t_end = int(C.v["vote_date"].min()), int(C.v["vote_date"].max())
    if C.has_fdate:
        d = e["friend_date"].values.astype(np.int64)
        d = d[d > 0]
        out["friend_date_quantiles_iso"] = {f"{q:g}": str(pd.to_datetime(np.quantile(d, q), unit="s")) for q in [0, .1, .25, .5, .75, .9, 1]}
        mo = pd.to_datetime(d, unit="s").to_period("M").astype(str)
        out["edges_by_month"] = mo.value_counts().sort_index().to_dict()
        tf = C.S["t_first"].values
        cut = {"vote_window_start": t_start, "vote_window_end": t_end}
        for q in (.5, .7, .8):
            cut[f"story_first_vote_p{int(q * 100)}"] = int(np.quantile(tf, q))
        out["share_edges_created_after_cutoff"] = {k: float((d > t).mean()) for k, t in cut.items()}
        inwin = d[(d >= t_start) & (d <= t_end)]
        out["edges_created_per_day_inside_vote_window"] = np.bincount(((inwin - t_start) // 86400).astype(int))
    log(f"edges={out['edges']:,} (collapsed-reciprocal count would be {out['edges_if_mutual_pairs_collapsed']:,})  users={C.N:,}")
    log(f"reciprocity (edge has reverse edge)={out['reciprocity_edges_with_reverse']:.1%}; mutual-flag column present={C.has_mutual}; date column present={C.has_fdate}")
    if C.has_mutual:
        log(f"  mutual flag share={out['mutual_flag_share']:.1%}; mutual rows with reverse present={out['mutual_rows_with_reverse_present']}; "
            f"non-mutual rows with reverse={out['nonmutual_rows_with_reverse_present']}; listed in both directions: {out['mutual_edges_listed_in_both_directions']}")
    for nm in ("out_degree_(people_followed)", "in_degree_(followers)"):
        o = out[nm]
        log(f"{nm}: zero-share={o['share_zero_among_graph_users']:.1%} mean={o['mean_all']:.2f} gini={o['gini_all']:.2f} alpha(p50/p90)={o['hill_alpha_xmin_p50_pos']} / {o['hill_alpha_xmin_p90_pos']}")
        log("   positive-only quantiles:", qline(o["quantiles_positive_only"]))
    log(f"Spearman(in,out degree)={out['spearman_in_vs_out_degree']:.3f}; components={ncomp:,}, largest={out['largest_component_share']:.1%}")
    if C.has_fdate:
        log("friend_date quantiles:", out["friend_date_quantiles_iso"])
        log("share of edges created AFTER cutoffs (leakage risk):", {k: round(x, 3) for k, x in out["share_edges_created_after_cutoff"].items()})
        if "reciprocal_pair_date_gap_days_abs" in out:
            log("reciprocal-pair date gap (days, abs):", qline(out["reciprocal_pair_date_gap_days_abs"]), f" identical date share={out['reciprocal_pair_share_identical_date']:.1%}")
    return out


@section("coverage_ids", "7. VOTER/GRAPH COVERAGE & ID SPACE")
def s_coverage(C):
    v, S, U = C.v, C.S, C.U
    voters = U.index.values.astype(np.int64)
    in_g = U["in_graph"].values
    gset = C.gi_ids
    out = {
        "share_voters_in_graph": float(in_g.mean()),
        "share_voters_with_out_edge": float((U["out_deg"] > 0).mean()),
        "share_voters_with_in_edge": float((U["in_deg"] > 0).mean()),
        "share_votes_by_in_graph_voters": float(v["in_graph"].mean()),
        "graph_users_never_voted": int(C.N - in_g.sum()),
        "graph_users_never_voted_share": float((C.N - in_g.sum()) / C.N),
        "voter_id": {"min": int(voters.min()), "max": int(voters.max()), "n": int(len(voters)),
                     "density_in_range": float(len(voters) / (voters.max() - voters.min() + 1))},
        "graph_user_id": {"min": int(gset.min()), "max": int(gset.max()), "n": int(len(gset)),
                          "density_in_range": float(len(gset) / (gset.max() - gset.min() + 1))},
        "story_id": {"min": int(S.index.min()), "max": int(S.index.max()), "n": int(len(S)),
                     "density_in_range": float(len(S) / (S.index.max() - S.index.min() + 1))},
        "votes_per_user_in_graph_voters": qcurve(U.loc[in_g, "n_votes"].values, [.25, .5, .75, .9, .99]),
        "votes_per_user_not_in_graph": qcurve(U.loc[~in_g, "n_votes"].values, [.25, .5, .75, .9, .99]),
        "in_degree_voters_in_graph": qcurve(U.loc[in_g, "in_deg"].values, [.25, .5, .75, .9, .99, 1.0]),
        "out_degree_voters_in_graph": qcurve(U.loc[in_g, "out_deg"].values, [.25, .5, .75, .9, .99, 1.0]),
        "spearman_nvotes_vs_in_deg_(in_graph_voters)": spearman(U.loc[in_g, "n_votes"], U.loc[in_g, "in_deg"]),
        "spearman_nvotes_vs_out_deg_(in_graph_voters)": spearman(U.loc[in_g, "n_votes"], U.loc[in_g, "out_deg"]),
        "spearman_nvotes_vs_in_deg_(all_voters_zero_filled)": spearman(U["n_votes"], U["in_deg"]),
        "spearman_user_id_vs_nvotes": spearman(voters, U["n_votes"].values),
        "spearman_user_id_vs_first_vote_time": spearman(voters, U["t_first"].values),
        "hub_in_degree_threshold_(p99_of_in_graph_voters)": C.hub_thr,
    }
    ng = np.setdiff1d(gset, voters, assume_unique=False)
    pos = np.searchsorted(gset, ng)
    out["in_degree_graph_nonvoters"] = qcurve(C.in_deg[pos], [.5, .9, .99, 1.0]) if len(ng) else {}
    out["out_degree_graph_nonvoters"] = qcurve(C.out_deg[pos], [.5, .9, .99, 1.0]) if len(ng) else {}
    log(f"voters in contact graph: {out['share_voters_in_graph']:.1%} (with out-edge {out['share_voters_with_out_edge']:.1%}, with followers {out['share_voters_with_in_edge']:.1%})")
    log(f"votes cast by in-graph voters: {out['share_votes_by_in_graph_voters']:.1%};  graph users who never voted: {out['graph_users_never_voted']:,} ({out['graph_users_never_voted_share']:.1%})")
    log("ID space:", {k: out[k] for k in ("voter_id", "graph_user_id", "story_id")})
    log("votes/user: in-graph voters", qline(out["votes_per_user_in_graph_voters"]))
    log("votes/user: not in graph   ", qline(out["votes_per_user_not_in_graph"]))
    log("in-degree of in-graph voters:", qline(out["in_degree_voters_in_graph"]))
    log("in-degree of graph non-voters:", qline(out["in_degree_graph_nonvoters"]))
    log(f"Spearman(n_votes, in_deg | in-graph voters)={out['spearman_nvotes_vs_in_deg_(in_graph_voters)']:.3f}   (n_votes, out_deg)={out['spearman_nvotes_vs_out_deg_(in_graph_voters)']:.3f}")
    log(f"Spearman(user_id, n_votes)={out['spearman_user_id_vs_nvotes']:.3f}  (user_id, first vote time)={out['spearman_user_id_vs_first_vote_time']:.3f}")
    return out


# ----- exposure machinery
def compute_exposure(C, direction, chunk=20000):
    """Vote rows whose voter had >=1 'influencing' neighbour vote earlier on the same photo.
    direction 'followee': voter follows the earlier voter (user_id=voter, friend_id=earlier voter)  [influence]
    direction 'follower': the earlier voter follows the current voter                                [placebo]"""
    a_all, b_all = (C.aidx, C.bidx) if direction == "followee" else (C.bidx, C.aidx)
    ing = C.v_gi >= 0
    vm = np.zeros(C.N, bool)
    vm[C.v_gi[ing]] = True
    sel = vm[a_all] & vm[b_all]
    a, b = a_all[sel].astype(np.int64), b_all[sel].astype(np.int64)
    n_candidate_edges = int(len(a))
    sampled = False
    cap = C.args.max_exposure_edges
    if cap and len(a) > cap:
        rng = np.random.default_rng(C.seed)
        ua = np.unique(a)
        keep_u = np.zeros(C.N, bool)
        keep_u[ua[rng.random(len(ua)) < cap / len(a)]] = True
        k = keep_u[a]
        a, b = a[k], b[k]
        sampled = True
    rows = np.flatnonzero(ing)
    Vg = pd.DataFrame({"gi": C.v_gi[ing].astype(np.int64), "story_id": C.v["story_id"].values[ing],
                       "t": C.v["vote_date"].values[ing], "row": rows})
    Vb = Vg.rename(columns={"gi": "b", "t": "t_b"})[["b", "story_id", "t_b"]]
    Va = Vg.rename(columns={"gi": "a", "t": "t_a"})[["a", "story_id", "t_a", "row"]]
    parts = []
    for s in range(0, len(a), chunk):
        E = pd.DataFrame({"a": a[s:s + chunk], "b": b[s:s + chunk]})
        m = E.merge(Vb, on="b").merge(Va, on=["a", "story_id"])
        m = m[m["t_b"] < m["t_a"]]
        if len(m):
            parts.append(m.groupby("row").agg(n_exp=("b", "size"), first_tb=("t_b", "min"), last_tb=("t_b", "max")))
    at_risk_user = np.zeros(C.N, bool)
    at_risk_user[a] = True
    n = len(C.v)
    n_exp = np.zeros(n, np.int32)
    d_last = np.full(n, np.nan)
    d_first = np.full(n, np.nan)
    if parts:
        ex = pd.concat(parts).groupby(level=0).agg(n_exp=("n_exp", "sum"), first_tb=("first_tb", "min"), last_tb=("last_tb", "max"))
        r = ex.index.values
        ta = C.v["vote_date"].values[r]
        n_exp[r] = ex["n_exp"].values
        d_last[r] = (ta - ex["last_tb"].values) / 3600.0
        d_first[r] = (ta - ex["first_tb"].values) / 3600.0
    at_risk_vote = np.where(ing, at_risk_user[np.maximum(C.v_gi, 0)], False)
    return n_exp, d_last, d_first, at_risk_vote, sampled, n_candidate_edges


def exposure_summary(C, n_exp, d_last, d_first, at_risk):
    v = C.v
    exposed = n_exp > 0
    res = {"n_at_risk_votes": int(at_risk.sum()),
           "exposed_share_all_votes": float(exposed.mean()),
           "exposed_share_at_risk_votes": float(exposed[at_risk].mean()) if at_risk.any() else None}
    res["by_story_bucket_at_risk"] = {nm: float(exposed[at_risk & (v["bk"].values == i)].mean()) if (at_risk & (v["bk"].values == i)).any() else None
                                      for i, nm in enumerate(BUCKET_NAMES)}
    pb = np.digitize(v["pos"].values, [2, 11, 101, 1001])
    res["by_cascade_position_at_risk"] = {nm: float(exposed[at_risk & (pb == i)].mean()) if (at_risk & (pb == i)).any() else None
                                          for i, nm in enumerate(["1st", "2-10", "11-100", "101-1000", ">1000"])}
    ab = np.digitize(v["age_h"].values, [1, 24, 168, 720])
    res["by_story_age_at_risk"] = {nm: float(exposed[at_risk & (ab == i)].mean()) if (at_risk & (ab == i)).any() else None
                                   for i, nm in enumerate(["<1h", "1-24h", "1-7d", "7-30d", ">30d"])}
    res["n_exposing_friends_among_exposed"] = qcurve(n_exp[exposed], [.5, .75, .9, .99])
    res["delay_h_from_last_exposing_vote"] = qcurve(d_last[exposed], [.1, .25, .5, .75, .9, .95, .99])
    res["delay_h_from_first_exposing_vote"] = qcurve(d_first[exposed], [.1, .25, .5, .75, .9, .95, .99])
    dl = d_last[exposed]
    res["share_exposed_with_last_exposure_within"] = {f"{h}h": float((dl <= h).mean()) for h in (0.25, 1, 6, 24, 168, 720)} if len(dl) else {}
    return res


@section("exposure", "8. SOCIAL EXPOSURE (was the voter preceded by a vote of someone they follow?)")
def s_exposure(C):
    if C.args.skip_exposure:
        log("skipped (--skip-exposure)")
        return {"skipped": True}
    out = {}
    for direction in ("followee", "follower"):
        n_exp, dl, df_, risk, sampled, n_edges = compute_exposure(C, direction)
        out[direction] = exposure_summary(C, n_exp, dl, df_, risk)
        out[direction]["edges_between_voters"] = n_edges
        out[direction]["followers_sampled_due_to_cap"] = sampled
        if direction == "followee":
            C.v["n_exp"] = n_exp
            C.v["exposed"] = n_exp > 0
            C.exposure_sampled = sampled
        r = out[direction]
        log(f"[{direction}] edges between voters={n_edges:,} (sampled={sampled}); exposed share: all votes {r['exposed_share_all_votes']:.2%}; "
            f"among votes by users who have >=1 voting friend {r['exposed_share_at_risk_votes']}")
        log("   by photo bucket:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_story_bucket_at_risk"].items()})
        log("   by cascade position:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_cascade_position_at_risk"].items()})
        log("   by photo age:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_story_age_at_risk"].items()})
        log("   #exposing friends | exposed:", qline(r["n_exposing_friends_among_exposed"]))
        log("   delay from last exposing vote (h):", qline(r["delay_h_from_last_exposing_vote"]))
    a = out["followee"]["exposed_share_at_risk_votes"]
    b = out["follower"]["exposed_share_at_risk_votes"]
    out["influence_direction_ratio_followee_over_follower"] = (a / b) if (a and b) else None
    log(f"direction check: followee/follower exposure ratio = {out['influence_direction_ratio_followee_over_follower']}  (>>1 means 'you favorite after the people you follow')")
    if getattr(C, "exposure_sampled", False):
        log("  NOTE: edges were capped -> exposure shares are exact only for the sampled followers; 'exposed_share_all_votes' is underestimated "
            "and the exp_share feature in section 11 is partly zero-filled.  Raise --max-exposure-edges for exact numbers.")
    return out


@section("adoption_vs_exposure", "9. ADOPTION vs NUMBER OF VOTING FRIENDS (sampled photos, end-of-life exposure)")
def s_adoption(C):
    rng = np.random.default_rng(C.seed)
    S = C.S
    risk = C.out_deg > 0
    K = 10
    AdjT = C.Adj.T.tocsr()          # row b -> followers of b
    groups = {"small(p00-50)": [0], "mid(p50-95)": [1, 2], "large(p95-100)": [3, 4]}
    out = {}
    nv = S["n_votes"].values
    for gname, bks in groups.items():
        pool = np.flatnonzero(np.isin(S["bk"].values, bks))
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(ADOPT_SAMPLE, len(pool)), replace=False)
        at, vo = np.zeros(K + 1), np.zeros(K + 1)
        for j in pick:
            seg = C.v_gi[C.starts[j]: C.starts[j] + nv[j]]
            idx = np.unique(seg[seg >= 0])
            if len(idx) == 0:
                continue
            vm = np.zeros(C.N, bool)
            vm[idx] = True
            k = np.bincount(AdjT[idx].indices, minlength=C.N)   # number of voting contacts per user
            kk = np.minimum(k, K)[risk]
            at += np.bincount(kk, minlength=K + 1)
            vo += np.bincount(kk[vm[risk]], minlength=K + 1)
        out[gname] = {"n_stories_sampled": int(len(pick)),
                      "by_k_voting_friends": {str(i): {"at_risk": int(at[i]), "voted": int(vo[i]), "rate": float(vo[i] / at[i]) if at[i] > 0 else None}
                                              for i in range(K + 1)}}
        rates = [out[gname]["by_k_voting_friends"][str(i)]["rate"] for i in (0, 1, 2, 3, 5, 10)]
        log(f"{gname:<16} P(user votes | k friends voted), k=0,1,2,3,5,10+: " + ", ".join("n/a" if r is None else f"{r:.2e}" for r in rates))
    log("NOTE: this mixes influence with shared interest (homophily); it is the upper bound for the 'friend lift' in the generator.")
    return out


@section("homophily", "10. HOMOPHILY & LATENT DIMENSION (do contacts favorite the same photos?)")
def s_homophily(C):
    rng = np.random.default_rng(C.seed)
    U, S, v = C.U, C.S, C.v
    uidx = U.index.get_indexer(v["voter_id"].values)
    sidx = S.index.get_indexer(v["story_id"].values)
    A = sparse.csr_matrix((np.ones(len(v), np.float32), (uidx, sidx)), shape=(len(U), len(S)))
    nvote = np.asarray(A.sum(axis=1)).ravel()
    gmap = U.index.get_indexer(C.gi_ids)
    sel = (gmap[C.aidx] >= 0) & (gmap[C.bidx] >= 0)
    ea, eb = gmap[C.aidx[sel]], gmap[C.bidx[sel]]
    keep = (nvote[ea] >= 3) & (nvote[eb] >= 3)
    ea, eb = ea[keep], eb[keep]
    out = {"friend_pairs_used": int(len(ea))}
    if len(ea) < 100:
        log("too few friend pairs among voters with >=3 votes; skipping")
        return out
    if len(ea) > C.args.homophily_pairs:
        p = rng.choice(len(ea), C.args.homophily_pairs, replace=False)
        ea, eb = ea[p], eb[p]

    def jac(x, y):
        inter = np.asarray(A[x].multiply(A[y]).sum(axis=1)).ravel()
        return inter, inter / (nvote[x] + nvote[y] - inter)

    i_f, j_f = jac(ea, eb)
    # null 1: partner replaced by a random voter of similar activity (decile-matched)
    dec = pd.qcut(pd.Series(nvote).rank(method="first"), 10, labels=False).values
    ing = np.zeros(len(U), bool)
    ing[gmap[gmap >= 0]] = True
    cand = ing & (nvote >= 3)
    eb_null = eb.copy()
    for d in range(10):
        m = dec[eb] == d
        pool = np.flatnonzero(cand & (dec == d))
        if m.any() and len(pool):
            eb_null[m] = rng.choice(pool, m.sum())
    i_m, j_m = jac(ea, eb_null)
    # null 2: partner replaced by any random in-graph voter
    pool_all = np.flatnonzero(cand)
    eb_r = rng.choice(pool_all, len(ea))
    i_r, j_r = jac(ea, eb_r)
    out.update({
        "friends": {"mean_shared_stories": float(i_f.mean()), "mean_jaccard": float(j_f.mean()), "share_ge1_shared": float((i_f >= 1).mean())},
        "activity_matched_random": {"mean_shared_stories": float(i_m.mean()), "mean_jaccard": float(j_m.mean()), "share_ge1_shared": float((i_m >= 1).mean())},
        "random_in_graph_voter": {"mean_shared_stories": float(i_r.mean()), "mean_jaccard": float(j_r.mean()), "share_ge1_shared": float((i_r >= 1).mean())},
        "homophily_ratio_jaccard_vs_matched": float(j_f.mean() / j_m.mean()) if j_m.mean() > 0 else None,
        "homophily_ratio_shared_vs_matched": float(i_f.mean() / i_m.mean()) if i_m.mean() > 0 else None,
    })
    log(f"pairs={len(ea):,}  mean shared photos: friends={i_f.mean():.2f}  activity-matched random={i_m.mean():.2f}  random={i_r.mean():.2f}")
    log(f"mean Jaccard: friends={j_f.mean():.4f}  matched={j_m.mean():.4f}  -> homophily ratio {out['homophily_ratio_jaccard_vs_matched']}")
    if not C.skip_svd:
        try:
            from scipy.sparse.linalg import svds
            rows = np.flatnonzero(nvote >= 5)
            if len(rows) > 50000:
                rows = np.sort(rng.choice(rows, 50000, replace=False))
            A5 = A[rows]
            A5 = sparse.diags(1.0 / np.sqrt(nvote[rows])) @ A5
            k = int(min(15, min(A5.shape) - 2))
            sv = np.sort(svds(A5, k=k, return_singular_vectors=False))[::-1]
            out["svd_variance_share_top_components_(unit-norm_user_rows)"] = (sv ** 2 / A5.shape[0])
            out["svd_users_used"] = int(A5.shape[0])
            log(f"SVD variance share, top components (1st is mostly 'popularity'; {A5.shape[0]:,} users):",
                np.round(out["svd_variance_share_top_components_(unit-norm_user_rows)"], 4).tolist())
        except Exception as ex:
            log(f"SVD skipped: {ex}")
    return out


def build_story_features(C, sel):
    """Early-window features for the photos at positional indices `sel` of C.S."""
    v, S, U = C.v, C.S, C.U
    Ssel = S.iloc[sel]
    ids = Ssel.index.values
    F = pd.DataFrame(index=Ssel.index)
    F["n_votes"] = Ssel["n_votes"]
    F["t_first"] = Ssel["t_first"]
    F["t_last"] = Ssel["t_last"]
    F["bucket"] = [BUCKET_NAMES[i] for i in Ssel["bk"].values]
    for col in Ssel.columns:
        if col.startswith("v_") or col in ("peak_age_d", "peak_votes_d", "peak_share", "lifetime_h"):
            F[col] = Ssel[col]
    first = v.iloc[C.starts[sel]]
    F["first_voter_id"] = first["voter_id"].values
    F["first_voter_indeg"] = first["indeg"].values
    F["first_voter_outdeg"] = first["outdeg"].values
    F["first_voter_nvotes"] = U["n_votes"].reindex(first["voter_id"].values).values
    dt = pd.to_datetime(Ssel["t_first"].values, unit="s", utc=True)
    F["hour"] = np.asarray(dt.hour)
    F["dow"] = np.asarray(dt.weekday)
    vs = v[np.isin(v["story_id"].values, ids)]
    for w in FEAT_W:
        ws = f"{w:g}"
        sub = vs[vs["age_h"] <= w]
        g = sub.groupby("story_id")
        F[f"n_{ws}"] = Ssel[f"v_{ws}h"]
        F[f"fol_sum_{ws}"] = g["indeg"].sum().reindex(F.index).fillna(0)
        F[f"fol_mean_{ws}"] = g["indeg"].mean().reindex(F.index)
        F[f"fol_max_{ws}"] = g["indeg"].max().reindex(F.index)
        F[f"in_graph_{ws}"] = g["in_graph"].mean().reindex(F.index)
        F[f"hub_{ws}"] = g["is_hub"].sum().reindex(F.index).fillna(0)
        F[f"exp_share_{ws}"] = g["exposed"].mean().reindex(F.index)
        F[f"gap_mean_{ws}"] = g["gap_s"].mean().reindex(F.index)
        F[f"gap_std_{ws}"] = g["gap_s"].std().reindex(F.index)
        half = Ssel[f"v_{w / 2:g}h"]
        F[f"accel_{ws}"] = (Ssel[f"v_{ws}h"] - half) / half.clip(lower=1)
    vgi = C.v_gi
    starts = C.starts[sel]
    for w in EARLY_W:
        ws = f"{w:g}"
        nw = Ssel[f"v_{ws}h"].values
        dens, edges_n, conn = np.full(len(sel), np.nan), np.zeros(len(sel)), np.full(len(sel), np.nan)
        for k in range(len(sel)):
            seg = vgi[starts[k]: starts[k] + nw[k]]
            idx = np.unique(seg[seg >= 0][:300])
            m = len(idx)
            if m >= 2:
                sg = C.Adj[idx][:, idx]
                edges_n[k] = sg.nnz
                dens[k] = sg.nnz / (m * (m - 1))
                deg = np.asarray(sg.sum(axis=0)).ravel() + np.asarray(sg.sum(axis=1)).ravel()
                conn[k] = float((deg > 0).mean())
        F[f"clust_dens_{ws}"] = dens
        F[f"clust_edges_{ws}"] = edges_n
        F[f"clust_conn_{ws}"] = conn
    return F


@section("early_signal", "11. EARLY-SIGNAL STRENGTH (what each information layer is worth on REAL data; sampled photos)")
def s_early(C):
    rng = np.random.default_rng(C.seed)
    S = C.S
    eligible = np.flatnonzero(S["n_votes"].values >= C.args.early_min_votes)
    n_sample = min(C.args.early_sample, len(eligible))
    if n_sample < 100:
        log(f"too few photos with >= {C.args.early_min_votes} votes ({len(eligible)}); skipping")
        return {"skipped": True, "eligible": int(len(eligible))}
    sel = np.sort(rng.choice(eligible, n_sample, replace=False))
    log(f"population: photos with >= {C.args.early_min_votes} votes ({len(eligible):,}); sampled {n_sample:,}")
    F = build_story_features(C, sel)
    C.F = F
    n = F["n_votes"].values.astype(float)
    cand = {"top50": n >= np.quantile(n, .5), "top20": n >= np.quantile(n, .8), "ge10": n >= 10, "ge100": n >= 100}
    labels = {k: y for k, y in cand.items() if 0.02 <= y.mean() <= 0.98}
    out = {"population_min_votes": int(C.args.early_min_votes), "n_population": int(len(eligible)), "n_sampled": int(n_sample),
           "label_positive_rate": {k: float(y.mean()) for k, y in labels.items()},
           "majority_class_accuracy": {k: float(max(y.mean(), 1 - y.mean())) for k, y in labels.items()},
           "labels_dropped_(positive_rate_outside_2-98%)": [k for k in cand if k not in labels]}
    log("label positive rates:", out["label_positive_rate"], " majority-class accuracy:", out["majority_class_accuracy"],
        " dropped:", out["labels_dropped_(positive_rate_outside_2-98%)"])
    if not labels:
        log("no usable label; skipping")
        return out

    # (a) single-feature separability
    feat_names = ["n", "fol_sum", "fol_mean", "fol_max", "in_graph", "hub", "exp_share", "accel", "gap_mean", "gap_std", "clust_dens", "clust_edges", "clust_conn"]
    single = {}
    rows = []
    for w in EARLY_W:
        ws = f"{w:g}"
        for fn in feat_names:
            col = f"{fn}_{ws}"
            if col not in F:
                continue
            r = {"feature": f"{fn}@{ws}h"}
            for lb, y in labels.items():
                auc, bal = auc_bal(F[col].values, y)
                r[f"auc_{lb}"], r[f"bal_{lb}"] = auc, bal
                single[f"{col}|{lb}"] = {"auc": auc, "best_threshold_balanced_acc": bal}
            rows.append(r)
    for col in ("first_voter_indeg", "first_voter_nvotes", "hour", "dow"):
        r = {"feature": col}
        for lb, y in labels.items():
            auc, bal = auc_bal(F[col].values, y)
            r[f"auc_{lb}"], r[f"bal_{lb}"] = auc, bal
            single[f"{col}|{lb}"] = {"auc": auc, "best_threshold_balanced_acc": bal}
        rows.append(r)
    out["single_feature"] = single
    log("single-feature AUC / best-threshold balanced accuracy (0.5 = no signal):")
    log(pd.DataFrame(rows).set_index("feature").round(3).to_string())

    # (b) Spearman of early counts with final
    out["spearman_early_votes_vs_final"] = {f"{w:g}h": spearman(F[f"n_{w:g}"].values, n) for w in FEAT_W}
    out["spearman_early_votes_vs_remaining"] = {f"{w:g}h": spearman(F[f"n_{w:g}"].values, n - F[f"n_{w:g}"].values) for w in FEAT_W}
    log("Spearman(early votes, final votes):", {k: round(x, 3) for k, x in out["spearman_early_votes_vs_final"].items()})
    log("Spearman(early votes, remaining votes):", {k: round(x, 3) for k, x in out["spearman_early_votes_vs_remaining"].items()})

    # (c) cumulative-layer gradient boosting (counts -> aggregates -> structure -> temporal)
    if C.skip_cv:
        log("CV skipped (--skip-cv).")
        return out
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold
    except Exception as ex:
        log(f"sklearn not available, CV skipped: {ex}")
        return out

    def sets(ws):
        A = [f"n_{ws}"]
        B = A + [f"fol_sum_{ws}", f"fol_mean_{ws}", f"fol_max_{ws}", f"in_graph_{ws}", "first_voter_indeg", "hour", "dow"]
        Cc = B + [f"clust_dens_{ws}", f"clust_edges_{ws}", f"clust_conn_{ws}", f"hub_{ws}"]
        D = Cc + [f"exp_share_{ws}", f"accel_{ws}", f"gap_mean_{ws}", f"gap_std_{ws}"]
        return OrderedDict([("1_counts_only", A), ("2_+aggregates(XGB-like)", B), ("3_+graph_structure", Cc), ("4_+timing/exposure", D)])

    def mk():
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, random_state=C.seed)

    order = np.argsort(F["t_first"].values, kind="mergesort")
    ntr = int(0.7 * len(F))
    cv = {}
    for lb, y in labels.items():
        if y.sum() < 10 or (~y).sum() < 10:
            continue
        cv[lb] = {}
        for w in EARLY_W:
            ws = f"{w:g}"
            cv[lb][ws] = {}
            for sname, cols in sets(ws).items():
                X = F[cols].values.astype(float)
                accs, bals = [], []
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.seed).split(X, y):
                    m = mk().fit(X[tr], y[tr])
                    p = m.predict(X[te])
                    accs.append(accuracy_score(y[te], p))
                    bals.append(balanced_accuracy_score(y[te], p))
                tr, te = order[:ntr], order[ntr:]
                p = mk().fit(X[tr], y[tr]).predict(X[te])
                cv[lb][ws][sname] = {"cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
                                     "cv_bal_acc_mean": float(np.mean(bals)),
                                     "temporal_holdout_acc": float(accuracy_score(y[te], p)),
                                     "temporal_holdout_bal_acc": float(balanced_accuracy_score(y[te], p))}
    out["layered_gradient_boosting"] = cv
    for lb in cv:
        log(f"label={lb} (majority-class acc {out['majority_class_accuracy'][lb]:.3f}) -- 5-fold CV accuracy (temporal-holdout acc) by window and feature layer:")
        tbl = pd.DataFrame({f"{ws}h": {s: f"{r['cv_acc_mean']:.3f} ({r['temporal_holdout_acc']:.3f})" for s, r in cv[lb][ws].items()} for ws in cv[lb]})
        log(tbl.to_string())
    return out


@section("save", "12. SAVING")
def s_save(C):
    od = C.out_dir
    PROFILE["meta"] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": "flickr", "data_dir": C.data_dir, "seed": C.seed,
                       "entity_names": {"story": "photo", "vote": "favorite marking", "friend_edge": "user follows contact"},
                       "buckets": BUCKET_NAMES, "bucket_edges_pct": BUCKET_EDGES, "windows_h": WINDOWS_H,
                       "early_windows_h": EARLY_W, "args": {k: v for k, v in vars(C.args).items()}}
    C.S.reset_index().to_csv(os.path.join(od, "story_table.csv"), index=False)
    C.U.reset_index().to_csv(os.path.join(od, "user_table.csv"), index=False)
    if hasattr(C, "F"):
        C.F.reset_index().to_csv(os.path.join(od, "story_features_early_sample.csv"), index=False)
    with open(os.path.join(od, "profile.json"), "w") as fh:
        json.dump(jsonable(PROFILE), fh, indent=1)
    log(f"wrote story_table.csv ({len(C.S):,} rows), user_table.csv ({len(C.U):,} rows), "
        f"{'story_features_early_sample.csv, ' if hasattr(C, 'F') else ''}profile.json in {od}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Extract the statistics needed to synthesise Flickr-like data.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--votes", nargs="*", default=None, help="explicit interaction (favorite) file(s)")
    ap.add_argument("--friends", default=None, help="explicit graph file")
    ap.add_argument("--vote-cols", default=None, help="0-based file column idx: time,user,photo   e.g. 2,0,1")
    ap.add_argument("--friend-cols", default=None, help="0-based idx: user,contact[,date[,mutual]]  (-1 = absent)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-svd", action="store_true")
    ap.add_argument("--skip-exposure", action="store_true")
    ap.add_argument("--sample-items", type=int, default=0, help="keep only N random photos (0 = all)")
    ap.add_argument("--min-item-votes", type=int, default=1, help="drop photos with fewer votes than this")
    ap.add_argument("--early-sample", type=int, default=20000)
    ap.add_argument("--early-min-votes", type=int, default=2)
    ap.add_argument("--max-exposure-edges", type=int, default=8_000_000)
    ap.add_argument("--homophily-pairs", type=int, default=150000)
    args, _ = ap.parse_known_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.data_dir.rstrip("/"))), "flickr_profile")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.abspath("flickr_profile")
        os.makedirs(out_dir, exist_ok=True)

    t00 = time.time()
    C = SimpleNamespace(data_dir=args.data_dir, out_dir=out_dir, seed=args.seed, skip_cv=args.skip_cv,
                        skip_svd=args.skip_svd, args=args)
    hdr("0. LOADING FILES (columns identified by value checks; mapping printed below)")
    log(f"args: {vars(args)}")
    vpaths, fpath = find_files(args)
    log(f"  interaction file(s): {vpaths}")
    log(f"  graph file: {fpath}")
    C.e, C.finfo = load_friends(fpath, args.friend_cols)
    C.has_mutual = bool(C.finfo["has_mutual_column"])
    C.has_fdate = bool(C.finfo["has_date_column"])
    C.gids = np.unique(np.r_[C.e["user_id"].values, C.e["friend_id"].values].astype(np.int64))
    C.v, C.vinfo = load_votes(vpaths, C.gids, args)
    prep(C)

    s_overview(C)
    s_popularity(C)
    s_users(C)
    s_temporal(C)
    s_calendar(C)
    s_graph(C)
    s_coverage(C)
    s_exposure(C)
    s_adoption(C)
    s_homophily(C)
    s_early(C)
    s_save(C)

    hdr("DONE")
    log(f"total time {time.time() - t00:.0f}s. Outputs in: {out_dir}")
    log("Paste profile_report.txt back to Claude; keep profile.json for the generator.")
    with open(os.path.join(out_dir, "profile_report.txt"), "w") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    main()