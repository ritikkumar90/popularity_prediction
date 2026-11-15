#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_digg_profile.py
=======================
Reads the REAL Digg-2009 files (headerless CSVs) and extracts every statistic a
synthetic-data generator needs to imitate them.

Input files (auto-found in --data-dir):
    digg_votes1.csv    vote_date, voter_id, story_id
    digg_friends.csv   mutual, friend_date, user_id, friend_id     (user_id follows friend_id)

Outputs (in --out-dir):
    profile_report.txt   human-readable report  <- paste this back to Claude
    profile.json         machine-readable profile (the generator reads this)
    story_table.csv      one row per story: final votes, early-window features, labels
    user_table.csv       one row per voter: activity, degrees, early-adopter score

Usage (Colab):
    !python extract_digg_profile.py
    !python extract_digg_profile.py --data-dir "/content/drive/MyDrive/.../raw/digg" --out-dir digg_profile
Options:
    --skip-cv     skip the cross-validated early-signal models (fastest)
    --skip-svd    skip the latent-dimension SVD
    --votes F...  explicit vote file(s)      --friends F  explicit friends file

Design notes
  * Columns are mapped by NAME/VALUE-RANGE checks, never by "whatever looks right".
  * Every section runs inside try/except, so one failure does not kill the rest.
  * Nothing is assumed about the real data: every number is measured.
"""
import argparse
import gzip
import json
import os
import sys
import time
import traceback
import warnings
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse, stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
DEFAULT_DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw/digg"
VOTE_COLS = ["vote_date", "voter_id", "story_id"]
FRIEND_COLS = ["mutual", "friend_date", "user_id", "friend_id"]

WINDOWS_H = [0.25, 0.5, 1, 2, 3, 4, 6, 12, 24, 48, 72, 168, 336, 720]   # hours after 1st vote
FEAT_W = [0.5, 1, 2, 4, 6]            # windows for per-story early features
EARLY_W = [1, 2, 4]                   # windows used for the early-signal models
KTH = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
QS = [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, .995, .999, 1.0]
BUCKET_EDGES = [0, .5, .8, .95, .99, 1.0]
BUCKET_NAMES = ["p00-50", "p50-80", "p80-95", "p95-99", "p99-100"]

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


def sniff(path):
    with _open(path) as fh:
        first = fh.readline().strip()
    sep = "," if "," in first else None
    parts = first.split(",") if sep else first.split()

    def isnum(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    header = not all(isnum(p) for p in parts)
    return (sep or r"\s+"), header, len(parts)


def read_raw(path):
    sep, header, ncols = sniff(path)
    df = pd.read_csv(path, sep=sep, header=0 if header else None)
    df.columns = range(df.shape[1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(np.int64)
    log(f"  read {os.path.basename(path)}: rows={len(df):,} cols={df.shape[1]} sep={sep!r} header_row_found={header}")
    return df


def map_vote_cols(df):
    """Find (vote_date, voter_id, story_id) by value ranges, falling back to file order."""
    med = {c: float(df[c].median()) for c in df.columns}
    ts = [c for c in df.columns if 9e8 < med[c] < 1.5e9]
    if len(ts) == 1:
        others = [c for c in df.columns if c != ts[0]]
        story = min(others, key=lambda c: df[c].nunique())
        voter = [c for c in others if c != story][0]
        return {"vote_date": ts[0], "voter_id": voter, "story_id": story}
    log("  WARNING: could not identify the timestamp column by range; assuming file order vote_date,voter_id,story_id")
    return {"vote_date": 0, "voter_id": 1, "story_id": 2}


def map_friend_cols(df):
    cols = list(df.columns)
    mut = [c for c in cols if set(pd.unique(df[c].iloc[:200000])) <= {0, 1}]
    ts = [c for c in cols if 9e8 < df[c].median() < 1.5e9]
    if len(mut) == 1 and len(ts) == 1:
        rest = [c for c in cols if c not in (mut[0], ts[0])]
        return {"mutual": mut[0], "friend_date": ts[0], "user_id": rest[0], "friend_id": rest[1]}
    log("  WARNING: could not identify mutual/date columns by value; assuming file order mutual,friend_date,user_id,friend_id")
    return {"mutual": 0, "friend_date": 1, "user_id": 2, "friend_id": 3}


def find_files(args):
    d = args.data_dir
    if args.votes:
        vp = args.votes
    else:
        vp = [os.path.join(d, n) for n in sorted(os.listdir(d))
              if "vote" in n.lower() and n.endswith((".csv", ".txt", ".gz"))]
    if args.friends:
        fp = args.friends
    else:
        c = [os.path.join(d, n) for n in sorted(os.listdir(d))
             if "friend" in n.lower() and n.endswith((".csv", ".txt", ".gz"))]
        fp = c[0] if c else None
    if not vp or not fp:
        raise FileNotFoundError(f"Need a votes file and a friends file in {d} (found votes={vp}, friends={fp})")
    return vp, fp


def load_votes(paths):
    frames = []
    for p in paths:
        raw = read_raw(p)
        if raw.shape[1] < 3:
            raise ValueError(f"{p} has {raw.shape[1]} columns, expected 3")
        raw = raw.iloc[:, :3]
        m = map_vote_cols(raw)
        expected = {"vote_date": 0, "voter_id": 1, "story_id": 2}
        log(f"  vote column mapping (file col index): {m}" + ("" if m == expected else "   <-- differs from expected file order!"))
        frames.append(pd.DataFrame({k: raw[m[k]].values for k in VOTE_COLS}))
    v = pd.concat(frames, ignore_index=True)
    info = {"rows_raw": int(len(v)), "files": [os.path.basename(p) for p in paths]}
    n0 = len(v)
    v = v.drop_duplicates()
    info["exact_duplicate_rows_dropped"] = int(n0 - len(v))
    n0 = len(v)
    v = v.sort_values(["vote_date", "voter_id"], kind="mergesort").drop_duplicates(["voter_id", "story_id"], keep="first")
    info["duplicate_user_story_pairs_dropped"] = int(n0 - len(v))
    v = v.reset_index(drop=True)
    for c in ("voter_id", "story_id"):
        if v[c].max() < 2 ** 31:
            v[c] = v[c].astype(np.int32)
    return v, info


def load_friends(path):
    raw = read_raw(path)
    if raw.shape[1] < 4:
        raise ValueError(f"{path} has {raw.shape[1]} columns, expected 4")
    raw = raw.iloc[:, :4]
    m = map_friend_cols(raw)
    expected = {"mutual": 0, "friend_date": 1, "user_id": 2, "friend_id": 3}
    log(f"  friend column mapping (file col index): {m}" + ("" if m == expected else "   <-- differs from expected file order!"))
    log("  NOTE: user_id -> friend_id direction (user follows friend) is taken from the dataset description, not detectable from values.")
    e = pd.DataFrame({k: raw[m[k]].values for k in FRIEND_COLS})
    info = {"rows_raw": int(len(e)), "file": os.path.basename(path)}
    n0 = len(e)
    e = e.drop_duplicates()
    info["exact_duplicate_rows_dropped"] = int(n0 - len(e))
    n0 = len(e)
    e = e.drop_duplicates(["user_id", "friend_id"], keep="first")
    info["duplicate_directed_pairs_dropped"] = int(n0 - len(e))
    n0 = len(e)
    e = e[e["user_id"] != e["friend_id"]]
    info["self_loops_dropped"] = int(n0 - len(e))
    info["rows_with_nonpositive_date"] = int((e["friend_date"] <= 0).sum())
    return e.reset_index(drop=True), info


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
    for w in WINDOWS_H:
        S[f"v_{w:g}h"] = (v["age_h"] <= w).groupby(v["story_id"]).sum().reindex(S.index).astype(int)
    C.S = S
    C.starts = np.r_[0, np.cumsum(S["n_votes"].values)][:-1]

    U = v.groupby("voter_id").agg(n_votes=("story_id", "size"), t_first=("vote_date", "min"),
                                  t_last=("vote_date", "max"), mean_rank_frac=("rank_frac", "mean"),
                                  mean_log_story_n=("log_story_n", "mean"))
    C.U = U

    e = C.e
    u = e["user_id"].values.astype(np.int64)
    f = e["friend_id"].values.astype(np.int64)
    ids = np.unique(np.r_[u, f])
    gi = pd.Index(ids)
    C.gi_ids = ids
    C.aidx = gi.get_indexer(u)
    C.bidx = gi.get_indexer(f)
    C.N = len(ids)
    C.Adj = sparse.csr_matrix((np.ones(len(u), dtype=np.int8), (C.aidx, C.bidx)), shape=(C.N, C.N))
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
    out = {
        "votes": int(len(v)), "stories": int(len(S)), "voters": int(len(U)),
        "vote_time_min": t0, "vote_time_max": t1,
        "vote_time_min_iso": str(pd.to_datetime(t0, unit="s")), "vote_time_max_iso": str(pd.to_datetime(t1, unit="s")),
        "span_days": (t1 - t0) / 86400.0,
        "votes_info": C.vinfo, "friends_info": C.finfo,
        "edges_after_cleaning": int(len(e)), "graph_users": int(C.N),
    }
    log(f"votes={len(v):,}  stories={len(S):,}  voters={len(U):,}  edges={len(e):,}  graph users={C.N:,}")
    log(f"time range {out['vote_time_min_iso']} -> {out['vote_time_max_iso']}  ({out['span_days']:.1f} days)")
    log(f"votes cleaning: {C.vinfo}")
    log(f"friends cleaning: {C.finfo}")
    checks = []
    checks.append(("stories ~3,553", 1000 <= len(S) <= 10000))
    checks.append(("voters ~139,409", 50000 <= len(U) <= 400000))
    checks.append(("votes ~3.0M", 1.5e6 <= len(v) <= 6e6))
    checks.append(("timestamps look like 2009 unix seconds", 1.2e9 < t0 < 1.3e9 and t1 < 1.3e9))
    checks.append(("story ids are not a flag (>100 distinct)", len(S) > 100))
    for name, ok in checks:
        log(f"  [{'PASS' if ok else 'WARN'}] {name}")
    out["sanity_checks"] = {n: bool(ok) for n, ok in checks}
    return out


@section("story_popularity", "2. STORY POPULARITY (final votes per story)")
def s_popularity(C):
    n = C.S["n_votes"].values.astype(float)
    out = {
        "n_stories": int(len(n)), "n_votes_total": int(n.sum()), "mean": float(n.mean()), "std": float(n.std()),
        "skew": float(stats.skew(n)), "skew_log": float(stats.skew(np.log(n))),
        "log_mu": float(np.log(n).mean()), "log_sigma": float(np.log(n).std()),
        "gini": gini(n), "top1pct_share": top_share(n, .01), "top10pct_share": top_share(n, .10),
        "top20pct_share": top_share(n, .20),
        "hill_alpha_xmin_p50": hill(n, np.quantile(n, .5)), "hill_alpha_xmin_p80": hill(n, np.quantile(n, .8)),
        "hill_alpha_xmin_p95": hill(n, np.quantile(n, .95)),
        "quantiles": qcurve(n),
    }
    labels = {}
    for nm, q in [("top50", .5), ("top20", .8), ("top10", .9)]:
        thr = float(np.quantile(n, q))
        labels[nm] = {"threshold_votes": thr, "positive_rate": float((n >= thr).mean())}
    out["label_rules"] = labels
    log("votes/story quantiles:", qline(out["quantiles"]))
    log(f"mean={out['mean']:.1f} std={out['std']:.1f} log-normal fit mu={out['log_mu']:.3f} sigma={out['log_sigma']:.3f}  gini={out['gini']:.3f}")
    log(f"top1%/10%/20% of stories hold {out['top1pct_share']:.1%} / {out['top10pct_share']:.1%} / {out['top20pct_share']:.1%} of votes")
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
        "top1pct_share": top_share(n, .01), "top10pct_share": top_share(n, .10),
        "hill_alpha_xmin_p50": hill(n, max(1, np.quantile(n, .5))), "hill_alpha_xmin_p90": hill(n, max(1, np.quantile(n, .9))),
        "active_span_h_quantiles_(users>=2votes)": qcurve(span, [.1, .25, .5, .75, .9]),
        "share_users_all_votes_within_24h_(users>=2votes)": float((span <= 24).mean()) if len(span) else None,
        "spearman_nvotes_vs_mean_rank_frac": spearman(n, U["mean_rank_frac"].values),
        "spearman_nvotes_vs_mean_log_story_size": spearman(n, U["mean_log_story_n"].values),
        "mean_rank_frac_overall": float(U["mean_rank_frac"].mean()),
    }
    log("votes/user quantiles:", qline(out["quantiles"]))
    log(f"mean={out['mean']:.2f} gini={out['gini']:.3f} one-vote users={out['share_one_vote']:.1%} (>=10 votes {out['share_ge_10']:.1%}, >=100 {out['share_ge_100']:.2%})")
    log(f"top1% of users cast {out['top1pct_share']:.1%} of votes; Hill alpha (p50/p90) {out['hill_alpha_xmin_p50']:.2f} / {out['hill_alpha_xmin_p90']:.2f}")
    log(f"active span (h), users with >=2 votes: {qline(out['active_span_h_quantiles_(users>=2votes)'])}")
    log(f"heavy voters vote earlier in cascades? Spearman(n_votes, mean cascade position)={out['spearman_nvotes_vs_mean_rank_frac']:.3f}")
    log(f"heavy voters pick bigger stories? Spearman(n_votes, mean log story size)={out['spearman_nvotes_vs_mean_log_story_size']:.3f}")
    return out


@section("temporal", "4. TEMPORAL DYNAMICS (vote time vs. story age)")
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
    show = [0.25, 1, 2, 4, 6, 12, 24, 72]
    log("MEDIAN votes accumulated by age (rows=final-size bucket, cols=hours since first vote):")
    log(pd.DataFrame({f"{w:g}h": [casc[nm][f"{w:g}"]["abs_median"] for nm in names] for w in show}, index=names).round(1).to_string())
    log("MEDIAN share of final votes reached by age:")
    log(pd.DataFrame({f"{w:g}h": [casc[nm][f"{w:g}"]["share_median"] for nm in names] for w in show}, index=names).round(3).to_string())

    # --- time to k-th vote
    kth = {}
    for k in KTH:
        m = v["pos"].values == k
        ages, bks = v["age_h"].values[m], v["bk"].values[m]
        row = {"reached_share_all": float(m.sum() / len(S)), "age_h_all": qcurve(ages, [.1, .5, .9])}
        for bi, nm in enumerate(BUCKET_NAMES):
            a = ages[bks == bi]
            row[nm] = {"reached_share": float(len(a) / max(1, (S["bk"].values == bi).sum())), "age_h": qcurve(a, [.1, .5, .9])}
        kth[str(k)] = row
    out["time_to_kth_vote"] = kth
    log("time to k-th vote (hours, median [p10-p90]) over stories that reach k:")
    for k in KTH:
        r = kth[str(k)]["age_h_all"]
        if r:
            log(f"  k={k:<5} reached by {kth[str(k)]['reached_share_all']:.1%}  median={r['0.5']:.2f}h  [{r['0.1']:.2f} - {r['0.9']:.2f}]")

    # --- time to reach X% of the story's final votes
    tts = {}
    for qf in [.1, .25, .5, .75, .9]:
        target = np.ceil(qf * v["n_story"].values)
        m = v["pos"].values == target
        ages, bks = v["age_h"].values[m], v["bk"].values[m]
        ages_story = pd.Series(ages, index=v["story_id"].values[m]).groupby(level=0).min()
        row = {"all": qcurve(ages_story.values, [.1, .5, .9])}
        bk_story = S["bk"].reindex(ages_story.index).values
        for bi, nm in enumerate(BUCKET_NAMES):
            row[nm] = qcurve(ages_story.values[bk_story == bi], [.1, .5, .9])
        tts[f"{qf:g}"] = row
    out["age_h_to_reach_share_of_final"] = tts
    log("age (h) at which a story reaches X% of its final votes (median):")
    log(pd.DataFrame({f"{q}": [tts[q][nm].get("0.5", np.nan) for nm in names if nm in tts[q]] for q in tts},
                     index=[nm for nm in names if nm in tts["0.5"]]).round(1).to_string())

    # --- vote-rate profile vs age
    edges_log = np.r_[0, np.geomspace(0.05, 1000, 41)]
    edges_h = np.arange(0, 97, 1.0)
    prof = {"edges_log_h": edges_log, "edges_hourly_h": edges_h}
    decay = {}
    ages_all = v["age_h"].values
    for nm in names:
        m = np.ones(len(v), bool) if nm == "all" else (v["bk"].values == BUCKET_NAMES.index(nm))
        a = ages_all[m]
        cl, _ = np.histogram(a, bins=edges_log)
        ch, _ = np.histogram(a, bins=edges_h)
        sl = cl / max(1, len(a))
        prof[nm] = {"share_per_log_bin": sl, "rate_per_h_log_bin": sl / np.diff(edges_log),
                    "share_per_hour_0_96h": ch / max(1, len(a))}
        cen = np.sqrt(np.maximum(edges_log[:-1], 1e-9) * edges_log[1:])
        sel = (edges_log[:-1] >= 24) & (edges_log[1:] <= 600) & (sl > 0)
        if sel.sum() >= 4:
            slope = np.polyfit(np.log(cen[sel]), np.log(sl[sel] / np.diff(edges_log)[sel]), 1)[0]
            decay[nm] = float(slope)
    out["age_profile"] = prof
    out["late_decay_loglog_slope_24h_to_600h"] = decay
    log("late-life decay: log-log slope of vote rate vs age (24h-600h):", {k: round(x, 2) for k, x in decay.items()})

    # --- peak hour
    ab = np.floor(v["age_h"].values).astype(np.int64)
    cnt = pd.DataFrame({"story_id": v["story_id"].values, "ab": ab}).groupby(["story_id", "ab"]).size().reset_index(name="c")
    cnt = cnt.sort_values(["story_id", "c", "ab"], ascending=[True, False, True]).drop_duplicates("story_id").set_index("story_id")
    S["peak_age_h"] = cnt["ab"].reindex(S.index).values
    S["peak_votes_h"] = cnt["c"].reindex(S.index).values
    S["peak_share"] = S["peak_votes_h"] / S["n_votes"]
    pk = {}
    for nm in names:
        sub = sub_of(nm)
        pk[nm] = {"peak_age_h": qcurve(sub["peak_age_h"].values, [.1, .5, .9]),
                  "peak_votes_per_h": qcurve(sub["peak_votes_h"].values, [.1, .5, .9]),
                  "peak_share_of_final": qcurve(sub["peak_share"].values, [.1, .5, .9])}
    out["peak_hour"] = pk
    log("peak vote-hour age (h, median) by bucket:", {nm: round(pk[nm]["peak_age_h"].get("0.5", np.nan), 1) for nm in names})
    log("peak-hour share of final votes (median) by bucket:", {nm: round(pk[nm]["peak_share_of_final"].get("0.5", np.nan), 3) for nm in names})

    # --- lifetime / censoring / submission-time distribution
    t_end, t_start = int(v["vote_date"].max()), int(v["vote_date"].min())
    left_h = (t_end - S["t_first"].values) / 3600.0
    out["lifetime_h"] = {nm: qcurve(sub_of(nm)["lifetime_h"].values, [.1, .5, .9]) for nm in names}
    out["share_stories_first_vote_within_48h_of_end"] = float((left_h < 48).mean())
    out["share_stories_first_vote_within_168h_of_end"] = float((left_h < 168).mean())
    out["story_first_vote_per_day_since_start"] = np.bincount(((S["t_first"].values - t_start) // 86400).astype(int))
    log(f"lifetime (h) median by bucket: { {nm: round(out['lifetime_h'][nm].get('0.5', np.nan), 0) for nm in names} }")
    log(f"stories first voted <48h before dataset end (right-censored): {out['share_stories_first_vote_within_48h_of_end']:.1%}")
    log("stories first-voted per day since start:", out["story_first_vote_per_day_since_start"].tolist())
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
    log("votes per day since start:", out["votes_per_day_since_start"].tolist())
    log(f"eta^2 (first-vote hour / weekday vs log final votes): {out['eta2_first_vote_hour_vs_log_final']:.3f} / {out['eta2_first_vote_weekday_vs_log_final']:.3f}")

    tie = v["tie"].values
    ab = np.digitize(v["age_h"].values, [1, 24])
    out["tie_share_votes"] = float(tie.mean())
    out["tie_share_by_age"] = {n: float(tie[ab == i].mean()) if (ab == i).any() else None for i, n in enumerate(["<1h", "1-24h", ">24h"])}
    out["tie_share_by_bucket"] = {nm: float(tie[v["bk"].values == i].mean()) for i, nm in enumerate(BUCKET_NAMES)}
    gs = v.groupby(["story_id", "vote_date"]).size().values
    out["same_second_group_size_share_of_groups"] = {k: float(((gs == k) if k < 5 else (gs >= 5)).mean()) for k in range(1, 6)}
    t2 = v.loc[v["pos"] == 2].set_index("story_id")["vote_date"]
    out["share_stories_first_two_votes_same_second"] = float((t2.reindex(S.index).values == S["t_first"].values).mean())
    log(f"share of votes sharing a (story, second) with another vote: {out['tie_share_votes']:.1%}")
    log("  by age:", {k: round(x, 3) if x is not None else None for k, x in out["tie_share_by_age"].items()},
        " by bucket:", {k: round(x, 3) for k, x in out["tie_share_by_bucket"].items()})
    log("  same-second group sizes (share of groups; 5 = 5+):", {k: round(x, 3) for k, x in out["same_second_group_size_share_of_groups"].items()})
    log(f"  stories whose first two votes share a second: {out['share_stories_first_two_votes_same_second']:.1%}")
    return out


@section("friend_graph", "6. FRIEND GRAPH (edge user_id -> friend_id = 'user follows friend')")
def s_graph(C):
    e = C.e
    u, f = e["user_id"].values.astype(np.int64), e["friend_id"].values.astype(np.int64)
    key, rkey = (u << 32) | f, (f << 32) | u
    has_rev = np.isin(rkey, key)
    mut = e["mutual"].values == 1
    out = {
        "edges": int(len(e)), "graph_users": int(C.N),
        "reciprocity_edges_with_reverse": float(has_rev.mean()),
        "mutual_flag_share": float(mut.mean()),
        "mutual_rows_with_reverse_present": float(has_rev[mut].mean()) if mut.any() else None,
        "nonmutual_rows_with_reverse_present": float(has_rev[~mut].mean()) if (~mut).any() else None,
        "mutual_flag_agrees_with_reverse_presence": float((mut == has_rev).mean()),
    }
    # edge count if both directions of mutual pairs were collapsed
    out["edges_if_mutual_pairs_collapsed"] = int((~has_rev).sum() + has_rev.sum() // 2)
    out["mutual_edges_listed_in_both_directions"] = bool(out["mutual_rows_with_reverse_present"] is not None and out["mutual_rows_with_reverse_present"] > 0.95)
    # date relation inside reciprocal pairs
    mm = mut & has_rev
    if mm.any():
        sdate = pd.Series(e["friend_date"].values, index=key)
        rd = sdate.reindex(rkey[mm]).values
        d = (e["friend_date"].values[mm] - rd) / 86400.0
        out["mutual_pair_date_gap_days_abs"] = qcurve(np.abs(d), [.1, .25, .5, .75, .9, .99])
        out["mutual_pair_share_identical_date"] = float((d == 0).mean())
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
    ncomp, lab = sparse.csgraph.connected_components(C.Adj, directed=False)
    sz = np.bincount(lab)
    out["weak_components"] = int(ncomp)
    out["largest_component_share"] = float(sz.max() / C.N)
    # edge creation dates
    d = e["friend_date"].values.astype(np.int64)
    out["friend_date_quantiles_iso"] = {f"{q:g}": str(pd.to_datetime(np.quantile(d, q), unit="s")) for q in [0, .1, .25, .5, .75, .9, 1]}
    mo = pd.to_datetime(d, unit="s").to_period("M").astype(str)
    out["edges_by_month"] = mo.value_counts().sort_index().to_dict()
    t_start, t_end = int(C.v["vote_date"].min()), int(C.v["vote_date"].max())
    tf = C.S["t_first"].values
    cut = {"vote_window_start": t_start, "vote_window_end": t_end}
    for q in (.5, .7, .8):
        cut[f"story_first_vote_p{int(q * 100)}"] = int(np.quantile(tf, q))
    out["share_edges_created_after_cutoff"] = {k: float((d > t).mean()) for k, t in cut.items()}
    inwin = d[(d >= t_start) & (d <= t_end)]
    out["edges_created_per_day_inside_vote_window"] = np.bincount(((inwin - t_start) // 86400).astype(int))
    log(f"edges={out['edges']:,} (collapsed-mutual count would be {out['edges_if_mutual_pairs_collapsed']:,})  users={C.N:,}")
    log(f"reciprocity={out['reciprocity_edges_with_reverse']:.1%}; mutual flag share={out['mutual_flag_share']:.1%}; "
        f"mutual rows with reverse present={out['mutual_rows_with_reverse_present']}; non-mutual rows with reverse={out['nonmutual_rows_with_reverse_present']}")
    log(f"  -> mutual links listed in both directions: {out['mutual_edges_listed_in_both_directions']}")
    for nm in ("out_degree_(people_followed)", "in_degree_(followers)"):
        o = out[nm]
        log(f"{nm}: zero-share={o['share_zero_among_graph_users']:.1%} mean={o['mean_all']:.2f} gini={o['gini_all']:.2f} alpha(p50/p90)={o['hill_alpha_xmin_p50_pos']} / {o['hill_alpha_xmin_p90_pos']}")
        log("   positive-only quantiles:", qline(o["quantiles_positive_only"]))
    log(f"Spearman(in,out degree)={out['spearman_in_vs_out_degree']:.3f}; components={ncomp:,}, largest={out['largest_component_share']:.1%}")
    log("friend_date quantiles:", out["friend_date_quantiles_iso"])
    log("share of edges created AFTER cutoffs (leakage risk):", {k: round(x, 3) for k, x in out["share_edges_created_after_cutoff"].items()})
    if "mutual_pair_date_gap_days_abs" in out:
        log("reciprocal-pair date gap (days, abs):", qline(out["mutual_pair_date_gap_days_abs"]), f" identical date share={out['mutual_pair_share_identical_date']:.1%}")
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
    ng_in = C.in_deg[pd.Index(gset).get_indexer(ng)]
    ng_out = C.out_deg[pd.Index(gset).get_indexer(ng)]
    out["in_degree_graph_nonvoters"] = qcurve(ng_in, [.5, .9, .99, 1.0])
    out["out_degree_graph_nonvoters"] = qcurve(ng_out, [.5, .9, .99, 1.0])
    log(f"voters in friend graph: {out['share_voters_in_graph']:.1%} (with out-edge {out['share_voters_with_out_edge']:.1%}, with followers {out['share_voters_with_in_edge']:.1%})")
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
def compute_exposure(C, direction, chunk=50000):
    """Vote rows whose voter had >=1 'influencing' neighbour vote earlier on the same story.
    direction 'followee': voter follows the earlier voter (user_id=voter, friend_id=earlier voter)  [influence]
    direction 'follower': the earlier voter follows the current voter                                [placebo]"""
    a_all, b_all = (C.aidx, C.bidx) if direction == "followee" else (C.bidx, C.aidx)
    ing = C.v_gi >= 0
    vm = np.zeros(C.N, bool)
    vm[C.v_gi[ing]] = True
    sel = vm[a_all] & vm[b_all]
    a, b = a_all[sel].astype(np.int64), b_all[sel].astype(np.int64)
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
    return n_exp, d_last, d_first, at_risk_vote


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
    ab = np.digitize(v["age_h"].values, [1, 4, 24])
    res["by_story_age_at_risk"] = {nm: float(exposed[at_risk & (ab == i)].mean()) if (at_risk & (ab == i)).any() else None
                                   for i, nm in enumerate(["<1h", "1-4h", "4-24h", ">24h"])}
    res["n_exposing_friends_among_exposed"] = qcurve(n_exp[exposed], [.5, .75, .9, .99])
    res["delay_h_from_last_exposing_vote"] = qcurve(d_last[exposed], [.1, .25, .5, .75, .9])
    res["delay_h_from_first_exposing_vote"] = qcurve(d_first[exposed], [.1, .25, .5, .75, .9])
    dl = d_last[exposed]
    res["share_exposed_with_last_exposure_within"] = {f"{h}h": float((dl <= h).mean()) for h in (0.25, 1, 6, 24)} if len(dl) else {}
    return res


@section("exposure", "8. SOCIAL EXPOSURE (was the voter preceded by a vote of someone they follow?)")
def s_exposure(C):
    out = {}
    for direction in ("followee", "follower"):
        n_exp, dl, df_, risk = compute_exposure(C, direction)
        out[direction] = exposure_summary(C, n_exp, dl, df_, risk)
        if direction == "followee":
            C.v["n_exp"] = n_exp
            C.v["exposed"] = n_exp > 0
        r = out[direction]
        log(f"[{direction}] exposed share: all votes {r['exposed_share_all_votes']:.2%}; among votes by users who have >=1 voting friend {r['exposed_share_at_risk_votes']}")
        log("   by story bucket:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_story_bucket_at_risk"].items()})
        log("   by cascade position:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_cascade_position_at_risk"].items()})
        log("   by story age:", {k: (round(x, 3) if x is not None else None) for k, x in r["by_story_age_at_risk"].items()})
        log("   #exposing friends | exposed:", qline(r["n_exposing_friends_among_exposed"]))
        log("   delay from last exposing vote (h):", qline(r["delay_h_from_last_exposing_vote"]))
    a = out["followee"]["exposed_share_at_risk_votes"]
    b = out["follower"]["exposed_share_at_risk_votes"]
    out["influence_direction_ratio_followee_over_follower"] = (a / b) if (a and b) else None
    log(f"direction check: followee/follower exposure ratio = {out['influence_direction_ratio_followee_over_follower']}  (>>1 means 'you vote after the people you follow')")
    return out


@section("adoption_vs_exposure", "9. ADOPTION vs NUMBER OF VOTING FRIENDS (sampled stories, end-of-story exposure)")
def s_adoption(C):
    rng = np.random.default_rng(C.seed)
    S = C.S
    risk = C.out_deg > 0
    K = 10
    groups = {"small(p00-50)": [0], "mid(p50-95)": [1, 2], "large(p95-100)": [3, 4]}
    out = {}
    for gname, bks in groups.items():
        pool = np.flatnonzero(np.isin(S["bk"].values, bks))
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(60, len(pool)), replace=False)
        at, vo = np.zeros(K + 1), np.zeros(K + 1)
        for j in pick:
            seg = C.v_gi[C.starts[j]: C.starts[j] + S["n_votes"].values[j]]
            vm = np.zeros(C.N, bool)
            vm[seg[seg >= 0]] = True
            k = np.bincount(C.aidx[vm[C.bidx]], minlength=C.N)
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


@section("homophily", "10. HOMOPHILY & LATENT DIMENSION (do friends vote for the same stories?)")
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
    if len(ea) > 200000:
        p = rng.choice(len(ea), 200000, replace=False)
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
    log(f"pairs={len(ea):,}  mean shared stories: friends={i_f.mean():.2f}  activity-matched random={i_m.mean():.2f}  random={i_r.mean():.2f}")
    log(f"mean Jaccard: friends={j_f.mean():.4f}  matched={j_m.mean():.4f}  -> homophily ratio {out['homophily_ratio_jaccard_vs_matched']}")
    if not C.skip_svd:
        try:
            from scipy.sparse.linalg import svds
            rows = np.flatnonzero(nvote >= 5)
            A5 = A[rows]
            A5 = sparse.diags(1.0 / np.sqrt(nvote[rows])) @ A5
            k = int(min(15, min(A5.shape) - 2))
            sv = np.sort(svds(A5, k=k, return_singular_vectors=False))[::-1]
            out["svd_variance_share_top_components_(unit-norm_user_rows)"] = (sv ** 2 / A5.shape[0])
            log("SVD variance share, top components (1st is mostly 'popularity'):", np.round(out["svd_variance_share_top_components_(unit-norm_user_rows)"], 4).tolist())
        except Exception as ex:
            log(f"SVD skipped: {ex}")
    return out


def build_story_features(C):
    v, S, U = C.v, C.S, C.U
    F = pd.DataFrame(index=S.index)
    F["n_votes"] = S["n_votes"]
    F["t_first"] = S["t_first"]
    F["t_last"] = S["t_last"]
    F["bucket"] = [BUCKET_NAMES[i] for i in S["bk"].values]
    for col in S.columns:
        if col.startswith("v_") or col in ("peak_age_h", "peak_votes_h", "peak_share", "lifetime_h"):
            F[col] = S[col]
    first = v.iloc[C.starts]
    F["first_voter_id"] = first["voter_id"].values
    F["first_voter_indeg"] = first["indeg"].values
    F["first_voter_outdeg"] = first["outdeg"].values
    F["first_voter_nvotes"] = U["n_votes"].reindex(first["voter_id"].values).values
    dt = pd.to_datetime(S["t_first"].values, unit="s", utc=True)
    F["hour"] = np.asarray(dt.hour)
    F["dow"] = np.asarray(dt.weekday)
    for w in FEAT_W:
        ws = f"{w:g}"
        sub = v[v["age_h"] <= w]
        g = sub.groupby("story_id")
        F[f"n_{ws}"] = S[f"v_{ws}h"]
        F[f"fol_sum_{ws}"] = g["indeg"].sum().reindex(F.index).fillna(0)
        F[f"fol_mean_{ws}"] = g["indeg"].mean().reindex(F.index)
        F[f"fol_max_{ws}"] = g["indeg"].max().reindex(F.index)
        F[f"in_graph_{ws}"] = g["in_graph"].mean().reindex(F.index)
        F[f"hub_{ws}"] = g["is_hub"].sum().reindex(F.index).fillna(0)
        F[f"exp_share_{ws}"] = g["exposed"].mean().reindex(F.index)
        F[f"gap_mean_{ws}"] = g["gap_s"].mean().reindex(F.index)
        F[f"gap_std_{ws}"] = g["gap_s"].std().reindex(F.index)
        half = S[f"v_{w / 2:g}h"]
        F[f"accel_{ws}"] = (S[f"v_{ws}h"] - half) / half.clip(lower=1)
    vgi = C.v_gi
    nv = S["n_votes"].values
    for w in EARLY_W:
        ws = f"{w:g}"
        nw = S[f"v_{ws}h"].values
        dens, edges_n, conn = np.full(len(S), np.nan), np.zeros(len(S)), np.full(len(S), np.nan)
        for j in range(len(S)):
            seg = vgi[C.starts[j]: C.starts[j] + nw[j]]
            idx = np.unique(seg[seg >= 0][:300])
            m = len(idx)
            if m >= 2:
                sub = C.Adj[idx][:, idx]
                edges_n[j] = sub.nnz
                dens[j] = sub.nnz / (m * (m - 1))
                deg = np.asarray(sub.sum(axis=0)).ravel() + np.asarray(sub.sum(axis=1)).ravel()
                conn[j] = float((deg > 0).mean())
        F[f"clust_dens_{ws}"] = dens
        F[f"clust_edges_{ws}"] = edges_n
        F[f"clust_conn_{ws}"] = conn
    return F


@section("early_signal", "11. EARLY-SIGNAL STRENGTH (what each information layer is worth on REAL data)")
def s_early(C):
    F = build_story_features(C)
    C.F = F
    n = F["n_votes"].values.astype(float)
    labels = {"top50": n >= np.quantile(n, .5), "top20": n >= np.quantile(n, .8)}
    out = {"label_positive_rate": {k: float(y.mean()) for k, y in labels.items()},
           "majority_class_accuracy": {k: float(max(y.mean(), 1 - y.mean())) for k, y in labels.items()}}
    log("label positive rates:", out["label_positive_rate"], " majority-class accuracy:", out["majority_class_accuracy"])

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
    PROFILE["meta"] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "data_dir": C.data_dir, "seed": C.seed,
                       "buckets": BUCKET_NAMES, "bucket_edges_pct": BUCKET_EDGES, "windows_h": WINDOWS_H}
    if hasattr(C, "F"):
        extra = C.F
    else:
        extra = C.S
    extra.reset_index().to_csv(os.path.join(od, "story_table.csv"), index=False)
    C.U.reset_index().to_csv(os.path.join(od, "user_table.csv"), index=False)
    with open(os.path.join(od, "profile.json"), "w") as fh:
        json.dump(jsonable(PROFILE), fh, indent=1)
    log(f"wrote story_table.csv ({len(extra):,} rows), user_table.csv ({len(C.U):,} rows), profile.json in {od}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Extract the statistics needed to synthesise Digg-like data.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--votes", nargs="*", default=None)
    ap.add_argument("--friends", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-svd", action="store_true")
    args, _ = ap.parse_known_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.data_dir.rstrip("/"))), "digg_profile")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.abspath("digg_profile")
        os.makedirs(out_dir, exist_ok=True)

    t00 = time.time()
    C = SimpleNamespace(data_dir=args.data_dir, out_dir=out_dir, seed=args.seed, skip_cv=args.skip_cv, skip_svd=args.skip_svd)
    hdr("0. LOADING FILES (headerless CSVs, columns verified by value range)")
    vpaths, fpath = find_files(args)
    C.v, C.vinfo = load_votes(vpaths)
    C.e, C.finfo = load_friends(fpath)
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