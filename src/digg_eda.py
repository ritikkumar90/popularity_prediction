#!/usr/bin/env python3
"""
Digg EDA for popularity prediction (XGBoost / R-GCN / HGT design)
=================================================================
Run in Colab:   %run digg_eda.py      (or paste into a cell and call main())
Run locally:    python digg_eda.py /path/to/digg_folder

Expected raw files (Digg2009, Lerman et al.):
  digg_votes*.csv    : vote_date, voter_id, story_id
  digg_friends.csv   : mutual, friend_date, user_id, friend_id
Headers are optional; columns are auto-detected. Override in CONFIG if needed.

Outputs (in OUT_DIR): report.txt, metrics.json, story_table.csv, fig1/fig2 PNGs.
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore")

# =============================== CONFIG ====================================
DATA_DIR = os.environ.get(
    "DIGG_DIR",
    "/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw/digg",
)
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    DATA_DIR = sys.argv[1]
OUT_DIR = "digg_eda_output"

VOTES_FILE = None        # e.g. "digg_votes1.csv"  (None = auto: filenames containing 'vote')
FRIENDS_FILE = None      # e.g. "digg_friends.csv" (None = auto: filenames containing 'friend')
VOTES_COLS = None        # e.g. {"ts": "vote_date", "user": "voter_id", "story": "story_id"}
FRIENDS_COLS = None      # e.g. {"src": "user_id", "dst": "friend_id", "mutual": "mutual", "ts": "friend_date"}

# In Digg2009, row (user_id, friend_id) means user_id *follows / watches* friend_id.
# The script tests both directions and tells you which one looks right.
FOLLOW_DIRECTION = "src_follows_dst"     # or "dst_follows_src"

EARLY_WINDOWS_H = [0.25, 0.5, 1, 2, 6, 12, 24]   # observation windows (hours after 1st vote)
FEATURE_WINDOW_H = 1                              # window used for the quick feature-signal table
FINAL_AGE_H = 48                                  # story must be >= this old at dataset end to count as "mature"
SOCIAL_SAMPLE_STORIES = 300                       # stories sampled for the friend-exposure join
SPLIT = (0.70, 0.15, 0.15)                        # temporal split by story submission time
SEED = 42
EXT = (".csv", ".txt", ".gz", ".tsv", ".dat")
# ===========================================================================

LOG, REPORT = [], {}
rng = np.random.default_rng(SEED)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def section(t):
    log("\n" + "=" * 8 + f" {t} " + "=" * 8)


# ------------------------------ helpers ------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    c = np.cumsum(x)
    return float((len(x) + 1 - 2 * c.sum() / c[-1]) / len(x))


def describe(x, name):
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, [0.5, 0.9, 0.99])
    d = dict(n=len(x), mean=x.mean(), median=q[0], p90=q[1], p99=q[2], max=x.max(),
             skew=float(stats.skew(x)), gini=gini(x))
    log(f"{name}: n={d['n']:,}  mean={d['mean']:.2f}  median={d['median']:.1f}  p90={d['p90']:.1f}  "
        f"p99={d['p99']:.1f}  max={d['max']:.0f}  skew={d['skew']:.1f}  gini={d['gini']:.3f}")
    return d


def hill_alpha(x, xmin):
    t = np.asarray(x, dtype=float)
    t = t[t >= xmin]
    if len(t) < 30:
        return float("nan")
    return float(1 + len(t) / np.sum(np.log(t / (xmin - 0.5))))


def ccdf(ax, x, label):
    x = np.asarray(x)
    x = x[x > 0]
    v, c = np.unique(x, return_counts=True)
    y = 1 - (np.cumsum(c) - c) / len(x)
    ax.loglog(v, y, label=label)


def eta_sq(cat, y):
    df = pd.DataFrame({"c": np.asarray(cat), "y": np.asarray(y, dtype=float)})
    tot = ((df.y - df.y.mean()) ** 2).sum()
    if tot == 0:
        return float("nan")
    g = df.groupby("c").y.agg(["size", "mean"])
    return float((g["size"] * (g["mean"] - df.y.mean()) ** 2).sum() / tot)


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).correlation)


# ------------------------------ loading ------------------------------------
def sniff_sep(path):
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        first = f.readline()
    if "," in first:
        return ","
    if "\t" in first:
        return "\t"
    return r"\s+"


def read_table(path):
    sep = sniff_sep(path)
    eng = "python" if sep == r"\s+" else "c"
    probe = pd.read_csv(path, sep=sep, header=None, nrows=2, engine=eng)
    header = bool(pd.to_numeric(probe.iloc[0], errors="coerce").isna().all())
    df = pd.read_csv(path, sep=sep, header=0 if header else None, engine=eng)
    if not header:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df, header


def _find(cols, keys):
    for c in cols:
        if any(k in str(c).lower() for k in keys):
            return c
    return None


def infer_votes_cols(df):
    cols = list(df.columns)
    ts, st, us = _find(cols, ["date", "time", "stamp"]), _find(cols, ["story", "news", "item"]), _find(cols, ["voter", "user"])
    if ts and st and us and len({ts, st, us}) == 3:
        return dict(ts=ts, story=st, user=us)
    num = df.apply(pd.to_numeric, errors="coerce")
    ts = num.median().idxmax()
    rest = [c for c in cols if c != ts]
    nun = num[rest].nunique()
    st = nun.idxmin()          # fewer distinct values -> stories
    us = nun.idxmax()          # more distinct values  -> users
    return dict(ts=ts, story=st, user=us)


def infer_friends_cols(df):
    cols = list(df.columns)
    num = df.apply(pd.to_numeric, errors="coerce")
    m = {}
    mut = _find(cols, ["mutual"])
    if mut is None:
        for c in cols:
            if num[c].nunique() <= 2 and set(num[c].dropna().unique()) <= {0, 1}:
                mut = c
                break
    m["mutual"] = mut
    ts = _find(cols, ["date", "time", "stamp"])
    if ts is None:
        cand = [c for c in cols if c != mut and num[c].median() > 1e8]
        ts = cand[0] if cand else None
    m["ts"] = ts
    src = _find(cols, ["user"])
    dst = _find(cols, ["friend"])
    if src is None or dst is None or src == dst or src in (mut, ts) or dst in (mut, ts):
        rest = [c for c in cols if c not in (mut, ts)]
        src, dst = rest[0], rest[1]       # documented order: user_id, friend_id
    m["src"], m["dst"] = src, dst
    return m


def to_seconds(ts):
    ts = pd.to_numeric(ts, errors="coerce").astype("float64")
    if np.nanmedian(ts) > 1e11:           # milliseconds
        ts = ts / 1000.0
    return ts


def load_data():
    section("1. LOADING")
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Directory not found: {DATA_DIR}")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(EXT))
    log("Files:", files)
    vfiles = [VOTES_FILE] if VOTES_FILE else [f for f in files if "vote" in f.lower()]
    ffiles = [FRIENDS_FILE] if FRIENDS_FILE else [f for f in files if "friend" in f.lower()]
    if not vfiles:
        raise SystemExit("No votes file found. Set VOTES_FILE in CONFIG.")

    parts = []
    for f in vfiles:
        df, hdr = read_table(os.path.join(DATA_DIR, f))
        m = VOTES_COLS or infer_votes_cols(df)
        log(f"votes file {f}: header={hdr}, rows={len(df):,}, mapping={m}")
        parts.append(pd.DataFrame({"user": df[m["user"]].values, "story": df[m["story"]].values,
                                   "ts": to_seconds(df[m["ts"]]).values}))
    votes = pd.concat(parts, ignore_index=True)
    n0 = len(votes)
    votes = votes.dropna(subset=["user", "story", "ts"])
    if len(votes) < n0:
        log(f"dropped {n0 - len(votes):,} vote rows with missing values")

    fr = None
    if ffiles:
        df, hdr = read_table(os.path.join(DATA_DIR, ffiles[0]))
        m = FRIENDS_COLS or infer_friends_cols(df)
        log(f"friends file {ffiles[0]}: header={hdr}, rows={len(df):,}, mapping={m}")
        fr = pd.DataFrame({"src": df[m["src"]].values, "dst": df[m["dst"]].values})
        fr["mutual"] = pd.to_numeric(df[m["mutual"]], errors="coerce").values if m.get("mutual") else np.nan
        fr["ts"] = to_seconds(df[m["ts"]]).values if m.get("ts") else np.nan
    else:
        log("No friends file found -> graph sections will be skipped.")

    # contiguous ids shared by votes and friends (needed for sparse matrices / PyG)
    pools = [votes.user.values] + ([fr.src.values, fr.dst.values] if fr is not None else [])
    user_idx = pd.Index(pd.unique(np.concatenate(pools)))
    votes["u"] = user_idx.get_indexer(votes.user)
    votes["s"], story_cat = pd.factorize(votes.story)
    if fr is not None:
        fr["src"] = user_idx.get_indexer(fr.src)
        fr["dst"] = user_idx.get_indexer(fr.dst)
    votes = votes[["u", "s", "ts"]].sort_values(["s", "ts"], kind="mergesort").reset_index(drop=True)
    return votes, fr, len(user_idx), story_cat


# --------------------------- analysis sections -----------------------------
def votes_overview(votes, N):
    section("2. VOTES OVERVIEW")
    dup = int(votes.duplicated(["u", "s"]).sum())
    log(f"votes={len(votes):,}  stories={votes.s.nunique():,}  voters={votes.u.nunique():,}  duplicate (user,story) pairs={dup:,}")
    t0, t1 = pd.to_datetime(votes.ts.min(), unit="s"), pd.to_datetime(votes.ts.max(), unit="s")
    log(f"time range: {t0}  ->  {t1}  ({(t1 - t0).days} days)")
    log(f"share of votes with a timestamp tie inside the same story: "
        f"{votes.duplicated(['s', 'ts']).mean():.2%}  (ties make 'who voted first' ambiguous)")
    REPORT["votes"] = dict(n=len(votes), stories=int(votes.s.nunique()), voters=int(votes.u.nunique()), dup_pairs=dup)
    return dup


def build_story_table(votes):
    g = votes.groupby("s").ts
    st = pd.DataFrame({"n_votes": g.size(), "first_ts": g.min(), "last_ts": g.max()})
    assert (st.index.values == np.arange(len(st))).all()
    t_end = votes.ts.max()
    st["lifetime_h"] = (st.last_ts - st.first_ts) / 3600
    st["age_end_h"] = (t_end - st.first_ts) / 3600
    votes["age_h"] = (votes.ts - st.first_ts.to_numpy()[votes.s.to_numpy()]) / 3600
    votes["rk"] = votes.groupby("s").cumcount()
    return st


def story_popularity(st):
    section("3. TARGET: STORY POPULARITY (final votes per story)")
    y = st.n_votes.values
    REPORT["popularity"] = describe(y, "votes/story")
    REPORT["popularity"]["log_skew"] = float(stats.skew(np.log1p(y)))
    log(f"skew after log1p: {REPORT['popularity']['log_skew']:.2f}")
    s = np.sort(y)[::-1]
    for p in (0.01, 0.10, 0.20):
        k = max(1, int(len(s) * p))
        log(f"top {p:.0%} of stories hold {s[:k].sum() / s.sum():.1%} of all votes")
    log(f"stories with <10 votes: {(y < 10).mean():.1%};  <50: {(y < 50).mean():.1%};  >=1000: {(y >= 1000).mean():.1%}")
    log(f"Hill tail exponent alpha (votes/story, xmin=p50): {hill_alpha(y, max(np.median(y), 2)):.2f}")
    log("\nCandidate label definitions (thresholds in votes):")
    rows = []
    for q in (0.5, 0.8, 0.9):
        thr = np.quantile(y, q)
        rows.append(dict(rule=f"popular = votes >= p{int(q * 100)}", threshold=thr, positives=int((y >= thr).sum()),
                         pos_rate=f"{(y >= thr).mean():.1%}"))
    log(pd.DataFrame(rows).to_string(index=False))


def temporal_analysis(votes, st):
    section("4. TEMPORAL DYNAMICS & EARLY-SIGNAL STRENGTH")
    log(f"story lifetime (hours, 1st->last vote): median={st.lifetime_h.median():.1f}  p90={st.lifetime_h.quantile(.9):.1f}")
    log(f"stories younger than {FINAL_AGE_H}h at dataset end (right-censored): {(st.age_end_h < FINAL_AGE_H).mean():.1%}")
    for k in (10, 50, 100):
        tk = votes.loc[votes.rk == k - 1, "age_h"]
        if len(tk):
            log(f"time to {k:>3}th vote: reached by {len(tk) / len(st):.1%} of stories; median={tk.median():.2f}h  p90={tk.quantile(.9):.2f}h")

    mature = st.age_end_h >= FINAL_AGE_H
    if mature.sum() < 30:
        log("WARNING: <30 mature stories; using all stories for the early-vs-final analysis.")
        mature = pd.Series(True, index=st.index)
    rows = []
    for T in WINDOWS:
        e = st[f"early_{T}h"][mature]
        f = st.n_votes[mature]
        rows.append(dict(window_h=T, median_votes_in_window=e.median(),
                         median_share_of_final=(e / f).median(),
                         only_submitter_vote=f"{(e <= 1).mean():.1%}",
                         spearman_vs_final=spear(e, f),
                         spearman_vs_remaining_votes=spear(e, f - e)))
    tab = pd.DataFrame(rows)
    log(f"\nEarly votes vs final popularity (mature stories, n={int(mature.sum()):,}):")
    log(tab.round(3).to_string(index=False))
    good = tab[tab.spearman_vs_final >= 0.8]
    REPORT["early_signal"] = tab.to_dict("records")
    REPORT["early_window_for_rho_0.8"] = float(good.window_h.iloc[0]) if len(good) else None
    return tab, mature


def split_analysis(votes, st):
    section("5. TEMPORAL SPLIT, DRIFT & COLD-START")
    q1, q2 = st.first_ts.quantile([SPLIT[0], SPLIT[0] + SPLIT[1]])
    st["split"] = np.where(st.first_ts <= q1, "train", np.where(st.first_ts <= q2, "val", "test"))
    log(f"split by story first-vote time at {pd.to_datetime(q1, unit='s')} and {pd.to_datetime(q2, unit='s')}")
    thr = st.loc[st.split == "train", "n_votes"].quantile(0.8)
    rows = []
    for sp in ("train", "val", "test"):
        y = st.loc[st.split == sp, "n_votes"]
        rows.append(dict(split=sp, stories=len(y), mean_votes=y.mean(), median_votes=y.median(),
                         mean_log_votes=np.log1p(y).mean(), share_above_train_p80=f"{(y >= thr).mean():.1%}"))
    log(pd.DataFrame(rows).round(2).to_string(index=False))
    ks = stats.ks_2samp(np.log1p(st.loc[st.split == "train", "n_votes"]), np.log1p(st.loc[st.split == "test", "n_votes"]))
    log(f"KS test train vs test (log votes): D={ks.statistic:.3f}, p={ks.pvalue:.3g}  "
        f"{'-> distribution DRIFTS across time' if ks.pvalue < 0.01 else '-> no strong drift'}")
    REPORT["split"] = dict(ks_D=float(ks.statistic), ks_p=float(ks.pvalue), q1=float(q1), q2=float(q2))

    sp_v = st.split.to_numpy()[votes.s.to_numpy()]
    train_users = set(votes.loc[sp_v == "train", "u"].unique())
    tv = votes[sp_v == "test"]
    unseen_rows = (~tv.u.isin(train_users)).mean() if len(tv) else float("nan")
    unseen_users = (~pd.Series(tv.u.unique()).isin(train_users)).mean() if len(tv) else float("nan")
    log(f"COLD-START: test-story votes from users never seen in train stories: {unseen_rows:.1%} of votes, {unseen_users:.1%} of users")
    REPORT["cold_start_user_share"] = float(unseen_users)
    return q1, q2


def user_analysis(votes):
    section("6. USER ACTIVITY (votes per user)")
    vpu = votes.groupby("u").size().values
    REPORT["votes_per_user"] = describe(vpu, "votes/user")
    s = np.sort(vpu)[::-1]
    k = max(1, int(len(s) * 0.01))
    log(f"top 1% of users cast {s[:k].sum() / s.sum():.1%} of votes;  users with exactly 1 vote: {(vpu == 1).mean():.1%}")
    log(f"Hill tail exponent alpha (votes/user, xmin=p50): {hill_alpha(vpu, max(np.median(vpu), 2)):.2f}")
    return vpu


def friend_graph(fr, votes, N, q1):
    section("7. FRIEND (SOCIAL) GRAPH")
    n_raw = len(fr)
    dups = int(fr.duplicated(["src", "dst"]).sum())
    loops = int((fr.src == fr.dst).sum())
    log(f"edges={n_raw:,}  duplicate edges={dups:,}  self-loops={loops:,}")
    fr.drop_duplicates(["src", "dst"], inplace=True)
    fr.drop(fr.index[fr.src == fr.dst], inplace=True)
    fr.reset_index(drop=True, inplace=True)

    out_deg = np.bincount(fr.src.values, minlength=N)
    in_deg = np.bincount(fr.dst.values, minlength=N)
    in_graph = (out_deg + in_deg) > 0
    log(f"users in friend graph: {in_graph.sum():,}")
    REPORT["out_degree"] = describe(out_deg[out_deg > 0], "out-degree (users with >=1 out-edge)")
    REPORT["in_degree"] = describe(in_deg[in_deg > 0], "in-degree  (users with >=1 in-edge)")

    key = fr.src.values.astype(np.int64) * N + fr.dst.values
    rev = fr.dst.values.astype(np.int64) * N + fr.src.values
    has_rev = np.isin(rev, key)
    log(f"reciprocity (edges whose reverse edge exists): {has_rev.mean():.1%}")
    if fr.mutual.notna().any():
        m1 = fr.mutual.values == 1
        log(f"'mutual' flag: {m1.mean():.1%} of rows flagged mutual; of those, reverse row present in {has_rev[m1].mean():.1%}")
        if m1.any() and has_rev[m1].mean() < 0.9:
            log("  -> mutual links are NOT always stored in both directions: add reverse edges before building the graph.")
    REPORT["reciprocity"] = float(has_rev.mean())

    A = coo_matrix((np.ones(len(fr)), (fr.src.values, fr.dst.values)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    sizes = np.sort(np.bincount(lab[in_graph]))[::-1]
    sizes = sizes[sizes > 0]
    log(f"weakly connected components: {len(sizes):,}; largest holds {sizes[0] / in_graph.sum():.1%} of graph users")

    voters = np.unique(votes.u.values)
    log(f"voters present in friend graph: {in_graph[voters].mean():.1%};  votes cast by in-graph users: {in_graph[votes.u.values].mean():.1%}")
    REPORT["voters_in_graph"] = float(in_graph[voters].mean())
    log(f"in-graph users who never voted: {(in_graph & ~np.isin(np.arange(N), voters)).sum():,}")

    if fr.ts.notna().any():
        ts = fr.ts[fr.ts > 0]
        if len(ts):
            log(f"friend_date range: {pd.to_datetime(ts.min(), unit='s')} -> {pd.to_datetime(ts.max(), unit='s')}  "
                f"(rows with ts<=0: {(fr.ts <= 0).sum():,})")
            late = (fr.ts > q1).mean()
            log(f"LEAKAGE: {late:.1%} of friend edges are created AFTER the train/val cutoff. "
                "Filter edges by friend_date <= prediction time when building graph features.")
            REPORT["edges_after_train_cutoff"] = float(late)
    return out_deg, in_deg, in_graph


def exposure_analysis(votes, fr, st, N):
    section("8. SOCIAL INFLUENCE: were voters 'exposed' to an earlier friend vote?")
    S = len(st)
    ids = rng.choice(st.index.values, size=min(SOCIAL_SAMPLE_STORIES, S), replace=False)
    vs = votes.loc[votes.s.isin(ids), ["u", "s", "ts", "rk"]]
    users_in = np.unique(vs.u.values)
    ed = fr[fr.src.isin(users_in) & fr.dst.isin(users_in)]
    log(f"sampled {len(ids)} stories, {len(vs):,} votes, {len(ed):,} relevant friend edges")

    def run(fcol, ecol):
        e = ed[[fcol, ecol]].rename(columns={fcol: "f", ecol: "e"})
        a = e.merge(vs.rename(columns={"u": "f", "ts": "tf"})[["f", "s", "tf"]], on="f")
        a = a.merge(vs.rename(columns={"u": "e", "ts": "te"})[["e", "s", "te"]], on=["e", "s"])
        a = a[a.te < a.tf]
        pairs = a.groupby(["f", "s"]).size()
        return pairs

    keyv = vs.u.values.astype(np.int64) * S + vs.s.values
    res = {}
    for name, (fc, ec) in {"src_follows_dst": ("src", "dst"), "dst_follows_src": ("dst", "src")}.items():
        pairs = run(fc, ec)
        pk = pairs.index.get_level_values(0).values.astype(np.int64) * S + pairs.index.get_level_values(1).values
        res[name] = (np.isin(keyv, pk), pairs)
        log(f"  direction {name:>16}: {res[name][0].mean():.1%} of votes had >=1 earlier-voting friend "
            f"(mean #exposing friends among exposed: {pairs.mean() if len(pairs) else 0:.2f})")
    better = max(res, key=lambda k: res[k][0].mean())
    log(f"  -> the higher-exposure direction is '{better}'. Configured FOLLOW_DIRECTION = '{FOLLOW_DIRECTION}'."
        + ("" if better == FOLLOW_DIRECTION else "  CHECK: you may want to flip FOLLOW_DIRECTION."))
    exposed = res[FOLLOW_DIRECTION][0]
    vs = vs.assign(exposed=exposed)
    vs["bucket"] = pd.cut(vs.rk, [-1, 0, 9, 99, 999, 1e9], labels=["1st", "2-10", "11-100", "101-1000", ">1000"])
    bt = vs.groupby("bucket").exposed.agg(["mean", "size"])
    log("\nExposure rate by vote position in the story's cascade:")
    log(bt.round(3).to_string())
    per_story = vs.groupby("s").exposed.mean()
    rho = spear(per_story.values, st.n_votes.reindex(per_story.index).values)
    log(f"Spearman(story exposure share, final votes) = {rho:.3f}")
    REPORT["exposure"] = dict(rate=float(exposed.mean()), best_direction=better, rho_with_popularity=rho)
    return bt


def submitter_and_features(votes, st, fr, N, in_deg, out_deg, in_graph, mature):
    section("9. SUBMITTER ('submits' edge) & QUICK FEATURE-SIGNAL TABLE")
    first = votes.groupby("s").head(1).set_index("s")
    tie = (votes.ts.to_numpy() == st.first_ts.to_numpy()[votes.s.to_numpy()])
    ties = pd.Series(tie).groupby(votes.s.values).sum() > 1
    log(f"first voter treated as submitter. Stories where the earliest timestamp is tied: {ties.mean():.1%}")
    sub = first.u
    st["submitter"] = sub.values
    per_sub = st.groupby("submitter").size()
    log(f"distinct submitters: {len(per_sub):,}; stories/submitter: median={per_sub.median():.0f} max={per_sub.max()}; "
        f"top 1% submitters own {np.sort(per_sub.values)[::-1][:max(1, len(per_sub) // 100)].sum() / len(st):.1%} of stories")
    if per_sub.max() < 2:
        log("  NOTE: nobody submits more than one story -> submitter-ID embeddings cannot generalise.")

    W = FEATURE_WINDOW_H
    followers = in_deg if FOLLOW_DIRECTION == "src_follows_dst" else out_deg
    votes["fol"] = followers[votes.u.values] if fr is not None else 0
    votes["ing"] = in_graph[votes.u.values] if fr is not None else False
    ew = votes[votes.age_h <= W]
    agg = ew.groupby("s").agg(early_fol_sum=("fol", "sum"), early_fol_mean=("fol", "mean"),
                              early_fol_max=("fol", "max"), early_in_graph=("ing", "mean"))
    feat = pd.DataFrame(index=st.index)
    feat[f"early_votes_{W}h"] = st[f"early_{W}h"]
    feat = feat.join(agg).fillna(0)
    feat["submitter_followers"] = followers[st.submitter.values] if fr is not None else 0
    feat["submitter_n_stories"] = per_sub.reindex(st.submitter.values).values
    dt = pd.to_datetime(st.first_ts, unit="s")
    st["hour"], st["dow"] = dt.dt.hour.values, dt.dt.dayofweek.values

    y = np.log1p(st.n_votes)
    y_late = np.log1p(st.n_votes - st[f"early_{W}h"])
    rows = []
    for c in feat.columns:
        rows.append(dict(feature=c, spearman_vs_log_final=spear(feat[c][mature], y[mature]),
                         spearman_vs_log_remaining=spear(feat[c][mature], y_late[mature])))
    for c in ("hour", "dow"):
        rows.append(dict(feature=f"{c} (eta^2)", spearman_vs_log_final=eta_sq(st[c][mature], y[mature]),
                         spearman_vs_log_remaining=eta_sq(st[c][mature], y_late[mature])))
    tab = pd.DataFrame(rows)
    log(f"\nFeature signal using only information from the first {W}h (mature stories):")
    log(tab.round(3).to_string(index=False))
    log("(hour/dow rows show eta^2 = variance explained, not a correlation.)")
    REPORT["feature_signal"] = tab.to_dict("records")
    out = st.join(feat.drop(columns=[f"early_votes_{W}h"]))
    out["split"] = st.get("split")
    out.to_csv(os.path.join(OUT_DIR, "story_table.csv"))
    log(f"\nSaved per-story table with baseline XGBoost features -> {OUT_DIR}/story_table.csv")


def graph_summary(votes, st, fr, N, vpu):
    section("10. HETEROGRAPH SIZE (for R-GCN / HGT)")
    n_users, n_st = int(votes.u.nunique()), len(st)
    log(f"node types: User={N:,} (voters={n_users:,}), Story={n_st:,}")
    rels = [("User", "votes_for", "Story", len(votes)), ("Story", "voted_by", "User", len(votes)),
            ("User", "submits", "Story", n_st), ("Story", "submitted_by", "User", n_st)]
    if fr is not None:
        rels.insert(0, ("User", "follows", "User", len(fr)))
    log(pd.DataFrame(rels, columns=["src", "relation", "dst", "edges"]).to_string(index=False))
    total = sum(r[3] for r in rels)
    log(f"total edges (with reverse): {total:,}")
    log(f"avg votes/story={len(votes) / n_st:.0f}, avg votes/user={len(votes) / n_users:.1f}")
    W = FEATURE_WINDOW_H
    early_edges = int((votes.age_h <= W).sum())
    log(f"if the model may only see votes from the first {W}h of each story, the vote graph keeps "
        f"{early_edges:,} edges ({early_edges / len(votes):.1%} of all)")
    log("There are NO node attributes in the raw data (no story title/topic/user profile): "
        "node features must be engineered or learned.")
    REPORT["graph"] = dict(users=N, stories=n_st, total_edges=total)


# ------------------------------- plots -------------------------------------
def make_plots(st, vpu, out_deg, in_deg, tab, bt, mature, q1, q2, fr):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].hist(np.log10(st.n_votes), bins=40, color="#4c72b0")
    ax[0, 0].set(title="Votes per story (log10)", xlabel="log10(votes)", ylabel="stories")
    ccdf(ax[0, 1], st.n_votes, "votes/story")
    ccdf(ax[0, 1], vpu, "votes/user")
    ax[0, 1].set(title="CCDF: story popularity & user activity", xlabel="count", ylabel="P(X >= x)")
    ax[0, 1].legend()
    if fr is not None:
        ccdf(ax[0, 2], out_deg, "out-degree")
        ccdf(ax[0, 2], in_deg, "in-degree")
        ax[0, 2].set(title="CCDF: friend-graph degree", xlabel="degree", ylabel="P(X >= x)")
        ax[0, 2].legend()
    cols = [f"early_{T}h" for T in WINDOWS]
    frac = np.column_stack([(st[c][mature] / st.n_votes[mature]).values for c in cols])
    ax[1, 0].plot(WINDOWS, np.median(frac, 0), marker="o")
    ax[1, 0].fill_between(WINDOWS, np.quantile(frac, .25, 0), np.quantile(frac, .75, 0), alpha=.25)
    ax[1, 0].set(title="Share of final votes by window (median, IQR)", xlabel="hours after first vote", xscale="log")
    W = FEATURE_WINDOW_H
    m = mature.values
    idx = rng.choice(np.where(m)[0], size=min(2000, m.sum()), replace=False)
    ax[1, 1].scatter(st[f"early_{W}h"].values[idx] + 1, st.n_votes.values[idx], s=6, alpha=.4)
    ax[1, 1].set(title=f"Votes in first {W}h vs final", xscale="log", yscale="log", xlabel=f"early votes +1", ylabel="final votes")
    ax[1, 2].plot(tab.window_h, tab.spearman_vs_final, marker="o", label="vs final")
    ax[1, 2].plot(tab.window_h, tab.spearman_vs_remaining_votes, marker="s", label="vs remaining (final-early)")
    ax[1, 2].set(title="Early-vote Spearman by window", xlabel="hours", xscale="log", ylim=(0, 1))
    ax[1, 2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_distributions.png"), dpi=120)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    d = pd.to_datetime(st.first_ts, unit="s").dt.floor("D").value_counts().sort_index()
    ax[0].bar(d.index, d.values, width=1.0)
    for q in (q1, q2):
        ax[0].axvline(pd.to_datetime(q, unit="s"), color="red", ls="--")
    ax[0].set(title="Stories per day (red = train/val/test cutoffs)")
    if bt is not None:
        ax[1].bar(bt.index.astype(str), bt["mean"].values)
        ax[1].set(title="Share of votes with an earlier-voting friend, by vote position", ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_time_and_social.png"), dpi=120)
    plt.show()


# ------------------------------ design notes -------------------------------
def design_notes(has_fr):
    section("11. DESIGN NOTES (rule-based, read together with the numbers above)")
    P = REPORT["popularity"]
    notes = []
    if P["skew"] > 2:
        notes.append(f"Target is heavy-tailed (skew {P['skew']:.1f}; {P['log_skew']:.1f} after log1p): regress on log1p(votes) "
                     "or use quantile classes; plain MSE on raw counts will be dominated by a few viral stories.")
    w = REPORT.get("early_window_for_rho_0.8")
    notes.append(f"Early-vote Spearman reaches 0.8 at window = {w}h." if w else
                 "Early votes never reach Spearman 0.8 in the tested windows: prediction from early votes alone will be limited.")
    notes.append("Pick the observation window from the early-signal table: the smallest window with a good "
                 "'vs remaining votes' correlation. Build ALL features and ALL graph edges from that window only.")
    if REPORT["split"]["ks_p"] < 0.01:
        notes.append("Popularity drifts over time: use a temporal split (as done here), never a random split.")
    if REPORT.get("cold_start_user_share", 0) > 0.2:
        notes.append(f"{REPORT['cold_start_user_share']:.0%} of test-story voters are unseen in training: avoid pure user-ID "
                     "embeddings; prefer inductive features (degree, activity counts) or neighbour-based aggregation.")
    if has_fr:
        if REPORT.get("voters_in_graph", 1) < 0.8:
            notes.append(f"Only {REPORT['voters_in_graph']:.0%} of voters appear in the friend graph: many User nodes will have no 'follows' edges.")
        if REPORT.get("edges_after_train_cutoff", 0) > 0.05:
            notes.append("Friend edges keep appearing after the cutoff: filter by friend_date to avoid leaking future structure.")
        ex = REPORT.get("exposure", {})
        if ex:
            notes.append(f"{ex['rate']:.0%} of votes (whole cascade) follow an earlier friend vote; Spearman(exposure, popularity)="
                         f"{ex['rho_with_popularity']:.2f}. Caution: this is measured over the FULL cascade and is partly mechanical "
                         "(big cascades are exposed by chance). It shows the 'follows' edges carry cascade structure, but as a model "
                         "feature recompute exposure ONLY inside the early window, otherwise it leaks the future.")
        d = REPORT["in_degree"]
        if d["skew"] > 5:
            notes.append("Degrees are extremely skewed: use neighbour sampling (e.g. PyG NeighborLoader) with capped fan-outs, "
                         "and log-transform degree features.")
    if REPORT["graph"]["total_edges"] > 5_000_000:
        notes.append("Graph is large for full-batch training: mini-batch around the target stories with sampling.")
    notes.append("No raw node attributes exist: for R-GCN/HGT initialise User nodes with log-degree/activity features "
                 "(or learned embeddings) and Story nodes with early-window statistics (early votes, early voter followers).")
    notes.append("XGBoost baseline: start from story_table.csv; it is the reference every graph model must beat. "
                 "Check the feature-signal table to see which engineered features carry information.")
    for i, n in enumerate(notes, 1):
        log(f"{i}. {n}")


# --------------------------------- main ------------------------------------
def main():
    global WINDOWS
    WINDOWS = sorted(set(EARLY_WINDOWS_H + [FEATURE_WINDOW_H]))
    os.makedirs(OUT_DIR, exist_ok=True)
    votes, fr, N, story_cat = load_data()
    has_fr = fr is not None

    votes_overview(votes, N)
    st = build_story_table(votes)
    for T in WINDOWS:
        st[f"early_{T}h"] = votes[votes.age_h <= T].groupby("s").size().reindex(st.index, fill_value=0)
    st.insert(0, "story_id", np.asarray(story_cat)[st.index.values])

    story_popularity(st)
    tab, mature = temporal_analysis(votes, st)
    q1, q2 = split_analysis(votes, st)
    vpu = user_analysis(votes)

    out_deg = in_deg = np.zeros(N, dtype=int)
    in_graph = np.zeros(N, dtype=bool)
    bt = None
    if has_fr:
        out_deg, in_deg, in_graph = friend_graph(fr, votes, N, q1)
        bt = exposure_analysis(votes, fr, st, N)
    submitter_and_features(votes, st, fr, N, in_deg, out_deg, in_graph, mature)
    graph_summary(votes, st, fr, N, vpu)
    make_plots(st, vpu, out_deg[out_deg > 0], in_deg[in_deg > 0], tab, bt, mature, q1, q2, fr if has_fr else None)
    design_notes(has_fr)

    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(LOG))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(REPORT, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nDone. Outputs in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()