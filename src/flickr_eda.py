#!/usr/bin/env python3
"""
Flickr EDA for photo-popularity prediction (XGBoost / R-GCN / HGT design)
=========================================================================
Run in Colab:   %run flickr_eda.py
Run locally:    python flickr_eda.py /path/to/flickr_folder

Files (MPI-SWS Flickr data, tab-separated, gzip, no header):
  flickr-all-photos.txt.gz                  : photo_id, upload_time, owner_id, views, comments, favorites
  flickr-all-photo-favorite-markings.txt.gz : user_id, photo_id, marking_time
  flickr-growth.txt.gz                      : user_a, user_b, date   (link observed from a to b)

The data is large (~34.7M markings, ~11.3M photos, ~33M links), so:
  * markings are sampled BY PHOTO (all markings of a sampled photo are kept, so cascades stay intact)
    -> PHOTO_SAMPLE_MOD = 4 keeps ~25% of photos. Use 1 for everything once the script works.
  * files are read in chunks.
Set MAX_ROWS for a quick dry run (rows are then a biased prefix; do not trust the statistics).

Outputs (OUT_DIR): report.txt, metrics.json, photo_table_sample.csv, fig1/fig2 PNGs.

Lessons from the Digg EDA that are built in:
  * popularity is measured over a FIXED horizon after upload (never "final count" with different ages);
  * early-signal strength is judged against the REMAINING markings (total-vs-early correlation is mechanical);
  * crawl-time totals (views / comments / favorites) are flagged as leakage, not used as features;
  * exposure statistics are descriptive only; time-respecting versions are reported when timestamps are comparable.
"""
import os
import sys
import gzip
import json
import time
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
    "FLICKR_DIR",
    "/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw/flickr",
)
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    DATA_DIR = sys.argv[1]
OUT_DIR = "flickr_eda_output"

GROWTH_FILE = "flickr-growth.txt.gz"
FAV_FILE = "flickr-all-photo-favorite-markings.txt.gz"
PHOTOS_FILE = "flickr-all-photos.txt.gz"

# Seconds per raw timestamp unit. None = auto (ms if values > 1e11, else seconds; date strings are parsed).
# The timestamps are described as "anonymized": check the unit warnings in section 1/4 and override if needed.
TIME_SCALE = {"photos": None, "fav": None, "growth": None}

PHOTO_SAMPLE_MOD = int(os.environ.get("FLICKR_MOD", 4))   # keep 1/MOD of photos (1 = all)
MAX_ROWS = {"photos": None, "fav": None, "growth": None}  # e.g. 3_000_000 for a dry run
CHUNK = 2_000_000

# Link (a -> b) = "a lists b as contact": a FOLLOWS b, so a sees b's favorites. Both directions are tested below.
FOLLOW_DIRECTION = "src_follows_dst"     # or "dst_follows_src"

HORIZONS_DAYS = [1, 7, 30, 90, 365]      # candidate popularity horizons after upload
TARGET_HORIZON_DAYS = 30                 # horizon used for the target / early-signal analysis
EARLY_WINDOWS_H = [1, 6, 24, 72, 168]    # observation windows after upload (hours)
FEATURE_WINDOW_H = 24                    # window used for the leak-safe feature-signal table
EXPOSURE_SAMPLE_PHOTOS = 3000            # photos (>=2 markings) sampled for the social-exposure join
FEATURE_SAMPLE_PHOTOS = 1_000_000
SAVE_SAMPLE_ROWS = 500_000
SPLIT = (0.70, 0.15, 0.15)               # temporal split by upload time
MAX_CORR_ROWS = 300_000                  # rows used per Spearman computation
SEED = 42
# ===========================================================================

LOG, REPORT = [], {}
rng = np.random.default_rng(SEED)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
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
    x = x[np.isfinite(x)]
    if len(x) == 0:
        log(f"{name}: no data")
        return {}
    q = np.quantile(x, [0.5, 0.9, 0.99])
    d = dict(n=len(x), mean=float(x.mean()), median=float(q[0]), p90=float(q[1]), p99=float(q[2]),
             max=float(x.max()), skew=float(stats.skew(x)), gini=gini(x))
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
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return
    v, c = np.unique(x, return_counts=True)
    y = 1 - (np.cumsum(c) - c) / len(x)
    ax.loglog(v, y, label=label)


def spear(a, b, cap=MAX_CORR_ROWS):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) > cap:
        i = rng.choice(len(a), cap, replace=False)
        a, b = a[i], b[i]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).correlation)


def eta_sq(cat, y):
    df = pd.DataFrame({"c": np.asarray(cat), "y": np.asarray(y, float)}).dropna()
    tot = ((df.y - df.y.mean()) ** 2).sum()
    if tot == 0 or len(df) < 3:
        return float("nan")
    g = df.groupby("c").y.agg(["size", "mean"])
    return float((g["size"] * (g["mean"] - df.y.mean()) ** 2).sum() / tot)


def dt_str(sec):
    try:
        return str(pd.to_datetime(float(sec), unit="s"))
    except Exception:
        return f"{sec:.0f}s"


# ------------------------------ reading ------------------------------------
class TimeParser:
    """Converts a raw timestamp column to float seconds; decides the format on the first chunk."""

    def __init__(self, name):
        self.name, self.mode, self.scale = name, None, TIME_SCALE.get(name)

    def _fit(self, s):
        num = pd.to_numeric(s, errors="coerce")
        if num.isna().mean() > 0.5:
            self.mode = "datetime"
            log(f"[{self.name}] timestamps look like date strings (example: {s.iloc[0]!r}) -> parsed as dates")
        else:
            self.mode = "numeric"
            med = float(num.median())
            if self.scale is None:
                self.scale = 1e-3 if med > 1e11 else 1.0
            log(f"[{self.name}] numeric timestamps, raw min/median/max in first chunk = "
                f"{num.min():.6g} / {med:.6g} / {num.max():.6g}; assuming {self.scale:g} seconds per unit "
                f"(override via TIME_SCALE if this is wrong, e.g. 86400 for days)")

    def __call__(self, s):
        if self.mode is None:
            self._fit(s)
        if self.mode == "datetime":
            dt = pd.to_datetime(s, errors="coerce")
            return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype="float64")
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64") * self.scale


def sniff(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        first = f.readline()
    return "\t" if "\t" in first else r"\s+"


def iter_chunks(path, ncols, max_rows):
    sep = sniff(path)
    eng = "python" if sep == r"\s+" else "c"
    probe = pd.read_csv(path, sep=sep, header=None, nrows=2, engine=eng, dtype=str)
    n = min(ncols, probe.shape[1])
    hdr = bool(pd.to_numeric(probe.iloc[0, :n], errors="coerce").isna().all())
    rd = pd.read_csv(path, sep=sep, header=0 if hdr else None, usecols=list(range(n)), chunksize=CHUNK, engine=eng)
    done = 0
    for ch in rd:
        ch.columns = list(range(n))
        if max_rows is not None and done + len(ch) > max_rows:
            ch = ch.iloc[: max_rows - done]
        done += len(ch)
        yield ch
        if max_rows is not None and done >= max_rows:
            break


def keep_photo(p):
    """Deterministic hash-based photo sample so photos, markings and cascades stay consistent."""
    if PHOTO_SAMPLE_MOD <= 1:
        return np.ones(len(p), dtype=bool)
    x = p.astype(np.uint64)
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x45D9F3B)
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x45D9F3B)
    x ^= x >> np.uint64(16)
    return (x % np.uint64(PHOTO_SAMPLE_MOD)) == 0


def num_col(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")


def load_photos():
    path = os.path.join(DATA_DIR, PHOTOS_FILE)
    if not os.path.exists(path):
        log(f"[photos] {PHOTOS_FILE} not found -> upload times / owners unavailable (most sections will be skipped)")
        return None
    tp, parts = TimeParser("photos"), []
    for ch in iter_chunks(path, 6, MAX_ROWS["photos"]):
        n = ch.shape[1]
        p, o, t = num_col(ch[0]), num_col(ch[2]), tp(ch[1])
        ok = np.isfinite(p) & np.isfinite(o)
        d = pd.DataFrame({"photo": p[ok].astype(np.int64), "upload_ts": t[ok], "owner": o[ok].astype(np.int64)})
        for j, nm in ((3, "views"), (4, "comments"), (5, "favs_meta")):
            d[nm] = num_col(ch[j])[ok].astype(np.float32) if n > j else np.float32(np.nan)
        parts.append(d)
    ph = pd.concat(parts, ignore_index=True)
    n0 = len(ph)
    ph = ph.drop_duplicates("photo")
    log(f"[photos] rows={n0:,}  duplicate photo ids dropped={n0 - len(ph):,}")
    return ph


def load_fav():
    path = os.path.join(DATA_DIR, FAV_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"Favorite-markings file not found: {path}")
    tp, parts, total = TimeParser("fav"), [], 0
    for ch in iter_chunks(path, 3, MAX_ROWS["fav"]):
        u, p, t = num_col(ch[0]), num_col(ch[1]), tp(ch[2])
        total += len(ch)
        ok = np.isfinite(u) & np.isfinite(p) & np.isfinite(t)
        u, p, t = u[ok].astype(np.int64), p[ok].astype(np.int64), t[ok]
        k = keep_photo(p)
        parts.append(pd.DataFrame({"user": u[k], "photo": p[k], "ts": t[k]}))
    fav = pd.concat(parts, ignore_index=True)
    log(f"[fav] rows read={total:,}; kept after photo sampling (1/{PHOTO_SAMPLE_MOD})={len(fav):,}")
    return fav


def load_growth():
    path = os.path.join(DATA_DIR, GROWTH_FILE)
    if not os.path.exists(path):
        log(f"[growth] {GROWTH_FILE} not found -> social-graph sections will be skipped")
        return None
    tp, parts = TimeParser("growth"), []
    for ch in iter_chunks(path, 3, MAX_ROWS["growth"]):
        a, b = num_col(ch[0]), num_col(ch[1])
        t = tp(ch[2]) if ch.shape[1] > 2 else np.full(len(ch), np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        parts.append(pd.DataFrame({"a": a[ok].astype(np.int64), "b": b[ok].astype(np.int64), "ts": t[ok]}))
    g = pd.concat(parts, ignore_index=True)
    log(f"[growth] rows={len(g):,}")
    return g


class Ctx:
    pass


C = Ctx()


# ------------------------------ 1. loading ---------------------------------
def sec_loading():
    section("1. LOADING & FORMAT CHECKS")
    log(f"DATA_DIR={DATA_DIR}  PHOTO_SAMPLE_MOD={PHOTO_SAMPLE_MOD}")
    photos = load_photos()
    fav = load_fav()
    growth = load_growth()

    if photos is not None:
        C.n_photos_file = len(photos)
        # owner statistics from the FULL photo table (before sampling)
        photos = photos.sort_values(["owner", "upload_ts"], kind="mergesort").reset_index(drop=True)
        photos["owner_prior_photos"] = photos.groupby("owner").cumcount().astype(np.int32)
        photos["owner_total_photos"] = photos.groupby("owner")["photo"].transform("size").astype(np.int32)
        C.owner_total = photos.groupby("owner").size().to_numpy()
        photos = photos[keep_photo(photos.photo.values)].reset_index(drop=True)
    else:
        C.n_photos_file, C.owner_total = 0, np.array([])

    # ---- id spaces -------------------------------------------------------
    arrs = [fav.user.values] + ([photos.owner.values] if photos is not None else []) \
        + ([growth.a.values, growth.b.values] if growth is not None else [])
    mx = max(int(a.max()) for a in arrs if len(a))
    if mx <= 50_000_000:
        ucat, N = None, mx + 1
    else:
        ucat = np.unique(np.concatenate(arrs))
        N = len(ucat)
        log(f"user ids are large (max {mx:,}); remapped to 0..{N - 1}")

    def U(a):
        return a if ucat is None else np.searchsorted(ucat, a)

    pcat = np.unique(np.concatenate([fav.photo.values] + ([photos.photo.values] if photos is not None else [])))
    P = len(pcat)
    C.N, C.P = N, P

    # ---- photo table -----------------------------------------------------
    upload = np.full(P, np.nan)
    owner = np.full(P, -1, dtype=np.int64)
    meta = {k: np.full(P, np.nan, dtype=np.float32) for k in ("views", "comments", "favs_meta")}
    prior = np.full(P, np.nan, dtype=np.float32)
    total = np.full(P, np.nan, dtype=np.float32)
    if photos is not None:
        ix = np.searchsorted(pcat, photos.photo.values)
        upload[ix] = photos.upload_ts.values
        owner[ix] = U(photos.owner.values)
        for k in meta:
            meta[k][ix] = photos[k].values
        prior[ix] = photos.owner_prior_photos.values
        total[ix] = photos.owner_total_photos.values
    C.upload, C.owner, C.meta, C.prior, C.total = upload, owner, meta, prior, total

    # ---- markings: sort by (photo, time) ----------------------------------
    p = np.searchsorted(pcat, fav.photo.values)
    u = U(fav.user.values)
    ts = fav.ts.values
    o = np.lexsort((ts, p))
    p, u, ts = p[o], u[o], ts[o]
    n = len(p)
    first = np.r_[True, p[1:] != p[:-1]]
    gs = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    C.p, C.u, C.ts, C.rk, C.n = p, u, ts, (np.arange(n) - gs), n
    C.t_end = float(ts.max())
    C.n_marks = np.bincount(p, minlength=P)

    age = (ts - upload[p]) / 3600.0
    C.age_h = age
    C.valid = np.isfinite(age) & (age >= 0)

    # ---- graph -------------------------------------------------------------
    C.edges = None
    if growth is not None:
        s, d, et = U(growth.a.values), U(growth.b.values), growth.ts.values
        C.n_edges_raw = len(s)
        key = s.astype(np.int64) * N + d
        order = np.argsort(key, kind="stable")
        ks = key[order]
        firstk = np.r_[True, ks[1:] != ks[:-1]]
        sel = order[firstk]
        C.n_edge_dups = int(len(s) - len(sel))
        s, d, et = s[sel], d[sel], et[sel]
        loops = s == d
        C.n_loops = int(loops.sum())
        s, d, et = s[~loops], d[~loops], et[~loops]
        C.e_src, C.e_dst, C.e_ts = s.astype(np.int32), d.astype(np.int32), et
        C.edges = True
    log(f"\nparsed tables: markings={n:,} (sampled), photos in marking table={P:,}, user-id space={N:,}")

    # ---- format sanity -----------------------------------------------------
    if photos is not None:
        log("\nfirst parsed rows (check that columns are what you expect):")
        log("photos  : " + photos.head(3)[["photo", "upload_ts", "owner", "views", "comments", "favs_meta"]]
            .to_string(index=False).replace("\n", "\n          "))
        log("markings: " + fav.head(3).to_string(index=False).replace("\n", "\n          "))
        known = np.isfinite(upload[np.unique(p)])
        log(f"photos in markings with an entry in the photos file: {known.mean():.1%}")
        fans = np.zeros(N, bool)
        fans[u] = True
        ow = owner[owner >= 0]
        log(f"photo owners that also appear as fans: {fans[np.unique(ow)].mean():.1%}")
        if growth is not None:
            ing = np.zeros(N, bool)
            ing[C.e_src] = True
            ing[C.e_dst] = True
            log(f"photo owners present in the friend graph: {ing[np.unique(ow)].mean():.1%};  "
                f"fans present in the friend graph: {ing[np.unique(u)].mean():.1%}")
            if ing[np.unique(ow)].mean() < 0.05:
                log("  WARNING: almost no owner id matches the graph -> check the column order of the photos file.")
        v = C.valid
        negative = np.isfinite(age) & (age < 0)
        log(f"markings dated BEFORE the photo's upload time: {negative.sum() / max(np.isfinite(age).sum(), 1):.2%} "
            "(should be ~0; a large value means the two files use different time scales/offsets)")
        if v.any():
            med = float(np.median(age[v][:5_000_000]))
            log(f"median lag upload->marking = {med:.2f} hours")
            if med < 1 / 60 or med > 24 * 365 * 3:
                log("  WARNING: implausible median lag -> the time unit of 'photos' vs 'fav' is probably wrong; set TIME_SCALE.")
    return photos


# ------------------------------ 2. photo metadata --------------------------
def sec_photos():
    section("2. PHOTO METADATA & OWNERS")
    log(f"photos in photos file: {C.n_photos_file:,};  photos with >=1 sampled marking: {int((C.n_marks > 0).sum()):,}")
    zero = int(((C.n_marks == 0) & np.isfinite(C.upload)).sum())
    log(f"photos (sampled) in the photos file but with ZERO markings in the markings file: {zero:,} "
        f"({zero / max(np.isfinite(C.upload).sum(), 1):.1%}) -> if ~0%, the dataset only contains photos that were favorited at least once "
        "(survivorship: popularity is conditioned on >=1 favorite).")
    known = np.isfinite(C.upload)
    if known.any():
        log(f"upload time range: {dt_str(np.nanmin(C.upload))} -> {dt_str(np.nanmax(C.upload))};  last marking: {dt_str(C.t_end)}")
        age_d = (C.t_end - C.upload[known]) / 86400
        log(f"photo age at data end (days): median={np.median(age_d):.0f}, p10={np.quantile(age_d, .1):.0f}, p90={np.quantile(age_d, .9):.0f}")
    for k, nm in (("views", "views (crawl-time total)"), ("comments", "comments (crawl-time total)"),
                  ("favs_meta", "favorites (crawl-time total)")):
        v = C.meta[k][known]
        if len(v) and np.isfinite(v).any():
            REPORT[k] = describe(v, nm)
            log(f"    share equal to 0: {(v == 0).mean():.1%}")
    m, nm = C.meta["favs_meta"], C.n_marks
    ok = known & (m > 0) & (nm > 0)
    if ok.any():
        ratio = nm[ok] / m[ok]
        log(f"observed markings in file / favorites column: median={np.median(ratio):.2f}, p10={np.quantile(ratio, .1):.2f}; "
            f"photos where observed > column: {(nm[ok] > m[ok]).mean():.1%};  Spearman(observed, column)={spear(nm[ok], m[ok]):.3f}")
        log("  -> the favorites column is the photo's total at crawl time (all fans); the marking file only has fans "
            "among the crawled users, so the file under-counts popularity.")
    if len(C.owner_total):
        REPORT["photos_per_owner"] = describe(C.owner_total, "photos per owner (full photo file)")
        s = np.sort(C.owner_total)[::-1]
        k = max(1, int(len(s) * 0.01))
        log(f"top 1% of owners upload {s[:k].sum() / s.sum():.1%} of photos;  owners with exactly 1 photo: {(C.owner_total == 1).mean():.1%}")


# ------------------------------ 3. popularity ------------------------------
def sec_popularity():
    section("3. OBSERVED POPULARITY (markings per photo, whole period)")
    y = C.n_marks[C.n_marks > 0]
    REPORT["n_marks"] = describe(y, "markings/photo")
    log(f"photos with exactly 1 marking: {(y == 1).mean():.1%};  <=3: {(y <= 3).mean():.1%};  >=100: {(y >= 100).mean():.2%}")
    s = np.sort(y)[::-1]
    for q in (0.01, 0.10):
        k = max(1, int(len(s) * q))
        log(f"top {q:.0%} of photos hold {s[:k].sum() / s.sum():.1%} of all markings")
    log(f"Hill tail exponent alpha (xmin=p50, floor 2): {hill_alpha(y, max(np.median(y), 2)):.2f}")


# ------------------------------ 4. time dynamics ---------------------------
def sec_time():
    section("4. TIME DYNAMICS, HORIZON & EARLY-SIGNAL (relative to UPLOAD time)")
    age, valid, p = C.age_h, C.valid, C.p
    if not valid.any():
        log("No markings with a known, non-negative lag -> section skipped.")
        C.has_time = False
        return
    C.has_time = True
    a = age[valid]
    if len(a) > 5_000_000:
        a = rng.choice(a, 5_000_000, replace=False)
    qs = np.quantile(a, [.1, .25, .5, .75, .9, .99])
    log("lag upload->marking (hours): p10={:.2f} p25={:.2f} p50={:.2f} p75={:.1f} p90={:.0f} p99={:.0f}".format(*qs))
    cuts = [(1, "1h"), (24, "24h"), (168, "7d"), (720, "30d"), (2160, "90d"), (8760, "365d")]
    log("share of markings within: " + ", ".join(f"{nm}={np.mean(a <= h):.1%}" for h, nm in cuts))
    REPORT["lag_quantiles_h"] = dict(zip(["p10", "p25", "p50", "p75", "p90", "p99"], qs.tolist()))

    known = np.isfinite(C.upload)
    age_end_h = np.where(known, (C.t_end - C.upload) / 3600.0, np.nan)
    C.age_end_h = age_end_h
    nm_valid = np.bincount(p[valid], minlength=C.P)

    log("\nHorizon table (mature = photo uploaded at least H days before the last marking):")
    rows = []
    for H in HORIZONS_DAYS:
        mat = known & (age_end_h >= H * 24)
        cH = np.bincount(p[valid & (age <= H * 24)], minlength=C.P)
        cap = cH[mat].sum() / max(nm_valid[mat].sum(), 1)
        rows.append(dict(H_days=H, mature_photos=int(mat.sum()),
                         share_of_all_their_markings_within_H=f"{cap:.1%}",
                         share_with_0_markings_in_H=f"{(cH[mat] == 0).mean():.1%}",
                         median_markings_in_H=float(np.median(cH[mat])) if mat.any() else np.nan))
    tab = pd.DataFrame(rows)
    log(tab.to_string(index=False))
    REPORT["horizon_table"] = tab.to_dict("records")

    H = TARGET_HORIZON_DAYS
    mat = known & (age_end_h >= H * 24)
    C.H, C.mature = H, mat
    C.cntH = np.bincount(p[valid & (age <= H * 24)], minlength=C.P)
    log(f"\nTARGET = markings within {H} days of upload; mature photos = {int(mat.sum()):,}")
    y = C.cntH[mat]
    REPORT["target"] = describe(y, f"markings in first {H}d")
    REPORT["target"]["share_zero"] = float((y == 0).mean())
    REPORT["target"]["log_skew"] = float(stats.skew(np.log1p(y))) if len(y) > 2 else float("nan")
    log(f"share with 0: {(y == 0).mean():.1%};  with 1: {(y == 1).mean():.1%};  >=10: {(y >= 10).mean():.1%};  skew after log1p: {REPORT['target']['log_skew']:.2f}")
    if len(y):
        log("candidate classification thresholds (markings in horizon): " +
            ", ".join(f"p{q}: >={np.quantile(y, q / 100):.0f} -> {np.mean(y >= max(np.quantile(y, q / 100), 1)):.1%} positive" for q in (50, 80, 90)))

    log(f"\nEarly signal: windows after upload vs REMAINING markings in (window, {H}d]. "
        "('vs_total' is shown only to illustrate how misleading it is: it is mechanical.)")
    C.early = {}
    rows = []
    for T in sorted(set(EARLY_WINDOWS_H + [FEATURE_WINDOW_H])):
        if T >= H * 24:
            continue
        e = np.bincount(p[valid & (age <= T)], minlength=C.P)
        C.early[T] = e
        em = e[mat]
        rem = C.cntH[mat] - em
        rows.append(dict(window_h=T, share_photos_with_0_early=f"{(em == 0).mean():.1%}",
                         median_early=float(np.median(em)), mean_early=float(em.mean()),
                         spearman_vs_remaining=spear(em, rem), spearman_vs_total_MECHANICAL=spear(em, C.cntH[mat]),
                         mean_remaining=float(rem.mean())))
    et = pd.DataFrame(rows)
    log(et.round(3).to_string(index=False))
    REPORT["early_signal"] = et.to_dict("records")
    C.early_tab = et


# ------------------------------ 5. split -----------------------------------
def sec_split():
    section("5. TEMPORAL SPLIT, DRIFT & COLD-START")
    if not getattr(C, "has_time", False):
        log("skipped (no timing info)")
        return
    mat = C.mature
    idx = np.where(mat)[0]
    up = C.upload[idx]
    q1, q2 = np.quantile(up, [SPLIT[0], SPLIT[0] + SPLIT[1]])
    C.q1, C.q2 = float(q1), float(q2)
    sp = np.full(C.P, -1, dtype=np.int8)
    sp[idx] = np.where(up <= q1, 0, np.where(up <= q2, 1, 2))
    C.split_p = sp
    log(f"split by upload time (mature photos only): cutoffs {dt_str(q1)} and {dt_str(q2)}")
    y = C.cntH
    thr = np.quantile(y[sp == 0], 0.8)
    rows = []
    for k, nm in enumerate(("train", "val", "test")):
        v = y[sp == k]
        rows.append(dict(split=nm, photos=len(v), mean=v.mean(), median=np.median(v), mean_log1p=np.log1p(v).mean(),
                         share_zero=f"{(v == 0).mean():.1%}", share_above_train_p80=f"{(v >= thr).mean():.1%}"))
    log(pd.DataFrame(rows).round(2).to_string(index=False))
    ks = stats.ks_2samp(np.log1p(y[sp == 0])[:500_000], np.log1p(y[sp == 2])[:500_000])
    med_tr, med_te = np.median(y[sp == 0]), np.median(y[sp == 2])
    log(f"KS train vs test (log1p target): D={ks.statistic:.3f}, p={ks.pvalue:.3g};  median train={med_tr:.1f} vs test={med_te:.1f}")
    REPORT["split"] = dict(ks_D=float(ks.statistic), ks_p=float(ks.pvalue), median_train=float(med_tr), median_test=float(med_te))

    sm = sp[C.p]
    seen = np.zeros(C.N, bool)
    seen[C.u[sm == 0]] = True
    ut = C.u[sm == 2]
    if len(ut):
        log(f"COLD-START fans: test-photo markings by users never seen in train photos: {(~seen[ut]).mean():.1%} of markings, "
            f"{(~seen[np.unique(ut)]).mean():.1%} of distinct fans")
        REPORT["cold_fans"] = float((~seen[np.unique(ut)]).mean())
    ow = C.owner
    tr_o = np.zeros(C.N, bool)
    tr_o[ow[(sp == 0) & (ow >= 0)]] = True
    te_o = ow[(sp == 2) & (ow >= 0)]
    if len(te_o):
        log(f"COLD-START owners: test photos whose owner has no train photo: {(~tr_o[te_o]).mean():.1%}")
        REPORT["cold_owners"] = float((~tr_o[te_o]).mean())


# ------------------------------ 6. users -----------------------------------
def sec_users():
    section("6. USER ACTIVITY (markings per fan)")
    mpu = np.bincount(C.u, minlength=C.N)
    mpu = mpu[mpu > 0]
    REPORT["marks_per_user"] = describe(mpu, "markings/user (sampled photos)")
    s = np.sort(mpu)[::-1]
    k = max(1, int(len(s) * 0.01))
    log(f"top 1% of fans make {s[:k].sum() / s.sum():.1%} of markings;  fans with exactly 1 marking: {(mpu == 1).mean():.1%}")
    ow = C.owner[C.owner >= 0]
    log(f"distinct owners among sampled photos: {len(np.unique(ow)):,}")
    dup_key = np.sort(C.u.astype(np.int64) * C.P + C.p)
    log(f"duplicate (user, photo) markings: {int((dup_key[1:] == dup_key[:-1]).sum()):,}")


# ------------------------------ 7. social graph ----------------------------
def sec_graph():
    section("7. SOCIAL GRAPH (flickr-growth)")
    if C.edges is None:
        log("skipped (no growth file)")
        return
    N = C.N
    s, d, et = C.e_src.astype(np.int64), C.e_dst.astype(np.int64), C.e_ts
    log(f"raw edges={C.n_edges_raw:,}; duplicates removed={C.n_edge_dups:,}; self-loops removed={C.n_loops:,}; kept={len(s):,}")
    outd = np.bincount(s, minlength=N)
    ind = np.bincount(d, minlength=N)
    C.outd, C.ind = outd, ind
    C.in_graph = (outd + ind) > 0
    log(f"users in graph: {int(C.in_graph.sum()):,}")
    REPORT["out_degree"] = describe(outd[outd > 0], "out-degree (users with >=1 out-edge)")
    REPORT["in_degree"] = describe(ind[ind > 0], "in-degree  (users with >=1 in-edge)")

    ks = s * N + d     # sorted (edges were deduplicated in sorted-key order)
    rev = d * N + s
    pos = np.searchsorted(ks, rev)
    has = (pos < len(ks)) & (ks[np.minimum(pos, len(ks) - 1)] == rev)
    REPORT["reciprocity"] = float(has.mean())
    log(f"reciprocity (reverse edge exists): {has.mean():.1%}")

    A = coo_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab[C.in_graph])
    sizes = np.sort(sizes[sizes > 0])[::-1]
    log(f"weakly connected components: {len(sizes):,}; giant component = {sizes[0] / C.in_graph.sum():.1%} of graph users")

    fans = np.unique(C.u)
    REPORT["fans_in_graph"] = float(C.in_graph[fans].mean())
    log(f"distinct fans in graph: {REPORT['fans_in_graph']:.1%};  markings by in-graph fans: {C.in_graph[C.u].mean():.1%}")

    ok = np.isfinite(et)
    C.growth_ts_comparable = False
    if ok.any():
        v, cnt = np.unique(et[ok], return_counts=True)
        top = np.argsort(cnt)[::-1][:3]
        log(f"link timestamps: {dt_str(et[ok].min())} -> {dt_str(et[ok].max())};  {len(v):,} distinct values; "
            f"most common value holds {cnt.max() / ok.sum():.1%} of edges (top: " +
            ", ".join(f"{dt_str(v[i])}: {cnt[i]:,}" for i in top) + ")")
        log("  -> a large share on one value means the first-crawl snapshot was stamped with a single date: "
            "real creation time is unknown for those edges, so treat the graph as (almost) static.")
        lo, hi = float(C.ts.min()), float(C.ts.max())
        C.growth_ts_comparable = bool(et[ok].max() > lo and et[ok].min() < hi)
        log(f"link time range overlaps the marking time range: {C.growth_ts_comparable}")
        REPORT["top_date_share"] = float(cnt.max() / ok.sum())
        if C.growth_ts_comparable and hasattr(C, "q1"):
            late = float((et[ok] > C.q1).mean())
            log(f"LEAKAGE: {late:.1%} of dated links are newer than the train/val cutoff -> filter links by date <= prediction time.")
            REPORT["edges_after_cutoff"] = late
    C.followers = ind if FOLLOW_DIRECTION == "src_follows_dst" else outd
    C.following = outd if FOLLOW_DIRECTION == "src_follows_dst" else ind


# ------------------------------ 8. exposure --------------------------------
def sec_exposure():
    section("8. SOCIAL EXPOSURE (descriptive; sampled photos)")
    if C.edges is None or not getattr(C, "has_time", False):
        log("skipped")
        return
    elig = np.where((C.n_marks >= 2) & (C.owner >= 0))[0]
    if len(elig) == 0:
        log("no eligible photos")
        return
    samp = rng.choice(elig, min(EXPOSURE_SAMPLE_PHOTOS, len(elig)), replace=False)
    pm = np.zeros(C.P, bool)
    pm[samp] = True
    sel = pm[C.p]
    m = pd.DataFrame({"f": C.u[sel], "p": C.p[sel], "t": C.ts[sel], "rk": C.rk[sel], "age_h": C.age_h[sel]})
    m["owner"] = C.owner[m.p.values]
    m = m.reset_index(drop=True)
    log(f"sampled {len(samp):,} photos (>=2 markings) -> {len(m):,} markings; fan == owner in {(m.f == m.owner).mean():.2%} of markings")
    um = np.zeros(C.N, bool)
    um[m.f.values] = True
    um[m.owner.values] = True
    keep = um[C.e_src] & um[C.e_dst]
    ed = pd.DataFrame({"src": C.e_src[keep].astype(np.int64), "dst": C.e_dst[keep].astype(np.int64), "ets": C.e_ts[keep]})
    ed["hit"] = True
    log(f"relevant links: {len(ed):,}")
    P = C.P
    key_m = m.f.values.astype(np.int64) * P + m.p.values

    def evaluate(direction):
        fc, ec = ("src", "dst") if direction == "src_follows_dst" else ("dst", "src")
        link = ed[[fc, ec, "ets", "hit"]].rename(columns={fc: "f", ec: "e"})
        mm = m[["f", "owner", "t"]].merge(link.rename(columns={"e": "owner"}), on=["f", "owner"], how="left")
        own = mm.hit.fillna(False).to_numpy(dtype=bool)
        own_tr = own & (mm.ets.to_numpy() <= mm.t.to_numpy())
        a = link[["f", "e", "ets"]].merge(m[["f", "p", "t"]].rename(columns={"t": "tf"}), on="f")
        a = a.merge(m[["f", "p", "t"]].rename(columns={"f": "e", "t": "te"}), on=["e", "p"])
        a = a[a.te < a.tf]
        ka = np.unique(a.f.values.astype(np.int64) * P + a.p.values)
        fan = np.isin(key_m, ka)
        a_tr = a[a.ets <= a.tf]
        kt = np.unique(a_tr.f.values.astype(np.int64) * P + a_tr.p.values)
        fan_tr = np.isin(key_m, kt)
        return own, own_tr, fan, fan_tr

    res = {}
    for dname in ("src_follows_dst", "dst_follows_src"):
        res[dname] = evaluate(dname)
        own, own_tr, fan, fan_tr = res[dname]
        txt = (f"  {dname:>16}: fan follows owner {own.mean():.1%};  fan follows an EARLIER fan {fan.mean():.1%};  "
               f"either {(own | fan).mean():.1%}")
        if C.growth_ts_comparable:
            txt += f"   | time-respecting (link date <= marking): owner {own_tr.mean():.1%}, earlier fan {fan_tr.mean():.1%}"
        log(txt)
    better = max(res, key=lambda k: (res[k][0] | res[k][2]).mean())
    log(f"  -> higher exposure under '{better}' (configured: '{FOLLOW_DIRECTION}')."
        + ("" if better == FOLLOW_DIRECTION else "  The two can be close; verify with the dataset README before flipping."))
    own, own_tr, fan, fan_tr = res[FOLLOW_DIRECTION]
    cat = np.where(own & fan, "both", np.where(own, "owner only", np.where(fan, "earlier fan only", "neither")))
    m["cat"] = cat
    m["rk_b"] = pd.cut(m.rk, [-1, 0, 4, 19, 99, 1e12], labels=["1st", "2-5", "6-20", "21-100", ">100"])
    m["age_b"] = pd.cut(m.age_h, [-1e9, 1, 24, 168, 1e12], labels=["<1h", "1-24h", "1-7d", ">7d"])
    for col, nm in (("rk_b", "marking position within photo"), ("age_b", "time since upload")):
        t = pd.crosstab(m[col], m.cat, normalize="index").round(3)
        log(f"\nexposure type by {nm}:")
        log(t.to_string())
        if col == "rk_b":
            C.exposure_tab = t
    REPORT["exposure"] = dict(owner=float(own.mean()), fan=float(fan.mean()), neither=float((cat == "neither").mean()),
                              best_direction=better)


# ------------------------------ 9. features --------------------------------
def sec_features(photos):
    section("9. LEAK-SAFE FEATURE SIGNAL (+ leakage demonstration)")
    if not getattr(C, "has_time", False):
        log("skipped")
        return
    W, H = FEATURE_WINDOW_H, C.H
    if W not in C.early:
        log(f"FEATURE_WINDOW_H={W}h must be shorter than the target horizon; skipped")
        return
    idx = np.where(C.mature)[0]
    if len(idx) > FEATURE_SAMPLE_PHOTOS:
        idx = rng.choice(idx, FEATURE_SAMPLE_PHOTOS, replace=False)
    early = C.early[W][idx].astype(float)
    remaining = C.cntH[idx] - early
    y = np.log1p(remaining)
    F, kind = {}, {}
    F[f"early_markings_{W}h"], kind[f"early_markings_{W}h"] = early, "ok"
    own = C.owner[idx]
    has_o = own >= 0
    if C.edges is not None:
        fol = C.followers.astype(float)
        F["owner_followers"] = np.where(has_o, fol[np.maximum(own, 0)], np.nan)
        F["owner_following"] = np.where(has_o, C.following.astype(float)[np.maximum(own, 0)], np.nan)
        kind["owner_followers"] = kind["owner_following"] = "ok (static graph)"
        mk = C.valid & (C.age_h <= W)
        ws = np.bincount(C.p[mk], weights=fol[C.u[mk]], minlength=C.P)
        ne = C.early[W].astype(float)
        F["early_fans_mean_followers"] = np.where(ne[idx] > 0, ws[idx] / np.maximum(ne[idx], 1), np.nan)
        kind["early_fans_mean_followers"] = "ok"
    F["owner_prior_photos"], kind["owner_prior_photos"] = C.prior[idx].astype(float), "ok (uploads before this photo)"
    F["owner_total_photos"], kind["owner_total_photos"] = C.total[idx].astype(float), "LEAK-ish (includes later uploads)"
    for k, nm in (("views", "views"), ("comments", "comments"), ("favs_meta", "favorites_col")):
        F[f"{nm}_crawl_total"], kind[f"{nm}_crawl_total"] = C.meta[k][idx].astype(float), "LEAK (measured at crawl)"
    rows = []
    for k, v in F.items():
        rows.append(dict(feature=k, status=kind[k], spearman_vs_log_remaining=spear(v, y)))
    real_ts = np.nanmin(C.upload) > 1e8
    hr = (C.upload[idx] // 3600) % 24
    dw = (C.upload[idx] // 86400) % 7
    rows.append(dict(feature="upload hour (eta^2)" + ("" if real_ts else " [meaningless if clock is anonymized]"),
                     status="ok", spearman_vs_log_remaining=eta_sq(hr, y)))
    rows.append(dict(feature="upload weekday (eta^2)" + ("" if real_ts else " [meaningless if clock is anonymized]"),
                     status="ok", spearman_vs_log_remaining=eta_sq(dw, y)))
    tab = pd.DataFrame(rows)
    log(f"Target = log1p(markings in ({W}h, {H}d]); mature photos n={len(idx):,}; share with 0 early markings: {(early == 0).mean():.1%}")
    log(tab.round(3).to_string(index=False))
    log("(eta^2 rows are variance explained, not correlations.) LEAK rows show how a tempting column would inflate scores.")
    REPORT["feature_signal"] = tab.to_dict("records")
    out = pd.DataFrame({k: v for k, v in F.items()})
    out["photo_idx"] = idx
    out["target_log1p_remaining"] = y
    out["target_total_H"] = C.cntH[idx]
    out["split"] = np.array(["train", "val", "test"])[np.clip(C.split_p[idx], 0, 2)]
    out["upload_ts"] = C.upload[idx]
    if len(out) > SAVE_SAMPLE_ROWS:
        out = out.sample(SAVE_SAMPLE_ROWS, random_state=SEED)
    out.to_csv(os.path.join(OUT_DIR, "photo_table_sample.csv"), index=False)
    log(f"saved {len(out):,} rows -> {OUT_DIR}/photo_table_sample.csv (drop the LEAK columns before modelling)")


# ------------------------------ 10. graph size -----------------------------
def sec_graph_size():
    section("10. HETEROGRAPH SIZE (for R-GCN / HGT)")
    f = max(PHOTO_SAMPLE_MOD, 1)
    rels = []
    if C.edges is not None:
        rels.append(("User", "follows", "User", len(C.e_src)))
    rels += [("User", "favorites", "Photo", C.n), ("Photo", "favorited_by", "User", C.n)]
    n_up = int((C.owner >= 0).sum())
    rels += [("User", "uploads", "Photo", n_up), ("Photo", "uploaded_by", "User", n_up)]
    df = pd.DataFrame(rels, columns=["src", "relation", "dst", "edges_in_this_run"])
    df["edges_full_data_estimate"] = [r[3] if r[1] == "follows" else r[3] * f for r in rels]
    log(f"node types: User (id space {C.N:,}), Photo ({C.P:,} in this run; ~{C.P * f:,} in full data)")
    log(df.to_string(index=False))
    tot = int(df.edges_full_data_estimate.sum())
    log(f"estimated total edges incl. reverse (full data): {tot:,}")
    REPORT["graph"] = dict(total_edges_full=tot, users=C.N, photos_full=C.P * f)
    if getattr(C, "has_time", False) and FEATURE_WINDOW_H in C.early:
        ek = int((C.valid & (C.age_h <= FEATURE_WINDOW_H)).sum())
        log(f"if the model may only see markings from the first {FEATURE_WINDOW_H}h after upload, the favorites graph keeps "
            f"{ek:,} edges ({ek / C.n:.1%} of all markings)")
        REPORT["early_edge_share"] = ek / C.n
    log("Raw data has no node attributes (no image, tags, text, or user profile): features must be engineered or learned.")


# ------------------------------ plots --------------------------------------
def make_plots(photos):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    y = C.n_marks[C.n_marks > 0]
    ax[0, 0].hist(np.log10(y), bins=40, color="#4c72b0")
    ax[0, 0].set(title="Markings per photo (log10, whole period)", xlabel="log10(markings)", ylabel="photos")
    ccdf(ax[0, 1], y, "markings/photo")
    ccdf(ax[0, 1], np.bincount(C.u, minlength=C.N), "markings/fan")
    if len(C.owner_total):
        ccdf(ax[0, 1], C.owner_total, "photos/owner")
    ax[0, 1].set(title="CCDF: popularity & activity", xlabel="count", ylabel="P(X>=x)")
    ax[0, 1].legend()
    if C.edges is not None:
        ccdf(ax[0, 2], C.outd[C.outd > 0], "out-degree")
        ccdf(ax[0, 2], C.ind[C.ind > 0], "in-degree")
        ax[0, 2].set(title="CCDF: social-graph degree", xlabel="degree", ylabel="P(X>=x)")
        ax[0, 2].legend()
    if getattr(C, "has_time", False):
        lag = C.age_h[C.valid]
        if len(lag) > 2_000_000:
            lag = rng.choice(lag, 2_000_000, replace=False)
        ax[1, 0].hist(np.log10(lag + 0.01), bins=60, color="#55a868")
        ax[1, 0].set(title="Lag upload->marking (log10 hours)", xlabel="log10(hours)")
        et = C.early_tab
        ax[1, 1].plot(et.window_h, et.spearman_vs_remaining, marker="o", label="vs remaining (honest)")
        ax[1, 1].plot(et.window_h, et.spearman_vs_total_MECHANICAL, marker="s", ls="--", label="vs total (mechanical)")
        ax[1, 1].set(title=f"Early-marking Spearman (horizon {C.H}d)", xlabel="window (h)", xscale="log", ylim=(-0.2, 1))
        ax[1, 1].legend()
        W = FEATURE_WINDOW_H
        if W in C.early:
            idx = np.where(C.mature)[0]
            idx = rng.choice(idx, min(3000, len(idx)), replace=False)
            ax[1, 2].scatter(C.early[W][idx] + 1, C.cntH[idx] - C.early[W][idx] + 1, s=6, alpha=.35)
            ax[1, 2].set(title=f"Early ({W}h) vs remaining markings", xscale="log", yscale="log",
                         xlabel="early +1", ylabel="remaining +1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_distributions.png"), dpi=110)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    known = np.isfinite(C.upload)
    if known.any():
        mo = pd.to_datetime(C.upload[known], unit="s").to_period("M").astype(str).value_counts().sort_index()
        ax[0].bar(range(len(mo)), mo.values)
        step = max(1, len(mo) // 10)
        ax[0].set_xticks(range(0, len(mo), step))
        ax[0].set_xticklabels(mo.index[::step], rotation=45)
        ax[0].set(title="Sampled photos uploaded per month")
    t = getattr(C, "exposure_tab", None)
    if t is not None:
        t.plot(kind="bar", stacked=True, ax=ax[1], rot=0)
        ax[1].set(title="Exposure type by marking position", ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_upload_time_and_exposure.png"), dpi=110)
    plt.show()


# ------------------------------ design notes -------------------------------
def design_notes():
    section("11. DESIGN NOTES (rule-based; read together with the numbers above)")
    notes = []
    T = REPORT.get("target")
    if T:
        notes.append(f"Target ({C.H}d markings): median {T['median']:.0f}, skew {T['skew']:.1f} (log1p skew {T['log_skew']:.1f}), "
                     f"{T['share_zero']:.0%} zeros. Use log1p regression, or a classification/hurdle formulation if zeros dominate.")
    ht = REPORT.get("horizon_table")
    if ht:
        good = [r for r in ht if float(r["share_of_all_their_markings_within_H"].strip("%")) >= 80 and r["mature_photos"] > 0]
        if good:
            notes.append(f"Smallest horizon capturing >=80% of a photo's observed markings: {good[0]['H_days']} days. "
                         "A horizon below that measures a growth stage, not final popularity - decide which one you want to predict.")
        else:
            notes.append("No tested horizon captures 80% of markings: popularity keeps accumulating for a long time. "
                         "Pick the horizon by business need and keep it identical for every photo.")
    es = REPORT.get("early_signal")
    if es:
        best = max(es, key=lambda r: (r["spearman_vs_remaining"] if np.isfinite(r["spearman_vs_remaining"]) else -9))
        r = best["spearman_vs_remaining"]
        z = best["share_photos_with_0_early"]
        notes.append(f"Early markings predict REMAINING markings with Spearman {r:.2f} at best (window {best['window_h']}h); "
                     f"{z} of photos have no early marking at that window. "
                     + ("Early counts are weak: lean on owner/social features and expect a low ceiling." if r < 0.3 else
                        "Early counts are informative: they should be the core XGBoost feature."))
        notes.append("Ignore 'vs total' correlations; they are mechanical.")
    notes.append("Never use views / comments / favorites columns or total-uploads-per-owner as features: they are crawl-time or future "
                 "information (see LEAK rows in section 9).")
    sp = REPORT.get("split")
    if sp and sp["ks_D"] > 0.1:
        notes.append(f"Target distribution shifts over time (KS D={sp['ks_D']:.2f}; median {sp['median_train']:.0f} -> {sp['median_test']:.0f}): "
                     "use a temporal split and consider recency-normalised targets.")
    elif sp:
        notes.append(f"KS D={sp['ks_D']:.2f} between train and test targets: any statistical significance reflects sample size, the shift is small.")
    if REPORT.get("cold_fans", 0) > 0.2 or REPORT.get("cold_owners", 0) > 0.2:
        notes.append(f"Cold start: {REPORT.get('cold_fans', 0):.0%} of test fans and {REPORT.get('cold_owners', 0):.0%} of test owners are unseen in train: "
                     "avoid pure ID embeddings; prefer degree/activity features and neighbour aggregation (inductive).")
    if C.edges is not None:
        if REPORT.get("top_date_share", 0) > 0.3:
            notes.append(f"{REPORT['top_date_share']:.0%} of links share one timestamp (first-crawl snapshot): treat the social graph as static.")
        if REPORT.get("fans_in_graph", 1) < 0.8:
            notes.append(f"Only {REPORT['fans_in_graph']:.0%} of fans appear in the graph: many users have no 'follows' edges.")
        if REPORT.get("in_degree", {}).get("skew", 0) > 5:
            notes.append("In-degree is extremely skewed: log-transform degree features, cap neighbour fan-outs, use neighbour sampling.")
        ex = REPORT.get("exposure")
        if ex:
            notes.append(f"Social channel: {ex['owner']:.0%} of sampled markings come from fans following the owner, {ex['fan']:.0%} from fans following an earlier fan, "
                         f"{ex['neither']:.0%} have no social path (discovery by search/Explore). Descriptive only; "
                         "compute any exposure feature inside the early window.")
    g = REPORT.get("graph")
    if g and g["total_edges_full"] > 5_000_000:
        notes.append(f"~{g['total_edges_full']:,} edges in the full heterograph: use mini-batch neighbour sampling around target photos "
                     "(NeighborLoader/HGTLoader); full-batch R-GCN/HGT will not fit.")
    if "early_edge_share" in REPORT:
        notes.append(f"Message passing must only use markings up to the prediction time: that keeps {REPORT['early_edge_share']:.1%} of favorite edges. "
                     "Build one graph snapshot per observation window.")
    notes.append("No node attributes exist: initialise Users with log-degree/activity features and Photos with early-window statistics + owner features; "
                 "pure ID embeddings generalise poorly to new photos.")
    notes.append("XGBoost baseline: use photo_table_sample.csv (without LEAK columns); every graph model must beat it on the temporal test split.")
    for i, n in enumerate(notes, 1):
        log(f"{i}. {n}")


# --------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    photos = sec_loading()
    sec_photos()
    sec_popularity()
    sec_time()
    sec_split()
    sec_users()
    sec_graph()
    sec_exposure()
    sec_features(photos)
    sec_graph_size()
    make_plots(photos)
    design_notes()
    log(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in ./{OUT_DIR}/")
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(LOG))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(REPORT, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()