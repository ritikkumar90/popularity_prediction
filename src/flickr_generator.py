#!/usr/bin/env python3
"""
generate_flickr_synthetic.py
============================
Synthetic Flickr-style favorite cascades + follow graph + photo table with *tunable* hidden signals,
so that a strict model hierarchy appears as the observation window grows:

    0 - 24 h : NOISE PHASE   - viral / non-viral photos are statistically identical
    24 - 72 h: SIGNAL PHASE  - three independent, separately tunable signals switch on

        Signal 1  VOLUME     viral photos receive extra raw favorites     -> tabular models (XGBoost)
        Signal 2  CLUSTER    viral photos' early favoriters come from the -> static graph models (R-GCN)
                             same dense social community
        Signal 3  TEMPORAL   "influencer favorites -> its followers       -> time-aware hetero-graph (HGT)
                             favorite shortly AFTER".  Non-viral photos contain the SAME pairs but in
                             the wrong order (followers first), so static structure cannot tell them apart.

OUTPUT (same names / layout as data/raw/flickr: TAB-separated, headerless, gzip):
    <out>/flickr-all-photo-favorite-markings.txt.gz   user_id, photo_id, time          (time = relative seconds)
    <out>/flickr-growth.txt.gz                        user_id, contact_id, date        (date = YYYY-MM-DD; user follows contact)
    <out>/flickr-all-photos.txt.gz                    photo_id, col1, col2, col3, col4, col5
          col1 = upload time (relative seconds)   col2 = owner user id
          col3 / col4 / col5 = views / comments / favorites (INFERRED from the real file; see notes below)
    <out>/_synthetic_meta/photo_labels.csv, manifest.json   ground truth + latent strengths (kept OUTSIDE the 3 files)

NOTES ON THE PHOTOS FILE (inferred from your real data, not from documentation)
    col1 upload time: first favorite >= col1 for 99.9% of real photos, Spearman 0.89 with first-favorite time
    col2 owner:       user-id scale (1..2.2M), ~20 photos per value
    col3/col4/col5:   Spearman with favorite count 0.56 / 0.57 / 0.995; ~0.5% of rows hold the literal \\N
    !! col5 is (almost exactly) the FINAL favorite count, so it leaks the label.  Do not feed col5 to a model.

USAGE
    python generate_flickr_synthetic.py                              # defaults -> ./data/generated/flickr
    python generate_flickr_synthetic.py --proxy-check                # + quick sanity check of the 3-level hierarchy
    python generate_flickr_synthetic.py --n-photos 2000 --set volume_boost=0.6 noise_temporal=2.0
    python generate_flickr_synthetic.py --config my_params.json
    python generate_flickr_synthetic.py --print-config > my_params.json

QUICK GUIDE: which knob moves which model  (all live in the CONTROL PANEL = class Config just below)
    XGBoost level : volume_boost (up = easier), noise_volume (up = harder), count_dispersion (down = harder)
    R-GCN level   : cluster_strength, cluster_base, noise_cluster, p_intra, n_communities
    HGT level     : temporal_strength (up = easier), noise_temporal (up = harder), chain_rate,
                    chain_followers_mean, chain_lag_hours
    ALL models    : softness, shared_noise (label noise common to every model), early_leak (floor of the 0-24h accuracy)
    Size / time   : n_photos, min_size, viral_top_fraction, noise_phase_hours, signal_hours
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd


# ======================================================================================
#                                  CONTROL PANEL
#   Every number below can be changed here, from the command line (--set key=value),
#   or from a JSON file (--config file.json).
# ======================================================================================
@dataclass
class Config:
    # ---------------- output / reproducibility ----------------
    out_dir: str = "./data/generated/flickr"
    seed: int = 42
    votes_filename: str = "flickr-all-photo-favorite-markings.txt.gz"
    friends_filename: str = "flickr-growth.txt.gz"
    photos_filename: str = "flickr-all-photos.txt.gz"

    # ---------------- PHOTOS ("posts") ----------------
    n_photos: int = 5000
    min_size: int = 50                 # smallest nominal final favorite count (real data: 53k photos have >= 50)
    max_size: int = 3000
    size_mode: str = "real"            # real = tail copied from the real full-file counts | lognormal = size_median/size_sigma
    size_median: float = 90.0          # (lognormal mode only)
    size_sigma: float = 0.9            # (lognormal mode only)
    viral_top_fraction: float = 0.50   # label = top X% of photos by FINAL favorites (0.5 -> 'top50', 0.2 -> 'top20')

    # ---------------- time axis ----------------
    t_first: int = 14417               # earliest favorite time in the real file (relative seconds)
    t_end: int = 29023193              # latest favorite time in the real file (~336 days)
    min_observed_days: float = 7.0     # photos start at least this long before t_end (so 0-72h is never cut off)
    noise_phase_hours: float = 24.0    # class-independent before this photo age
    signal_hours: float = 72.0         # all hidden signals live in [noise_phase_hours, signal_hours]
    upload_delay_median_h: float = 2.0 # PLACEHOLDER: median gap upload -> first favorite (not measured yet)
    upload_delay_sigma: float = 1.5    # PLACEHOLDER: log-sd of that gap

    # ---------------- noise-phase volume (class independent) ----------------
    early_mean_A: float = 8.0          # mean favorites in the noise phase
    early_mean_B: float = 5.0          # mean *base* favorites in the signal window
    count_dispersion: float = 8.0      # negative-binomial r. LOWER = noisier counts
    early_leak: float = 0.05           # tiny class-dependent tilt of the noise-phase volume (0 = pure 50% floor)
    noise_arrival_power: float = 0.4   # noise-phase arrival density ~ t^(power-1); 0.4 = front-loaded like real photos
    min_late_votes: int = 8            # every photo gets at least this many favorites after the signal window

    # ---------------- latent label-noise (shared by every model) ----------------
    softness: float = 0.25             # log-size scale of the viral/non-viral transition. Bigger = fuzzier boundary
    shared_noise: float = 0.60         # logit noise common to all three signals (hurts ALL models equally)

    # ---------------- signal 1: VOLUME (tabular) ----------------
    volume_boost: float = 0.90         # viral photos get +X*100% extra base favorites in the signal window
    noise_volume: float = 1.0          # signal-specific logit noise (higher = weaker/noisier)

    # ---------------- signal 2: CLUSTER (static graph) ----------------
    cluster_strength: float = 0.65     # P(a signal-window favoriter comes from the photo's seed community), clearly viral
    cluster_base: float = 0.08         # same for a clearly non-viral photo
    noise_cluster: float = 1.0
    seed_extra_comms: float = 1.0      # seed communities per photo = 1 + Poisson(this)
    n_communities: int = 400
    community_size_sigma: float = 0.7
    p_intra: float = 0.60              # share of follow edges that stay inside a community

    # ---------------- signal 3: TEMPORAL meta-path (HGT) ----------------
    temporal_strength: float = 0.90    # 1 = viral chains always forward / non-viral always reversed; 0 = no signal
    noise_temporal: float = 1.0
    chain_rate: float = 2.5            # mean influencer->followers chains per photo (same for both classes!)
    chain_followers_mean: float = 3.0  # followers per chain = 1 + Poisson(this)
    chain_lag_hours: float = 3.0       # mean delay between influencer and follower favorites
    influencer_min_followers: int = 15 # min #voter-followers to count as an influencer

    # ---------------- late phase (after the signal window) ----------------
    social_share_late: float = 0.35    # share of late favorites that come from a follower of an earlier favoriter
    time_scale_sigma: float = 0.25     # per-photo lognormal stretch of the late timeline

    # ---------------- photo owners ("uploads" edge) ----------------
    owner_pop_exponent: float = 0.8    # owner picked with prob ~ (followers+1)^this
    owner_follower_share: float = 0.10 # share of early favorites that come from the OWNER's followers (class independent)

    # ---------------- user activity ----------------
    n_voters: int = 60000
    act_median: float = 3.0
    act_sigma: float = 2.0
    act_max: int = 3000
    in_graph_activity_tau: float = 1.0 # noise when deciding which voters are in the graph (smaller = heavier voters in graph)
    activity_popularity_rho: float = 0.40

    # ---------------- follow graph ----------------
    n_graph_users: int = 150000
    n_edges: int = 2000000
    id_max: int = 200000               # user ids are drawn from 1..id_max
    voters_in_graph_share: float = 0.967
    out_zero_share: float = 0.157
    out_sigma: float = 2.4
    out_median: float = 1.0
    out_max: int = 4000
    in_sigma: float = 2.2
    in_tail_frac: float = 0.001
    in_tail_alpha: float = 1.2
    reciprocity: float = 0.622
    recip_hub_penalty: float = 1.0     # >0 keeps reciprocal links away from mega-hubs
    recip_same_date_share: float = 0.82  # share of reciprocal pairs created on the same day
    voter_indeg_tilt: float = 1.5      # >0: voters are better-followed users (real: voters' median in-degree 73 vs 1)
    force_top_hubs: int = 1500

    # ---------------- photos-file columns 3-5 (views / comments / favorites) ----------------
    photo_stats_leak: float = 0.60     # 0 = views/comments unrelated to popularity, 1 = almost deterministic
    views_median: float = 1500.0       # PLACEHOLDER (real median is 67 over ALL photos; large photos have far more)
    views_sigma: float = 1.3
    comments_median: float = 6.0       # PLACEHOLDER
    comments_sigma: float = 1.2
    fav_count_extra_rate: float = 0.005  # col5 = final + Poisson(rate*final)
    null_rate: float = 0.0053          # share of photo rows whose cols 3-5 are the literal \N (real: 59,864 / 11.27M)
    photo_id_order: str = "random"     # photo ids: random | time


# ======================================================================================
# Real-data anchors (from your profile report / photos-file check)
# ======================================================================================
def _ts(y, m, d, hh=0, mm=0, ss=0):
    return calendar.timegm((y, m, d, hh, mm, ss))


# photos with >= k favorites in the FULL real file
SIZE_K = np.array([20, 50, 100, 500, 1000, 2998], float)
SIZE_N = np.array([233074, 53332, 14259, 269, 25, 1], float)

# share of final favorites reached by photo age (hours): blend of the p95-99 and p99-100 buckets of the real report
AGE_H = [0, 1, 6, 24, 72, 168, 720, 2160, 8760]
AGE_CDF = [0, .10, .21, .37, .50, .60, .75, .875, 1.0]

# photos first-favorited per week of the real data (48 weekly sums)
WEEKLY = [363, 335, 354, 360, 341, 337, 342, 322, 333, 346, 350, 358, 377, 346, 362, 407, 381, 402, 402, 376, 405,
          381, 391, 450, 462, 386, 426, 427, 447, 493, 464, 418, 455, 469, 438, 453, 459, 440, 425, 482, 495, 565,
          518, 487, 530, 495, 528, 417]

FRIEND_DATE_P = [0, .1, .25, .5, .75, .9, 1.0]
FRIEND_DATE_V = [_ts(2006, 11, 2)] * 4 + [_ts(2007, 2, 4), _ts(2007, 4, 8), _ts(2007, 5, 18)]

PHOTO_ID_MIN, PHOTO_ID_MAX = 14539175, 25807561


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def nb(rng, mean, r):
    mean = np.maximum(mean, 1e-6)
    return rng.negative_binomial(r, r / (r + mean))


def wdraw(rng, cumw, m):
    return np.minimum(np.searchsorted(cumw, rng.random(m) * cumw[-1], side="right"), len(cumw) - 1)


def real_size_quantile(min_size, p):
    """Size at cumulative probability p among real photos with >= min_size favorites."""
    lk, lN = np.log(SIZE_K), np.log(SIZE_N)
    lN_min = np.interp(np.log(min_size), lk, lN)
    target = np.clip(np.log(max(1 - p, 1e-12)) + lN_min, lN[-1], lN_min)
    return float(np.exp(np.interp(target, lN[::-1], lk[::-1])))


# ======================================================================================
# 1. FOLLOW GRAPH
# ======================================================================================
def _fit_outdeg(rng, n, total, sigma, median, dmax):
    z = rng.standard_normal(n)
    lo, hi = -6.0, 9.0
    for _ in range(60):
        mu = (lo + hi) / 2
        s = np.clip(np.round(np.exp(mu + sigma * z)), 1, dmax).sum()
        lo, hi = (mu, hi) if s < total else (lo, mu)
    d = np.clip(np.round(np.exp(mu + sigma * z)), 1, dmax).astype(np.int64)
    diff = int(total - d.sum())
    while diff != 0:
        k = min(abs(diff), n)
        idx = rng.choice(n, k, replace=False)
        if diff > 0:
            ok = idx[d[idx] < dmax]
            d[ok] += 1
        else:
            ok = idx[d[idx] > 1]
            d[ok] -= 1
        diff = int(total - d.sum())
    return d


def build_graph(cfg, rng):
    N, E, K = cfg.n_graph_users, cfg.n_edges, cfg.n_communities
    cw = rng.lognormal(0, cfg.community_size_sigma, K)
    comm = rng.choice(K, N, p=cw / cw.sum())
    order = np.argsort(comm, kind="stable")
    starts = np.searchsorted(comm[order], np.arange(K + 1))

    w = np.exp(cfg.in_sigma * rng.standard_normal(N))
    tail = rng.random(N) < cfg.in_tail_frac
    w[tail] *= rng.pareto(cfg.in_tail_alpha, int(tail.sum())) + 1.0
    cumw = np.cumsum(w)

    n_act = int(round((1 - cfg.out_zero_share) * N))
    active = rng.choice(N, n_act, replace=False)
    is_active = np.zeros(N, bool)
    is_active[active] = True

    R = int(round(cfg.reciprocity * E / 2))
    E_main = E - R
    d = _fit_outdeg(rng, n_act, E_main, cfg.out_sigma, cfg.out_median, cfg.out_max)

    def sample_targets(src):
        n = len(src)
        dst = np.empty(n, np.int64)
        intra = rng.random(n) < cfg.p_intra
        gi = np.flatnonzero(~intra)
        dst[gi] = wdraw(rng, cumw, len(gi))
        ii = np.flatnonzero(intra)
        if len(ii):
            sc = comm[src[ii]]
            o = np.argsort(sc, kind="stable")
            ii, sc = ii[o], sc[o]
            b = np.searchsorted(sc, np.arange(K + 1))
            for k in range(K):
                if b[k] == b[k + 1]:
                    continue
                nodes_k = order[starts[k]:starts[k + 1]]
                cwk = np.cumsum(w[nodes_k])
                dst[ii[b[k]:b[k + 1]]] = nodes_k[wdraw(rng, cwk, b[k + 1] - b[k])]
        return dst

    stubs = np.repeat(active, d)
    rng.shuffle(stubs)
    inactive = np.setdiff1d(np.arange(N), active)
    n1 = min(len(inactive), len(stubs))
    src = [stubs[:n1], stubs[n1:]]
    dst = [rng.permutation(inactive)[:n1], sample_targets(stubs[n1:])]
    src, dst = np.concatenate(src), np.concatenate(dst)

    def uniq(s, t):
        ok = s != t
        key = np.unique(s[ok] * N + t[ok])
        return key // N, key % N

    src, dst = uniq(src, dst)
    # reciprocal pairs: add the reverse of R edges that point to an active node
    cand = np.flatnonzero(is_active[dst])
    indeg0 = np.bincount(dst, minlength=N)
    gk = -np.log1p(indeg0[dst[cand]]) * cfg.recip_hub_penalty + rng.gumbel(size=len(cand))
    pick = cand[np.argsort(-gk)[:min(R, len(cand))]]
    src, dst = uniq(np.concatenate([src, dst[pick]]), np.concatenate([dst, src[pick]]))
    # top-up / trim to the exact edge budget
    for _ in range(8):
        short = E - len(src)
        if short <= 0:
            break
        s_new = rng.choice(active, int(short * 1.3) + 10, p=d / d.sum())
        src, dst = uniq(np.concatenate([src, s_new]), np.concatenate([dst, sample_targets(s_new)]))
    if len(src) > E:
        keep = np.sort(rng.choice(len(src), E, replace=False))
        src, dst = src[keep], dst[keep]
    return dict(src=src, dst=dst, comm=comm, N=N, K=K)


# ======================================================================================
# 2. VOTERS / ACTIVITY / FOLLOWER INDEX
# ======================================================================================
def build_users(cfg, rng, g):
    N = g["N"]
    graph_ids = np.sort(rng.choice(np.arange(1, cfg.id_max + 1), N, replace=False))
    in_set = np.zeros(cfg.id_max + 2, bool)
    in_set[graph_ids] = True
    non_graph_ids = np.flatnonzero(~in_set[:cfg.id_max + 1])
    non_graph_ids = non_graph_ids[non_graph_ids >= 2]

    n_out = min(int(round((1 - cfg.voters_in_graph_share) * cfg.n_voters)), len(non_graph_ids))
    n_in = cfg.n_voters - n_out
    indeg = np.bincount(g["dst"], minlength=N)

    # which graph nodes are voters
    key = cfg.voter_indeg_tilt * np.log1p(indeg) + rng.gumbel(size=N)
    hubs = np.argsort(-indeg)[:cfg.force_top_hubs]
    key[hubs] += 1e9
    sel_nodes = np.argpartition(-key, n_in - 1)[:n_in]

    # activity
    z = rng.standard_normal(cfg.n_voters)
    act = np.clip(np.round(np.exp(np.log(cfg.act_median) + cfg.act_sigma * z)), 1, cfg.act_max)
    score = z + cfg.in_graph_activity_tau * rng.standard_normal(cfg.n_voters)
    in_graph_voters = np.argpartition(-score, n_in - 1)[:n_in]
    in_mask = np.zeros(cfg.n_voters, bool)
    in_mask[in_graph_voters] = True

    # pair in-graph voters with graph nodes: busy voters tend to be better-followed nodes
    pop = np.log1p(indeg[sel_nodes])
    pop = (pop - pop.mean()) / (pop.std() + 1e-9)
    rho = cfg.activity_popularity_rho
    node_score = rho * pop + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_in)
    nodes_sorted = sel_nodes[np.argsort(node_score)]
    voters_sorted = in_graph_voters[np.argsort(z[in_graph_voters])]
    voter_node = -np.ones(cfg.n_voters, np.int64)
    voter_node[voters_sorted] = nodes_sorted

    voter_uid = np.zeros(cfg.n_voters, np.int64)
    voter_uid[in_mask] = graph_ids[voter_node[in_mask]]
    voter_uid[~in_mask] = rng.choice(non_graph_ids, int((~in_mask).sum()), replace=False)

    node2voter = -np.ones(N, np.int64)
    node2voter[voter_node[in_mask]] = np.flatnonzero(in_mask)

    # follower index in VOTER space: followers of voter v that are themselves voters (CSR)
    fa, fb = node2voter[g["src"]], node2voter[g["dst"]]
    m = (fa >= 0) & (fb >= 0)
    fa, fb = fa[m], fb[m]
    vfa, vfb = fa.copy(), fb.copy()
    o = np.argsort(fb, kind="stable")
    fol = fa[o].astype(np.int32)
    ptr = np.concatenate([[0], np.cumsum(np.bincount(fb, minlength=cfg.n_voters))])
    nfol = np.diff(ptr)

    # owner index: voter-followers of every GRAPH node (CSR over graph nodes)
    vs = node2voter[g["src"]]
    mo = vs >= 0
    od, ov = g["dst"][mo], vs[mo]
    oo = np.argsort(od, kind="stable")
    ofol = ov[oo].astype(np.int32)
    optr = np.concatenate([[0], np.cumsum(np.bincount(od, minlength=N))])

    # influencer pool
    infl = np.flatnonzero(nfol >= cfg.influencer_min_followers)
    if len(infl) < 50:
        infl = np.argsort(-nfol)[:max(200, int(0.002 * cfg.n_voters))]
    infl_cum = np.cumsum(np.sqrt(nfol[infl].astype(float)))

    # community pools (voters that sit in the graph)
    vcomm = np.where(voter_node >= 0, g["comm"][np.maximum(voter_node, 0)], -1)
    comm_voters, comm_cum = [], []
    for k in range(g["K"]):
        vs_k = np.flatnonzero(vcomm == k)
        comm_voters.append(vs_k)
        comm_cum.append(np.cumsum(act[vs_k]) if len(vs_k) else np.zeros(0))
    comm_size = np.array([len(v) for v in comm_voters], float)
    seed_p = np.where(comm_size >= 10, comm_size, 0.0)
    seed_cum = np.cumsum(seed_p)

    return dict(graph_ids=graph_ids, voter_uid=voter_uid, voter_node=voter_node, act=act, fol=fol, ptr=ptr,
                nfol=nfol, infl=infl, infl_cum=infl_cum, comm_voters=comm_voters, comm_cum=comm_cum,
                seed_cum=seed_cum, indeg=indeg, ofol=ofol, optr=optr, vfa=vfa, vfb=vfb)


# ======================================================================================
# 3. PHOTOS
# ======================================================================================
def build_photos(cfg, rng, g, U):
    n = cfg.n_photos
    u = (rng.permutation(n) + rng.random(n)) / n
    if cfg.size_mode == "lognormal":
        sizes = np.exp(np.log(cfg.size_median) + cfg.size_sigma * rng.standard_normal(n))
    else:
        lk, lN = np.log(SIZE_K), np.log(SIZE_N)
        lN_min = np.interp(np.log(cfg.min_size), lk, lN)
        target = np.clip(np.log(np.maximum(1.0 - u, 1e-12)) + lN_min, lN[-1], lN_min)
        sizes = np.exp(np.interp(target, lN[::-1], lk[::-1]))
    sizes = np.clip(np.round(sizes), cfg.min_size, cfg.max_size).astype(np.int64)

    # first-favorite time of every photo (weekly profile of the real data)
    nd = int((cfg.t_end - cfg.t_first) // 86400)
    dw = np.repeat(np.array(WEEKLY, float), 7)[:nd]
    max_start = cfg.t_end - cfg.min_observed_days * 86400
    dw = np.where(cfg.t_first + np.arange(nd) * 86400 <= max_start, dw, 0.0)
    day = rng.choice(nd, n, p=dw / dw.sum())
    start = (cfg.t_first + day * 86400 + rng.integers(0, 86400, n)).astype(np.int64)
    delay = np.maximum(60.0, np.exp(np.log(cfg.upload_delay_median_h * 3600) +
                                    cfg.upload_delay_sigma * rng.standard_normal(n)))
    upload = np.maximum(1, start - np.round(delay).astype(np.int64))

    # owners
    wown = (U["indeg"].astype(float) + 1.0) ** cfg.owner_pop_exponent
    owner_node = rng.choice(len(wown), n, p=wown / wown.sum())

    # photo ids
    ids = np.unique(rng.integers(PHOTO_ID_MIN, PHOTO_ID_MAX + 1, int(n * 1.3) + 10))
    rng.shuffle(ids)
    ids = ids[:n]
    if cfg.photo_id_order == "time":
        ids = np.sort(ids)[np.argsort(np.argsort(start, kind="stable"), kind="stable")]

    # latent strengths
    logs = np.log(sizes.astype(float))
    thr = np.quantile(logs, 1 - cfg.viral_top_fraction)
    viral = logs >= thr
    base = (logs - thr) / cfg.softness
    e_sh = rng.standard_normal(n) * cfg.shared_noise
    a_vol = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_volume)
    a_clu = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_cluster)
    a_tmp = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_temporal)
    return dict(sizes=sizes, start=start, upload=upload, owner_node=owner_node, ids=ids, viral=viral,
                a_vol=a_vol, a_clu=a_clu, a_tmp=a_tmp)


# ======================================================================================
# 4. FAVORITES
# ======================================================================================
def build_votes(cfg, rng, U, S):
    nV = cfg.n_voters
    act, fol, ptr, nfol = U["act"], U["fol"], U["ptr"], U["nfol"]
    ofol, optr = U["ofol"], U["optr"]
    cumw = np.cumsum(act)
    used = np.zeros(nV, bool)
    onset, sig = cfg.noise_phase_hours, cfg.signal_hours
    F_sig = float(np.interp(sig, AGE_H, AGE_CDF))
    n = cfg.n_photos
    width = max(sig - onset, 1e-6)

    ts_l, v_l, s_l = [], [], []
    story_off = np.zeros(n + 1, np.int64)
    n_social_total = 0

    def draw_unique(need):
        out, got = [], 0
        while got < need:
            cand = wdraw(rng, cumw, int((need - got) * 1.3) + 10)
            cand = cand[~used[cand]]
            _, idx = np.unique(cand, return_index=True)
            cand = cand[np.sort(idx)][:need - got]
            used[cand] = True
            out.append(cand)
            got += len(cand)
            if len(cand) == 0 and got < need and used.all():
                break
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    for j in range(n):
        size = int(S["sizes"][j])
        a_v, a_c, a_t = S["a_vol"][j], S["a_clu"][j], S["a_tmp"][j]
        own = int(S["owner_node"][j])
        Fo = ofol[optr[own]:optr[own + 1]]

        # ---------------- noise phase [0, onset): identical for both classes (up to early_leak) -------------
        mA = cfg.early_mean_A * (1 + cfg.early_leak * (a_v - 0.5))
        nA = int(nb(rng, mA, cfg.count_dispersion))
        age_A = onset * rng.random(nA) ** (1 / cfg.noise_arrival_power)
        usr_A = wdraw(rng, cumw, nA)
        if cfg.owner_follower_share > 0 and len(Fo) and nA:
            k = rng.binomial(nA, cfg.owner_follower_share)
            if k:
                usr_A[:k] = Fo[rng.integers(0, len(Fo), k)]

        # ---------------- signal window [onset, sig) ---------------------------------------------------------
        boost = cfg.volume_boost * a_v                     # signal 1 (volume)
        nB = int(nb(rng, cfg.early_mean_B * (1 + boost), cfg.count_dispersion))
        n_extra = rng.binomial(nB, boost / (1 + boost)) if nB else 0
        age_B = np.concatenate([onset + width * rng.random(nB - n_extra),
                                onset + width * np.sqrt(rng.random(n_extra))])
        usr_B = wdraw(rng, cumw, nB)
        if cfg.owner_follower_share > 0 and len(Fo) and nB:
            k = rng.binomial(nB, cfg.owner_follower_share)
            if k:
                usr_B[:k] = Fo[rng.integers(0, len(Fo), k)]
        # signal 2 (cluster): some window favoriters come from the photo's seed communities
        p_clu = cfg.cluster_base + (cfg.cluster_strength - cfg.cluster_base) * a_c
        flag = rng.random(nB) < p_clu
        if flag.any():
            n_seed = min(1 + rng.poisson(cfg.seed_extra_comms), 4)
            seeds = wdraw(rng, U["seed_cum"], n_seed)
            which = rng.integers(0, n_seed, int(flag.sum()))
            pos = np.flatnonzero(flag)
            for si in range(n_seed):
                sel = pos[which == si]
                cv, cc = U["comm_voters"][seeds[si]], U["comm_cum"][seeds[si]]
                if len(cv) and len(sel):
                    # without replacement: repeated voters would be deduplicated and silently cancel the volume boost
                    k = min(len(sel), len(cv))
                    pw = np.diff(cc, prepend=0.0)
                    usr_B[sel[:k]] = cv[rng.choice(len(cv), k, replace=False, p=pw / pw.sum())]

        # signal 3 (temporal meta-path): influencer -> followers.  direction decided by a_t
        q_fwd = 0.5 + 0.5 * cfg.temporal_strength * (2 * a_t - 1)
        n_ch = 1 + rng.poisson(max(cfg.chain_rate - 1, 0.0))
        age_C, usr_C = [], []
        for _ in range(n_ch):
            i = U["infl"][wdraw(rng, U["infl_cum"], 1)[0]]
            F = fol[ptr[i]:ptr[i + 1]]
            k = min(len(F), 1 + rng.poisson(cfg.chain_followers_mean))
            if k == 0:
                continue
            fs = rng.choice(F, k, replace=False)
            t0 = onset + 0.05 * width + rng.random() * 0.80 * width
            lags = rng.exponential(cfg.chain_lag_hours, k) + 1.0 / 3600
            if rng.random() < q_fwd:                        # forward: influencer first, followers after
                tf = np.minimum(t0 + lags, sig - 1e-4)
            else:                                           # reversed: followers first, influencer after
                tf = np.maximum(t0 - lags, onset + 1e-4)
            age_C += [t0] + list(tf)
            usr_C += [i] + list(fs)

        age_E = np.concatenate([age_A, age_B, np.array(age_C, float)])
        usr_E = np.concatenate([usr_A, usr_B, np.array(usr_C, np.int64)]).astype(np.int64)
        o = np.argsort(age_E, kind="stable")
        age_E, usr_E = age_E[o], usr_E[o]
        _, first = np.unique(usr_E, return_index=True)
        keep = np.sort(first)
        age_E, usr_E = age_E[keep], usr_E[keep]
        age_E = age_E - age_E.min()                         # first favorite defines photo start
        nE = len(usr_E)
        used[usr_E] = True

        # ---------------- late phase (after the signal window) -------------------------------------------------
        n_late = max(size - nE, cfg.min_late_votes)
        tau = float(np.exp(cfg.time_scale_sigma * rng.standard_normal()))
        horizon = (cfg.t_end - int(S["start"][j])) / 3600.0
        age_max_unscaled = sig + max(horizon - sig, 1.0) / tau
        v_max = (np.interp(age_max_unscaled, AGE_H, AGE_CDF) - F_sig) / (1 - F_sig)
        v = rng.random(n_late) * v_max
        age_L = np.interp(F_sig + (1 - F_sig) * v, AGE_CDF, AGE_H)
        age_L = np.sort(sig + (age_L - sig) * tau)

        base_u = rng.permutation(draw_unique(n_late))
        n_late = len(base_u)
        age_L = age_L[:n_late]
        users_L = base_u.copy()
        if cfg.social_share_late > 0 and n_late:
            all_base = np.concatenate([usr_E, base_u])
            social = rng.random(n_late) < cfg.social_share_late
            is_base = np.concatenate([np.ones(nE, bool), ~social])
            cum_base = np.cumsum(is_base)
            base_pos = np.flatnonzero(is_base)
            sp = np.flatnonzero(social)
            gpos = nE + sp
            nbefore = cum_base[gpos - 1]
            child = -np.ones(len(sp), np.int64)
            for _ in range(3):
                need = child < 0
                if not need.any():
                    break
                par = all_base[base_pos[np.minimum((rng.random(need.sum()) * nbefore[need]).astype(np.int64),
                                                   np.maximum(nbefore[need] - 1, 0))]]
                dg = ptr[par + 1] - ptr[par]
                okp = dg > 0
                c = -np.ones(len(par), np.int64)
                c[okp] = fol[ptr[par[okp]] + (rng.random(okp.sum()) * dg[okp]).astype(np.int64)]
                sub = np.flatnonzero(need)
                child[sub] = c
            okc = child >= 0
            okc[okc] = ~used[child[okc]]
            idx = np.flatnonzero(okc)
            _, fi = np.unique(child[idx], return_index=True)
            idx = idx[fi]
            users_L[sp[idx]] = child[idx]
            used[child[idx]] = True
            n_social_total += len(idx)

        users = np.concatenate([usr_E, users_L])
        ages = np.concatenate([age_E, age_L])
        used[users] = False
        used[base_u] = False
        ts = S["start"][j] + np.round(ages * 3600).astype(np.int64)
        ts_l.append(np.minimum(ts, cfg.t_end))
        v_l.append(users.astype(np.int32))
        s_l.append(np.full(len(users), j, np.int32))
        story_off[j + 1] = story_off[j] + len(users)
        if (j + 1) % 1000 == 0:
            log(f"  photos {j + 1}/{n}")

    ts = np.concatenate(ts_l)
    vv = np.concatenate(v_l)
    ss = np.concatenate(s_l)

    # every designated voter must have favorited at least once
    cnt = np.bincount(vv, minlength=nV)
    zero = np.flatnonzero(cnt == 0)
    if len(zero):
        sp_ = S["sizes"] / S["sizes"].sum()
        sj = rng.choice(n, len(zero), p=sp_)
        nrows = np.diff(story_off)[sj]
        row = story_off[sj] + (rng.random(len(zero)) * nrows).astype(np.int64)
        ts = np.concatenate([ts, np.minimum(ts[row] + rng.integers(0, 600, len(zero)), cfg.t_end)])
        vv = np.concatenate([vv, zero.astype(np.int32)])
        ss = np.concatenate([ss, sj.astype(np.int32)])
    return ts, vv, ss, n_social_total


# ======================================================================================
# 5. FILE TABLES
# ======================================================================================
def build_friend_table(cfg, rng, g, U):
    src, dst, N = g["src"], g["dst"], g["N"]
    E = len(src)
    key = src * N + dst
    rkey = dst * N + src
    so = np.argsort(key)
    skey = key[so]
    pos = np.minimum(np.searchsorted(skey, rkey), E - 1)
    has_rev = skey[pos] == rkey
    partner = np.where(has_rev, so[pos], -1)

    date = np.interp(rng.random(E), FRIEND_DATE_P, FRIEND_DATE_V)
    date = (date // 86400) * 86400
    fwd = np.flatnonzero(has_rev & (src < dst))
    gap = np.clip(rng.lognormal(np.log(1.0), 2.5, len(fwd)), 1e-4, 700) * 86400
    same = rng.random(len(fwd)) < cfg.recip_same_date_share
    date[partner[fwd]] = np.where(same, date[fwd], np.minimum(date[fwd] + gap, FRIEND_DATE_V[-1]))
    date = (date // 86400) * 86400
    uid = U["graph_ids"]
    df = pd.DataFrame({"user_id": uid[src], "friend_id": uid[dst],
                       "date": pd.to_datetime(date.astype(np.int64), unit="s").strftime("%Y-%m-%d")})
    df = df.sort_values(["user_id", "friend_id"], kind="stable")
    return df, float(has_rev.mean())


def nullable_int(a, mask):
    s = pd.array(np.asarray(a).astype(np.int64), dtype="Int64")
    s[mask] = pd.NA
    return s


def build_photo_table(cfg, rng, S, final, U):
    n = cfg.n_photos
    lf = np.log(np.maximum(final, 1).astype(float))
    z = (lf - lf.mean()) / (lf.std() + 1e-9)
    leak = cfg.photo_stats_leak
    lat_v = leak * z + np.sqrt(max(1 - leak ** 2, 0)) * rng.standard_normal(n)
    lat_c = leak * z + np.sqrt(max(1 - leak ** 2, 0)) * rng.standard_normal(n)
    views = np.maximum(1, np.round(np.exp(np.log(cfg.views_median) + cfg.views_sigma * lat_v)))
    comments = np.maximum(0, np.round(np.exp(np.log(cfg.comments_median) + cfg.comments_sigma * lat_c) - 0.5))
    favs = final + rng.poisson(cfg.fav_count_extra_rate * final)
    null = rng.random(n) < cfg.null_rate
    df = pd.DataFrame({0: S["ids"], 1: S["upload"], 2: U["graph_ids"][S["owner_node"]],
                       3: nullable_int(views, null), 4: nullable_int(comments, null), 5: nullable_int(favs, null)})
    return df.sort_values(0, kind="stable")


# ======================================================================================
# 6. OPTIONAL: PROXY CHECK OF THE 3-LEVEL HIERARCHY
# ======================================================================================
def proxy_check(cfg, U, ts, vv, ss, lab):
    """Cross-validated gradient-boosting accuracies on hand-made features.  A SANITY CHECK only: your real
    XGBoost / R-GCN / HGT will land somewhere near, not exactly on, these numbers."""
    try:
        from scipy import sparse
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception as ex:                                   # pragma: no cover
        log(f"proxy check skipped (needs scipy + scikit-learn): {ex}")
        return
    nV, n = cfg.n_voters, cfg.n_photos
    A = sparse.csr_matrix((np.ones(len(U["vfa"]), np.int8), (U["vfa"], U["vfb"])), shape=(nV, nV))
    order = np.lexsort((ts, ss))
    ts_o, vv_o = ts[order], vv[order]
    cnt = np.bincount(ss, minlength=n)
    off = np.r_[0, np.cumsum(cnt)]
    sig, nz = cfg.signal_hours, cfg.noise_phase_hours
    rows = []
    for j in range(n):
        t = ts_o[off[j]:off[j + 1]]
        v = vv_o[off[j]:off[j + 1]]
        age = (t - t[0]) / 3600.0
        n6, n12, nn, ns = (age <= 6).sum(), (age <= 12).sum(), (age <= nz).sum(), (age <= sig).sum()
        sel = age <= sig
        V, T = v[sel], age[sel]
        sub = A[V][:, V].tocoo()
        r, c = sub.row, sub.col
        d = T[r] - T[c]                                   # > 0: the follower favorited AFTER the person they follow
        fwd, rev = int((d > 0).sum()), int((d < 0).sum())
        lag = 2.0 * cfg.chain_lag_hours
        fc, rc = int(((d > 0) & (d <= lag)).sum()), int(((d < 0) & (d >= -lag)).sum())
        npair = len(r)
        rows.append([n6, n12, nn, ns, ns - nn, npair, npair / max(ns, 1), fwd, rev, fwd / (fwd + rev) if fwd + rev else 0.5,
                     fc, rc, fc / (fc + rc) if fc + rc else 0.5])
    X = np.array(rows, float)
    y = np.asarray(lab).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.seed)

    def acc(cols):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, random_state=cfg.seed)
        return float(cross_val_score(m, X[:, cols], y, cv=cv, scoring="accuracy").mean())

    print("\n================ PROXY CHECK (5-fold CV, gradient boosting; sanity check, not your models) ================")
    print(f"label = top {cfg.viral_top_fraction:.0%} by final favorites   majority-class accuracy = {max(y.mean(), 1 - y.mean()):.3f}")
    floor = float(np.mean(cnt - X[:, 3] <= cfg.min_late_votes))
    print(f"share of photos where the late-phase floor binds (early volume leaks into the label) = {floor:.1%}  (keep near 0%)")
    print(f"mean share of final favorites already inside 0-{sig:g}h = {np.mean(X[:, 3] / cnt):.2f}  (real big photos: ~0.50)")
    print(f"  0-{nz:g}h   counts only                     : {acc([0, 1, 2]):.3f}   (should sit near 0.5)")
    print(f"  0-{sig:g}h  counts                           : {acc([0, 1, 2, 3, 4]):.3f}   (XGBoost level)")
    print(f"  0-{sig:g}h  counts + static graph            : {acc([0, 1, 2, 3, 4, 5, 6]):.3f}   (R-GCN level)")
    print(f"  0-{sig:g}h  counts + graph + time order      : {acc(list(range(13))):.3f}   (HGT level)")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with parameter overrides")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-photos", type=int, help="number of photos (default 5000)")
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--proxy-check", action="store_true", help="print quick sanity accuracies for the 3 signal levels")
    args = ap.parse_args()

    cfg = Config()
    ov = {}
    if args.config:
        ov.update(json.load(open(args.config)))
    for kv in args.set:
        k, v = kv.split("=", 1)
        ov[k] = v
    if args.out:
        ov["out_dir"] = args.out
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.n_photos is not None:
        ov["n_photos"] = args.n_photos
    types = {f.name: f.type for f in fields(Config)}
    for k, v in ov.items():
        if k not in types:
            sys.exit(f"unknown parameter: {k}")
        t = {"int": int, "float": float, "bool": lambda s: str(s).lower() in ("1", "true", "yes"),
             "str": str}[types[k] if isinstance(types[k], str) else types[k].__name__]
        setattr(cfg, k, t(v))
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2))
        return

    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    meta = os.path.join(cfg.out_dir, "_synthetic_meta")
    os.makedirs(meta, exist_ok=True)

    log("building follow graph ...")
    g = build_graph(cfg, rng)
    log(f"  edges={len(g['src']):,}  nodes={g['N']:,}")
    log("building users ...")
    U = build_users(cfg, rng, g)
    log(f"  influencers={len(U['infl']):,}")
    log("building photos ...")
    S = build_photos(cfg, rng, g, U)
    log("building favorites ...")
    ts, vv, ss, n_social = build_votes(cfg, rng, U, S)
    log(f"  favorites={len(ts):,}  social-follow share of late favorites={n_social / len(ts):.1%}")

    n = cfg.n_photos
    final = np.bincount(ss, minlength=n)
    order = np.lexsort((rng.random(n), final))
    lab = np.zeros(n, int)
    lab[order[n - int(round(cfg.viral_top_fraction * n)):]] = 1

    votes = pd.DataFrame({"u": U["voter_uid"][vv], "p": S["ids"][ss], "t": ts})
    votes = votes.sort_values(["u", "t"], kind="stable")
    log("building follow table ...")
    fr, recip = build_friend_table(cfg, rng, g, U)
    photos = build_photo_table(cfg, rng, S, final, U)

    log("writing files ...")
    kw = dict(sep="\t", header=False, index=False, lineterminator="\n", na_rep="\\N")
    votes.to_csv(os.path.join(cfg.out_dir, cfg.votes_filename), **kw)
    fr.to_csv(os.path.join(cfg.out_dir, cfg.friends_filename), **kw)
    photos.to_csv(os.path.join(cfg.out_dir, cfg.photos_filename), **kw)
    pd.DataFrame({"photo_id": S["ids"], "final_votes": final, "viral_label": lab, "planned_viral": S["viral"].astype(int),
                  "a_volume": S["a_vol"].round(4), "a_cluster": S["a_clu"].round(4), "a_temporal": S["a_tmp"].round(4),
                  "first_vote_time": S["start"], "upload_time": S["upload"],
                  "owner_id": U["graph_ids"][S["owner_node"]]}).sort_values("photo_id").to_csv(
        os.path.join(meta, "photo_labels.csv"), index=False)
    json.dump({"synthetic": True, "generator": "generate_flickr_synthetic.py", "params": asdict(cfg)},
              open(os.path.join(meta, "manifest.json"), "w"), indent=2)

    # ---------------- quick self-check against the real profile ----------------
    pv = np.bincount(U["voter_uid"][vv], minlength=int(U["voter_uid"].max()) + 1)
    pv = pv[pv > 0]
    q = lambda a, p: np.quantile(a, p).round(1).tolist()
    od = np.bincount(g["src"], minlength=g["N"])
    idg = U["indeg"]
    exp_q = [round(real_size_quantile(cfg.min_size, p)) for p in (0, .1, .5, .9, .99, 1 - 1e-9)]
    print("\n================ SELF-CHECK (real-like target in brackets) ================")
    print(f"favorites={len(ts):,}  photos={n:,}  voters={len(pv):,}  edges={len(g['src']):,}")
    print(f"favorites/photo q(0,10,50,90,99,100) = {q(final, [0, .1, .5, .9, .99, 1])}  [~{exp_q}]  (real photos with >= {cfg.min_size})")
    print(f"favorites/user  q(10,50,90,99) = {q(pv, [.1, .5, .9, .99])}  one-vote share={np.mean(pv == 1):.1%}   (real full file: [1,5,115,1167], 28.3%; subset differs)")
    print(f"out-degree zero={np.mean(od == 0):.1%} [15.7%]  pos-only q(50,90,99)={q(od[od > 0], [.5, .9, .99])} [1,18,310]")
    print(f"in-degree  zero={np.mean(idg == 0):.1%} [2.5%]   pos-only q(50,90,99)={q(idg[idg > 0], [.5, .9, .99])} [1,14,281]  max={idg.max():,}")
    vin = idg[U["voter_node"][U["voter_node"] >= 0]]
    print(f"in-degree of in-graph voters q(25,50,90) = {q(vin, [.25, .5, .9])} [16,73,545]")
    print(f"reciprocity={recip:.1%} [62.2%]   voters in graph={np.mean(U['voter_node'] >= 0):.1%} [96.7%]")
    t_sorted = np.sort(ts)
    print(f"time range {t_sorted[0]:,} -> {t_sorted[-1]:,} s  ({(t_sorted[-1] - t_sorted[0]) / 86400:.0f} days) [14,417 -> 29,023,193]")
    print(f"label: top {cfg.viral_top_fraction:.0%} by final favorites -> threshold {np.sort(final)[n - int(round(cfg.viral_top_fraction * n))]} favorites")
    print(f"output: {os.path.abspath(cfg.out_dir)}")

    if args.proxy_check:
        proxy_check(cfg, U, ts, vv, ss, lab)


if __name__ == "__main__":
    main()
