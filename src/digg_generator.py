#!/usr/bin/env python3
"""
dig_generator.py
================
Synthetic Digg-2009-style vote cascades + follower graph optimized for 
Heterogeneous Graph Neural Networks (R-GCN, HGT).

Adjustments for GNNs:
- High voter-in-graph overlap (85%) to prevent isolated nodes during message passing.
- Stronger clustering and tighter community bounds (R-GCN signal).
- Heavier temporal influencer->follower chains (HGT signal).
- Reduced pure volume tabular signal so graph structural features dominate.
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
# ALL PARAMETERS (CONFIGURATION)
# ======================================================================================
@dataclass
class Config:
    # ---------------- output / reproducibility ----------------
    out_dir: str = "./data/generated/digg"
    seed: int = 42
    scale: float = 1.0                 
    size_scale: float = 1.0            
    size_adjust: float = 0.98          
    votes_filename: str = "digg_votes1.csv"
    friends_filename: str = "digg_friends.csv"
    friends_sort: str = "date"         
    story_id_order: str = "random"     
    emulate_artifacts: bool = True     

    # ---------------- scale (real values) ----------------
    n_stories: int = 3553
    n_voters: int = 139409
    n_graph_users: int = 279630
    n_edges: int = 1731652
    id_max: int = 336225
    # GNN OPTIMIZATION: Increased from 0.594 to 0.85 to prevent isolated nodes in message passing
    voters_in_graph_share: float = 0.850 

    # ---------------- label ----------------
    viral_top_fraction: float = 0.50   

    # ---------------- phases ----------------
    noise_phase_hours: float = 2.0     
    signal_hours: float = 4.0          

    # ---------------- noise-phase vote volume (class independent) ----------------
    early_mean_A: float = 33.0         
    early_mean_B: float = 17.0         
    count_dispersion: float = 8.0      
    early_leak: float = 0.05           

    # ---------------- latent label-noise (shared by every model) ----------------
    softness: float = 0.30             
    shared_noise: float = 0.60         

    # ---------------- signal 1: VOLUME (tabular) ----------------
    # GNN OPTIMIZATION: Reduced volume boost (0.90 -> 0.40) so tabular models struggle more
    volume_boost: float = 0.40         
    noise_volume: float = 1.5          

    # ---------------- signal 2: CLUSTER (static graph / R-GCN) ----------------
    # GNN OPTIMIZATION: Boosted cluster strength and reduced noise for R-GCN detectability
    cluster_strength: float = 0.85     
    cluster_base: float = 0.05         
    noise_cluster: float = 0.5         
    seed_extra_comms: float = 1.0      
    n_communities: int = 800
    community_size_sigma: float = 0.7
    # GNN OPTIMIZATION: Tighter community boundaries (0.60 -> 0.80)
    p_intra: float = 0.80              

    # ---------------- signal 3: TEMPORAL meta-path (HGT) ----------------
    # GNN OPTIMIZATION: Boosted temporal directionality and chain frequency for HGT attention
    temporal_strength: float = 0.98    
    noise_temporal: float = 0.4        
    chain_rate: float = 6.0            # Increased from 3.0
    chain_followers_mean: float = 8.0  # Increased from 4.0
    chain_lag_minutes: float = 6.0     
    influencer_min_followers: int = 25 

    # ---------------- late phase (after signal window) ----------------
    # GNN OPTIMIZATION: More cascade exposure in the late phase
    social_share_late: float = 0.55    
    time_scale_sigma: float = 0.25     

    # ---------------- user activity ----------------
    act_median: float = 4.0
    act_sigma: float = 1.889
    act_max: int = 3415
    in_graph_activity_tau: float = 1.0 
    activity_popularity_rho: float = 0.40

    # ---------------- friend graph ----------------
    out_zero_share: float = 0.745
    out_sigma: float = 2.80
    out_median: float = 3.0
    out_max: int = 800                 
    in_sigma: float = 2.50
    in_tail_frac: float = 0.0008
    in_tail_alpha: float = 1.00
    reciprocity: float = 0.212
    recip_hub_penalty: float = 1.5     
    mutual_flag_share: float = 0.512
    voter_indeg_tilt: float = 0.50
    force_top_hubs: int = 3000

# ======================================================================================
# CONSTANTS & DISTRIBUTIONS
# ======================================================================================
def _ts(y, m, d, hh=0, mm=0, ss=0):
    return calendar.timegm((y, m, d, hh, mm, ss))

T_START = _ts(2009, 5, 31, 11, 43, 41)
T_END = _ts(2009, 7, 5, 11, 29, 52)

STORY_Q_P = [0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99, .995, .999, 1.0]
STORY_Q_V = [122, 201.5, 248.6, 285, 343.4, 400, 452, 527, 609, 740, 1070, 1784, 2664, 4822, 5785, 7919, 24070]
AGE_H = [0, .25, 1, 2, 4, 6, 10.5, 11.5, 12, 24, 72, 168, 336, 840]
AGE_CDF = [0, .010, .036, .060, .098, .135, .172, .300, .333, .809, .973, .9917, .9969, 1.0]
STORIES_PER_DAY = [16, 100, 140, 145, 120, 99, 63, 85, 110, 128, 122, 119, 91, 74, 96, 153, 126, 134,
                   144, 89, 93, 84, 138, 134, 156, 124, 97, 62, 102, 131, 140, 111, 27]
HOUR_SHARE_UTC = [0.045, 0.045, 0.045, 0.044, 0.042, 0.038, 0.032, 0.026, 0.021, 0.019, 0.019, 0.023,
                  0.03, 0.04, 0.046, 0.052, 0.057, 0.059, 0.057, 0.055, 0.054, 0.052, 0.05, 0.048]
FRIEND_DATE_P = [0, .01, .05, .10, .25, .50, .75, .90, 1.0]
FRIEND_DATE_V = [_ts(2006, 6, 1), _ts(2006, 12, 1), _ts(2007, 6, 15), _ts(2007, 10, 31, 13, 54, 33),
                 _ts(2008, 4, 21, 13, 9, 6), _ts(2008, 10, 24, 2, 51, 10), _ts(2009, 3, 26, 21, 35, 0),
                 _ts(2009, 6, 3, 14, 31, 4), _ts(2009, 7, 8, 6, 0, 5)]

# ======================================================================================
# GENERATOR FUNCTIONS
# ======================================================================================
def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def nb(rng, mean, r):
    mean = np.maximum(mean, 1e-6)
    return rng.negative_binomial(r, r / (r + mean))

def wdraw(rng, cumw, m):
    return np.minimum(np.searchsorted(cumw, rng.random(m) * cumw[-1], side="right"), len(cumw) - 1)

def _fit_outdeg(rng, n, total, sigma, median, dmax):
    z = rng.standard_normal(n)
    lo, hi = -3.0, 9.0
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
    cand = np.flatnonzero(is_active[dst])
    indeg0 = np.bincount(dst, minlength=N)
    gk = -np.log1p(indeg0[dst[cand]]) * cfg.recip_hub_penalty + rng.gumbel(size=len(cand))
    pick = cand[np.argsort(-gk)[:min(R, len(cand))]]
    src, dst = uniq(np.concatenate([src, dst[pick]]), np.concatenate([dst, src[pick]]))
    
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

    key = cfg.voter_indeg_tilt * np.log1p(indeg) + rng.gumbel(size=N)
    hubs = np.argsort(-indeg)[:cfg.force_top_hubs]
    key[hubs] += 1e9
    sel_nodes = np.argpartition(-key, n_in - 1)[:n_in]

    z = rng.standard_normal(cfg.n_voters)
    act = np.clip(np.round(np.exp(np.log(cfg.act_median) + cfg.act_sigma * z)), 1, cfg.act_max)
    score = z + cfg.in_graph_activity_tau * rng.standard_normal(cfg.n_voters)
    in_graph_voters = np.argpartition(-score, n_in - 1)[:n_in]
    in_mask = np.zeros(cfg.n_voters, bool)
    in_mask[in_graph_voters] = True

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

    fa, fb = node2voter[g["src"]], node2voter[g["dst"]]
    m = (fa >= 0) & (fb >= 0)
    fa, fb = fa[m], fb[m]
    o = np.argsort(fb, kind="stable")
    fol = fa[o].astype(np.int32)
    ptr = np.concatenate([[0], np.cumsum(np.bincount(fb, minlength=cfg.n_voters))])
    nfol = np.diff(ptr)

    infl = np.flatnonzero(nfol >= cfg.influencer_min_followers)
    if len(infl) < 50:
        infl = np.argsort(-nfol)[:max(200, int(0.002 * cfg.n_voters))]
    infl_cum = np.cumsum(np.sqrt(nfol[infl].astype(float)))

    vcomm = np.where(voter_node >= 0, g["comm"][np.maximum(voter_node, 0)], -1)
    comm_voters, comm_cum = [], []
    for k in range(g["K"]):
        vs = np.flatnonzero(vcomm == k)
        comm_voters.append(vs)
        comm_cum.append(np.cumsum(act[vs]) if len(vs) else np.zeros(0))
    comm_size = np.array([len(v) for v in comm_voters], float)
    seed_p = np.where(comm_size >= 10, comm_size, 0.0)
    seed_cum = np.cumsum(seed_p)

    return dict(graph_ids=graph_ids, voter_uid=voter_uid, voter_node=voter_node, act=act, fol=fol, ptr=ptr,
                nfol=nfol, infl=infl, infl_cum=infl_cum, comm_voters=comm_voters, comm_cum=comm_cum,
                seed_cum=seed_cum, indeg=indeg)

def build_stories(cfg, rng):
    n = cfg.n_stories
    u = (rng.permutation(n) + rng.random(n)) / n
    u[np.argmin(u)] = 0.0
    u[np.argmax(u)] = 1.0
    sizes = np.exp(np.interp(u, STORY_Q_P, np.log(STORY_Q_V))) * cfg.size_scale * cfg.size_adjust
    sizes = np.maximum(np.round(sizes), 20).astype(np.int64)

    spd = np.array(STORIES_PER_DAY, float)
    cnt = np.floor(spd / spd.sum() * n).astype(int)
    for i in rng.choice(len(spd), n - cnt.sum(), replace=False if n - cnt.sum() <= len(spd) else True):
        cnt[i] += 1
    day = np.repeat(np.arange(len(spd)), cnt)
    clock0 = (T_START % 86400) // 3600
    hrs = np.arange(24)
    abs_hour = (clock0 + hrs) % 24
    hp = np.array(HOUR_SHARE_UTC)[abs_hour]
    hp = hp / hp.sum()
    bins = rng.choice(24, n, p=hp)
    start = T_START + day * 86400 + bins * 3600 + rng.integers(0, 3600, n)
    start = np.maximum(start, T_START)
    start[np.argmin(start)] = T_START
    rng.shuffle(start)

    logs = np.log(sizes.astype(float))
    thr = np.quantile(logs, 1 - cfg.viral_top_fraction)
    viral = logs >= thr
    base = (logs - thr) / cfg.softness
    e_sh = rng.standard_normal(n) * cfg.shared_noise
    a_vol = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_volume)
    a_clu = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_cluster)
    a_tmp = sigmoid(base + e_sh + rng.standard_normal(n) * cfg.noise_temporal)
    return dict(sizes=sizes, start=start.astype(np.int64), viral=viral, a_vol=a_vol, a_clu=a_clu, a_tmp=a_tmp)

def build_votes(cfg, rng, U, S):
    nV = cfg.n_voters
    act, fol, ptr, nfol = U["act"], U["fol"], U["ptr"], U["nfol"]
    cumw = np.cumsum(act)
    used = np.zeros(nV, bool)
    onset, sig = cfg.noise_phase_hours, cfg.signal_hours
    F_sig = float(np.interp(sig, AGE_H, AGE_CDF))
    lag_h = cfg.chain_lag_minutes / 60.0
    n = cfg.n_stories

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

        mA = cfg.early_mean_A * (onset / 2.0) ** 0.75 * (1 + cfg.early_leak * (a_v - 0.5))
        nA = int(nb(rng, mA, cfg.count_dispersion))
        age_A = onset * rng.random(nA) ** (1 / 0.75)
        usr_A = wdraw(rng, cumw, nA)

        width = max(sig - onset, 1e-6)
        mB = cfg.early_mean_B * width / 2.0
        boost = cfg.volume_boost * a_v
        nB = int(nb(rng, mB * (1 + boost), cfg.count_dispersion))
        n_extra = rng.binomial(nB, boost / (1 + boost)) if nB else 0
        age_B = np.concatenate([onset + width * rng.random(nB - n_extra),
                                onset + width * np.sqrt(rng.random(n_extra))])
                                
        p_clu = cfg.cluster_base + (cfg.cluster_strength - cfg.cluster_base) * a_c
        usr_B = wdraw(rng, cumw, nB)
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
                    usr_B[sel] = cv[wdraw(rng, cc, len(sel))]

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
            lags = rng.exponential(lag_h, k) + 1.0 / 3600
            if rng.random() < q_fwd:
                tf = np.minimum(t0 + lags, sig - 1e-4)
            else:
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
        age_E = age_E - age_E.min()
        nE = len(usr_E)
        used[usr_E] = True

        n_late = max(size - nE, 10)
        tau = float(np.exp(cfg.time_scale_sigma * rng.standard_normal()))
        horizon = (T_END - int(S["start"][j])) / 3600.0
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
        ts_l.append(np.minimum(ts, T_END))
        v_l.append(users.astype(np.int32))
        s_l.append(np.full(len(users), j, np.int32))
        story_off[j + 1] = story_off[j] + len(users)

    ts = np.concatenate(ts_l)
    vv = np.concatenate(v_l)
    ss = np.concatenate(s_l)

    cnt = np.bincount(vv, minlength=nV)
    zero = np.flatnonzero(cnt == 0)
    if len(zero):
        sp_ = S["sizes"] / S["sizes"].sum()
        sj = rng.choice(n, len(zero), p=sp_)
        nrows = np.diff(story_off)[sj]
        row = story_off[sj] + (rng.random(len(zero)) * nrows).astype(np.int64)
        ts = np.concatenate([ts, np.minimum(ts[row] + rng.integers(0, 600, len(zero)), T_END)])
        vv = np.concatenate([vv, zero.astype(np.int32)])
        ss = np.concatenate([ss, sj.astype(np.int32)])

    return ts, vv, ss, n_social_total

def build_friend_table(cfg, rng, g, U):
    src, dst, N = g["src"], g["dst"], g["N"]
    E = len(src)
    key = src * N + dst
    rkey = dst * N + src
    so = np.argsort(key)
    skey = key[so]
    pos = np.searchsorted(skey, rkey)
    pos = np.minimum(pos, E - 1)
    has_rev = skey[pos] == rkey
    partner = np.where(has_rev, so[pos], -1)

    date = np.interp(rng.random(E), FRIEND_DATE_P, FRIEND_DATE_V)
    fwd = np.flatnonzero(has_rev & (src < dst))
    gap = np.clip(rng.lognormal(np.log(0.73), 3.5, len(fwd)), 1e-4, 700) * 86400
    date[partner[fwd]] = np.minimum(date[fwd] + gap, FRIEND_DATE_V[-1])
    date = date.astype(np.int64)

    mutual = has_rev.copy()
    extra = (cfg.mutual_flag_share - has_rev.mean()) / max(1 - has_rev.mean(), 1e-9)
    mutual |= (~has_rev) & (rng.random(E) < max(extra, 0))
    uid = U["graph_ids"]
    df = pd.DataFrame({"mutual": mutual.astype(np.int8), "friend_date": date,
                       "user_id": uid[src], "friend_id": uid[dst]})
    if cfg.emulate_artifacts:
        bad = rng.choice(E, int(round(1670 * E / 1731652)), replace=False)
        df.loc[bad, "friend_date"] = 0
        n_loop = max(1, int(round(5 * E / 1731652)))
        ids = rng.choice(uid, n_loop, replace=False)
        loops = pd.DataFrame({"mutual": 0, "friend_date": date[rng.integers(0, E, n_loop)],
                              "user_id": ids, "friend_id": ids})
        df = pd.concat([df, loops], ignore_index=True)
    if cfg.friends_sort == "date":
        df = df.sort_values("friend_date", kind="stable")
    elif cfg.friends_sort == "user":
        df = df.sort_values(["user_id", "friend_id"], kind="stable")
    else:
        df = df.sample(frac=1.0, random_state=cfg.seed)
    return df, has_rev.mean(), mutual.mean()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with parameter overrides")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    ap.add_argument("--out", help="output directory (default ./data/generated data)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--print-config", action="store_true")
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
    if args.scale is not None:
        ov["scale"] = args.scale
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

    if cfg.scale != 1.0:
        s = cfg.scale
        cfg.n_stories = max(int(cfg.n_stories * s ** 0.5), 50)
        cfg.size_scale = s ** 0.5
        cfg.n_voters = int(cfg.n_voters * s)
        cfg.n_graph_users = int(cfg.n_graph_users * s)
        cfg.n_edges = int(cfg.n_edges * s)
        cfg.id_max = int(cfg.id_max * s)
        cfg.n_communities = max(int(cfg.n_communities * s ** 0.5), 20)
        cfg.force_top_hubs = max(int(cfg.force_top_hubs * s), 50)
        cfg.influencer_min_followers = max(int(cfg.influencer_min_followers * s ** 0.5), 3)

    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    meta = os.path.join(cfg.out_dir, "_synthetic_meta")
    os.makedirs(meta, exist_ok=True)

    log("building friend graph ...")
    g = build_graph(cfg, rng)
    log("building users ...")
    U = build_users(cfg, rng, g)
    log("building stories ...")
    S = build_stories(cfg, rng)
    log("building votes ...")
    ts, vv, ss, n_social = build_votes(cfg, rng, U, S)

    n = cfg.n_stories
    if cfg.story_id_order == "time":
        sid = np.empty(n, np.int64)
        sid[np.argsort(S["start"], kind="stable")] = np.arange(1, n + 1)
    else:
        sid = rng.permutation(n) + 1
    votes = pd.DataFrame({"vote_date": ts, "voter_id": U["voter_uid"][vv], "story_id": sid[ss]})
    if cfg.emulate_artifacts:
        k = int(round(7298 * len(votes) / 3010897))
        rows = votes.iloc[rng.choice(len(votes), k, replace=False)].copy()
        rows["vote_date"] = np.minimum(rows["vote_date"] + np.maximum(1, rng.lognormal(6, 2, k)).astype(np.int64), T_END)
        votes = pd.concat([votes, rows, votes.iloc[:1]], ignore_index=True)
    votes = votes.sort_values("vote_date", kind="stable")

    log("building friends table ...")
    fr, recip, mut = build_friend_table(cfg, rng, g, U)

    log("writing CSVs ...")
    votes.to_csv(os.path.join(cfg.out_dir, cfg.votes_filename), header=False, index=False, lineterminator="\n")
    fr.to_csv(os.path.join(cfg.out_dir, cfg.friends_filename), header=False, index=False, lineterminator="\n")

    final = np.bincount(ss, minlength=n)
    lab = final >= np.quantile(final, 1 - cfg.viral_top_fraction)
    pd.DataFrame({"story_id": sid, "final_votes": final, "viral_label": lab.astype(int),
                  "planned_viral": S["viral"].astype(int), "a_volume": S["a_vol"].round(4),
                  "a_cluster": S["a_clu"].round(4), "a_temporal": S["a_tmp"].round(4),
                  "first_vote_ts": S["start"]}).sort_values("story_id").to_csv(
        os.path.join(meta, "story_labels.csv"), index=False)
    json.dump({"synthetic": True, "generator": "dig_generator.py", "params": asdict(cfg)},
              open(os.path.join(meta, "manifest.json"), "w"), indent=2)

if __name__ == "__main__":
    main()