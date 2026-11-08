"""Variants for the diffusion engine (simulator.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.diffusion.selectors import SeedSelectors

EVAL_OFFSET = 1_000_000   # random streams for final numbers

@dataclass(frozen=True)
class Opts:
    k_list: tuple = (1, 2, 3)
    selection_runs: int = 30     
    eval_runs: int = 200         
    candidate_top_n: int = 50    
    im_algos: tuple = ("greedy", "celf")  # Configurable IM algos

VARIANTS: dict[str, Callable] = {}

def variant(name: str):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco

class _Counted:
    """Counts how many spreads a selector asks for."""
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, seeds):
        self.calls += 1
        return self.fn(seeds)

def _cached_search_fn(post, o: Opts):
    """Spread with `selection_runs`, cached by seed set."""
    cache: dict = {}

    def fn(seeds):
        key = tuple(sorted(seeds))
        if key not in cache:
            cache[key] = post.spread(seeds=list(key), runs=o.selection_runs, offset=0)
        return cache[key]

    return fn

@variant("baseline")
def baseline(post, o: Opts) -> dict:
    """All users active at t_obs stay active, nothing added."""
    return {"baseline_pred": post.spread(seeds=(), runs=o.eval_runs, offset=EVAL_OFFSET)}

@variant("im_seeds")
def im_seeds(post, o: Opts) -> dict:
    """Baseline + k extra seeds chosen by dynamic influence maximization algorithms."""
    cands = post.candidates(o.candidate_top_n)
    kmax = max(o.k_list)
    search = _cached_search_fn(post, o)
    base_search = search([])
    base_eval = post.spread(seeds=(), runs=o.eval_runs, offset=EVAL_OFFSET)

    out: dict = {"n_candidates": len(cands)}
    
    # Dynamically build algorithms to run based on Opts
    algos_to_run = []
    if "greedy" in o.im_algos:
        algos_to_run.append(("greedy", SeedSelectors.greedy))
    if "celf" in o.im_algos:
        algos_to_run.append(("celf", SeedSelectors.celf))
        
    for name, select in algos_to_run:
        counted = _Counted(search)
        chosen, _ = select(cands, kmax, counted, base_search)
        out[f"{name}_calls"] = counted.calls
        for k in o.k_list:
            s = post.spread(seeds=chosen[:k], runs=o.eval_runs, offset=EVAL_OFFSET)
            out[f"{name}_k{k}"] = s
            out[f"{name}_k{k}_gain"] = s - base_eval
        out[f"{name}_seeds"] = "|".join(map(str, post.raw_ids(chosen[:kmax])))
    return out