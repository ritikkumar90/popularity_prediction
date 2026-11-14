"""Seed selection algorithms for Influence Maximization.

Provides the Greedy and CELF (Cost-Effective Lazy Forward) algorithms.
These functions fulfill the contract expected by variants.py:
  - objective.value(seeds) -> float
  - objective.values(base, cands) -> np.ndarray (batch evaluation)
  - Returns an object with .seeds, .values, and .n_evals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SelectionResult:
    seeds: list[int]
    values: list[float]
    n_evals: int


def greedy(objective: Any, candidates: list[int], k: int) -> SelectionResult:
    """Standard Greedy algorithm for Influence Maximization.
    
    Evaluates every remaining candidate in each iteration in parallel 
    and picks the one that provides the maximum expected spread.
    """
    cands = list(candidates)
    seeds: list[int] = []
    values: list[float] = []

    for _ in range(k):
        if not cands:
            break
        
        # Vectorized evaluation of all remaining candidates
        spreads = objective.values(seeds, cands)
        
        best_idx = int(np.argmax(spreads))
        best_cand = cands[best_idx]
        best_spread = float(spreads[best_idx])
        
        seeds.append(best_cand)
        values.append(best_spread)
        cands.pop(best_idx)

    return SelectionResult(seeds, values, objective.n_evals)


def celf(objective: Any, candidates: list[int], k: int) -> SelectionResult:
    """CELF (Cost-Effective Lazy Forward) algorithm.
    
    Exploits submodularity to drastically reduce the number of spread evaluations 
    needed compared to the standard Greedy approach.
    """
    cands = list(candidates)
    seeds: list[int] = []
    values: list[float] = []

    # 1. Initial evaluation of all candidates (vectorized)
    base_spread = float(objective.value([]))
    initial_spreads = objective.values([], cands)
    
    # Queue stores [marginal_gain, candidate]
    queue = [[float(s) - base_spread, c] for c, s in zip(cands, initial_spreads)]
    queue.sort(key=lambda x: x[0], reverse=True)

    current_spread = base_spread

    # 2. Select k seeds using lazy evaluations
    for _ in range(k):
        if not queue:
            break

        while True:
            candidate = queue[0][1]

            # Re-evaluate the marginal gain of the top candidate
            new_spread = float(objective.value(seeds + [candidate]))
            new_mg = new_spread - current_spread

            # If it's still better than or equal to the next candidate's upper bound, 
            # we don't need to re-evaluate the rest. Pick it.
            if len(queue) == 1 or new_mg >= queue[1][0]:
                seeds.append(candidate)
                values.append(new_spread)
                current_spread = new_spread
                queue.pop(0)  # Remove the chosen candidate
                break
            else:
                # It's no longer the best. Re-insert with its updated gain and re-sort.
                queue[0][0] = new_mg
                queue.sort(key=lambda x: x[0], reverse=True)

    return SelectionResult(seeds, values, objective.n_evals)