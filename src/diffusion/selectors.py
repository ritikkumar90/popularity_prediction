"""Seed selection algorithms for Influence Maximization.

Provides the Greedy and CELF (Cost-Effective Lazy Forward) algorithms to find a set of seeds 
that maximizes the spread of information in a network.
"""
from __future__ import annotations

from typing import Callable


class SeedSelectors:
    """Methods to select k seeds from a list of candidates."""

    @staticmethod
    def greedy(
        candidates: list[int],
        k: int,
        spread_fn: Callable[[list[int]], float],
        base_spread: float
    ) -> tuple[list[int], list[float]]:
        """Standard Greedy algorithm for Influence Maximization.
        
        Evaluates every candidate in each iteration and picks the one that provides 
        the maximum marginal gain.
        """
        seeds: list[int] = []
        gains: list[float] = []
        current_spread = base_spread
        pool = set(candidates)

        for _ in range(k):
            if not pool:
                break
            
            best_candidate = None
            best_spread = -1.0
            
            for c in pool:
                s = spread_fn(seeds + [c])
                if s > best_spread:
                    best_spread = s
                    best_candidate = c
                    
            if best_candidate is not None:
                seeds.append(best_candidate)
                gains.append(best_spread - current_spread)
                current_spread = best_spread
                pool.remove(best_candidate)

        return seeds, gains

    @staticmethod
    def celf(
        candidates: list[int],
        k: int,
        spread_fn: Callable[[list[int]], float],
        base_spread: float
    ) -> tuple[list[int], list[float]]:
        """CELF (Cost-Effective Lazy Forward) algorithm.
        
        Exploits submodularity to drastically reduce the number of spread evaluations 
        needed compared to the standard Greedy approach.
        """
        seeds: list[int] = []
        gains: list[float] = []
        current_spread = base_spread

        # 1. Initial evaluation of all candidates
        queue = []
        for c in candidates:
            s = spread_fn([c])
            mg = s - base_spread
            queue.append((mg, c))

        # Sort descending by marginal gain
        queue.sort(key=lambda x: x[0], reverse=True)

        # 2. Select k seeds using lazy evaluations
        for _ in range(k):
            if not queue:
                break

            while True:
                prev_mg, c = queue.pop(0)

                # Re-evaluate the marginal gain of the top candidate
                s = spread_fn(seeds + [c])
                new_mg = s - current_spread

                # If it's still better than or equal to the next candidate's upper bound, 
                # we don't need to re-evaluate the rest. Pick it.
                if not queue or new_mg >= queue[0][0]:
                    seeds.append(c)
                    gains.append(new_mg)
                    current_spread = s
                    break
                else:
                    # It's no longer the best. Re-insert with its updated gain and re-sort.
                    queue.append((new_mg, c))
                    queue.sort(key=lambda x: x[0], reverse=True)

        return seeds, gains