import heapq
from typing import Callable

class SeedSelectors:
    @staticmethod
    def greedy(
        candidates: list[int], 
        k_max: int, 
        spread_fn: Callable[[list[int]], float], 
        base_spread: float
    ) -> tuple[list[int], dict[int, float]]:
        """
        Standard Greedy Influence Maximization.
        Evaluates all remaining candidates at every step k = 1..k_max.
        """
        selected: list[int] = []
        spread_by_k: dict[int, float] = {}
        remaining = set(candidates)
        current_spread = base_spread

        for k in range(1, k_max + 1):
            if not remaining:
                spread_by_k[k] = current_spread
                continue

            best_node = None
            best_spread = current_spread

            for u in remaining:
                s = spread_fn(selected + [u])
                if s > best_spread or best_node is None:
                    best_spread = s
                    best_node = u

            selected.append(best_node)
            remaining.remove(best_node)
            current_spread = best_spread
            spread_by_k[k] = current_spread

        return selected, spread_by_k

    @staticmethod
    def celf(
        candidates: list[int], 
        k_max: int, 
        spread_fn: Callable[[list[int]], float], 
        base_spread: float
    ) -> tuple[list[int], dict[int, float]]:
        """
        CELF (Cost-Effective Lazy Forward) Influence Maximization.
        Exploits submodularity to skip redundant candidate evaluations.
        """
        selected: list[int] = []
        spread_by_k: dict[int, float] = {}

        if not candidates:
            for k in range(1, k_max + 1):
                spread_by_k[k] = base_spread
            return selected, spread_by_k

        # Pass 1: evaluate marginal gain of each individual candidate
        pq = []
        for u in candidates:
            s = spread_fn([u])
            mg = s - base_spread
            # Max-heap using negative marginal gain: (-mg, node, last_updated_step)
            heapq.heappush(pq, (-mg, u, 0))

        current_spread = base_spread

        # Pass 2..k_max: lazy forward updates
        for k in range(1, k_max + 1):
            while pq:
                neg_mg, u, step = heapq.heappop(pq)
                if step == k - 1:
                    # Top node is already updated against current seed set
                    selected.append(u)
                    current_spread += (-neg_mg)
                    spread_by_k[k] = current_spread
                    break
                else:
                    # Re-evaluate against current seed set and push back
                    new_spread = spread_fn(selected + [u])
                    new_mg = new_spread - current_spread
                    heapq.heappush(pq, (-new_mg, u, k - 1))
            else:
                spread_by_k[k] = current_spread

        return selected, spread_by_k