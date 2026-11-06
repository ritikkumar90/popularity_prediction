"""Build the canonical tables (events, posts, edges) for one dataset.

Python:
    from src.data.main_loader import prepare_data
    prepare_data("flickr", raw_dir="data/raw/flickr", subset_mode="post_fraction", subset_value=0.05)

Command line:
    python -m src.data.main_loader --dataset flickr --raw-dir data/raw/flickr \
        --subset-mode post_fraction --subset-value 0.05

Which dataset runs is decided only by the `dataset` argument. The adapters themselves are
listed in ADAPTER_REGISTRY (src/data/adapters/__init__.py). Output goes to
<base_path>/data/processed/<dataset>/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path

from src.data.adapters import ADAPTER_REGISTRY

log = logging.getLogger(__name__)


def prepare_data(
    dataset: str = "flickr",
    base_path: str | Path = ".",
    raw_dir: str | Path | None = None,
    **adapter_kwargs,
):
    """Run one adapter and write its canonical tables.

    dataset          name registered in ADAPTER_REGISTRY (e.g. "flickr")
    base_path        project root; output goes to <base_path>/data/processed/<dataset>
    raw_dir          folder with the raw files (default: <base_path>/data/raw/<dataset>)
    adapter_kwargs   passed straight to the adapter, e.g. subset_mode, subset_value, seed,
                     author_rule, restrict_edges (see the adapter's docstring)
    """
    if dataset not in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter for dataset '{dataset}' is not registered. "
                         f"Available: {sorted(ADAPTER_REGISTRY)}")

    base_path = Path(base_path)
    raw_dir = Path(raw_dir) if raw_dir is not None else base_path / "data" / "raw" / dataset
    adapter = ADAPTER_REGISTRY[dataset](base_path=base_path, raw_dir=raw_dir, **adapter_kwargs)

    log.info("Processing dataset: %s (raw_dir=%s) concurrently", adapter.dataset_name, raw_dir)

    # Two independent branches of work
    def process_events_and_posts():
        events = adapter.clean_events(adapter.build_events())
        posts = adapter.clean_posts(adapter.build_posts(events))
        return adapter.align_events_and_posts(events, posts)

    def process_edges():
        return adapter.clean_edges(adapter.build_edges())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_ep = executor.submit(process_events_and_posts)
        future_edges = executor.submit(process_edges)

        events, posts = future_ep.result()
        edges = future_edges.result()

    # Final synchronized validation and save
    adapter.check_time_ranges(events, posts)
    adapter.check_time_alignment(events, edges)
    adapter.validate(events, posts, edges)
    adapter.save(events, posts, edges)

    log.info(
        "Saved canonical tables to %s (events=%d, posts=%d, edges=%d)",
        adapter.processed_dir, len(events), len(posts), len(edges),
    )
    return adapter.processed_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical tables for a dataset.")
    p.add_argument("--dataset", default="flickr", choices=sorted(ADAPTER_REGISTRY))
    p.add_argument("--base-path", default=".")
    p.add_argument("--raw-dir", default=None, help="default: <base-path>/data/raw/<dataset>")
    p.add_argument("--subset-mode", default=None,
                   choices=["full", "post_fraction", "top_posts", "first_rows"])
    p.add_argument("--subset-value", type=float, default=None,
                   help="fraction (post_fraction) or count (top_posts, first_rows)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--author-rule", default=None, choices=["first_event", "owner_file"])
    p.add_argument("--no-restrict-edges", action="store_true",
                   help="keep the full graph even when a subset is used")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    a = _parse_args()

    kwargs = {}
    if a.subset_mode is not None:
        kwargs["subset_mode"] = a.subset_mode
    if a.subset_value is not None:
        # counts must be integers (top_posts, first_rows); fractions stay floats
        kwargs["subset_value"] = a.subset_value if a.subset_mode == "post_fraction" else int(a.subset_value)
    if a.seed is not None:
        kwargs["seed"] = a.seed
    if a.author_rule is not None:
        kwargs["author_rule"] = a.author_rule
    if a.no_restrict_edges:
        kwargs["restrict_edges"] = False

    prepare_data(a.dataset, base_path=a.base_path, raw_dir=a.raw_dir, **kwargs)