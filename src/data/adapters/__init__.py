"""Dataset adapters.

Each adapter maps one raw dataset onto the canonical tables (events, posts, edges).
Adapters are built directly from plain arguments (no config files), e.g.

    from src.data.adapters import FlickrAdapter
    FlickrAdapter(base_path=".", raw_dir="data/raw/flickr",
                  subset_mode="post_fraction", subset_value=0.05).process_and_save()
"""
from src.data.adapters.base import BaseAdapter
from src.data.adapters.flickr import FlickrAdapter

# To add a dataset: write an adapter (subclass of BaseAdapter), import it here and
# register it. DiggAdapter is left out until it is refactored to the same
# constructor style as FlickrAdapter (it still expects the old Config object).
ADAPTER_REGISTRY = {
    "flickr": FlickrAdapter,
}

__all__ = ["BaseAdapter", "FlickrAdapter", "ADAPTER_REGISTRY"]