import logging
from src.config import load_config

# If you have an __init__.py in adapters exporting ADAPTER_REGISTRY, you can import it:
# from src.data.adapters import ADAPTER_REGISTRY
# Otherwise, define it locally here:
from src.data.adapters.flickr import FlickrAdapter

ADAPTER_REGISTRY = {
    "flickr": FlickrAdapter,
    # "digg": DiggAdapter,
    # "higgs": HiggsAdapter
}

log = logging.getLogger(__name__)

def prepare_data():
    """Reads the active dataset from config, runs the adapter, and builds canonical tables."""
    cfg = load_config()
    dataset_name = cfg.dataset_name
    
    if dataset_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter for dataset '{dataset_name}' is not registered.")
        
    adapter_class = ADAPTER_REGISTRY[dataset_name]
    
    # Instantiate with the new config object to match BaseAdapter's signature
    adapter = adapter_class(cfg)
    
    adapter.process_and_save()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()