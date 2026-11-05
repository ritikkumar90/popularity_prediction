import logging
import concurrent.futures
from src.config import load_config
from src.data.adapters.flickr import FlickrAdapter

ADAPTER_REGISTRY = {
    "flickr": FlickrAdapter,
}

log = logging.getLogger(__name__)

def prepare_data():
    """Reads the active dataset from config, runs the adapter concurrently, and builds canonical tables."""
    cfg = load_config()
    dataset_name = cfg.dataset_name
    
    if dataset_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter for dataset '{dataset_name}' is not registered.")
        
    adapter_class = ADAPTER_REGISTRY[dataset_name]
    adapter = adapter_class(cfg)
    
    log.info(f"Processing dataset: {adapter.dataset_name} (Profile: {adapter.cfg.run.profile}) concurrently")

    # Define the two independent branches of work
    def process_events_and_posts():
        events = adapter.clean_events(adapter.build_events())
        posts = adapter.clean_posts(adapter.build_posts(events))
        return adapter.align_events_and_posts(events, posts)

    def process_edges():
        return adapter.clean_edges(adapter.build_edges())

    # Execute both branches simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_ep = executor.submit(process_events_and_posts)
        future_edges = executor.submit(process_edges)
        
        events, posts = future_ep.result()
        edges = future_edges.result()

    # Run the final synchronized validation and save steps
    adapter.check_time_ranges(events, posts)
    adapter.check_time_alignment(events, edges)
    adapter.validate(events, posts, edges)
    adapter.save(events, posts, edges)
    
    log.info(
        "Saved canonical tables to %s (events=%d, posts=%d, edges=%d)",
        adapter.processed_dir, len(events), len(posts), len(edges),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()