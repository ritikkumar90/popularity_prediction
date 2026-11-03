from src.data.adapters.digg import DiggAdapter
from src.data.adapters.flickr import FlickrAdapter

ADAPTER_REGISTRY = {
    "digg": DiggAdapter,
    "flickr": FlickrAdapter,
}