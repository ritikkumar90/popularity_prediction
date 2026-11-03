import pandas as pd
from src.config import cfg

# Mocking config and classes for test isolation
cfg.datasets["test_mock"] = cfg.datasets["digg"].copy()
cfg.datasets["test_mock"].edge_time_available = True
cfg.dataset["name"] = "test_mock"

from src.data.splitter import TimeSplitter

def test_no_temporal_leakage():
    # 6 nodes: a, b, c, d, e, f
    # Edge e -> f is created AFTER t_obs
    
    splitter = TimeSplitter()
    
    # Mock data directly into the instance
    splitter.edges = pd.DataFrame([
        {"source": "a", "target": "b", "created_at": 100},
        {"source": "c", "target": "d", "created_at": 200},
        {"source": "e", "target": "f", "created_at": 5000}  # CREATED LATER
    ]).sort_values("created_at")
    
    splitter.events = pd.DataFrame([
        {"post_id": "p1", "user": "a", "time": 150},
        {"post_id": "p1", "user": "b", "time": 160}
    ]).sort_values("time")
    
    splitter.splits_dir = Path("/tmp")
    
    # Fake meta where t_obs is 1000
    meta = pd.DataFrame([{"post_id": "p1", "t0": 150, "t_obs": 1000, "t_end": 80000, "label": 2}])
    meta.to_parquet("/tmp/split_meta.parquet", index=False)
    
    # Fetch snapshot
    valid_edges, valid_events, _ = splitter.get_snapshot("p1")
    
    # Assertions
    assert len(valid_edges) == 2, "Should only retrieve 2 edges"
    assert "f" not in valid_edges["target"].values, "Edge e->f MUST be ignored because 5000 > t_obs (1000)"
    print("Test passed: Future edges are strictly purged!")

if __name__ == "__main__":
    test_no_temporal_leakage()