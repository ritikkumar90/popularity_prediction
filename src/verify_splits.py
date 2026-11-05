import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SplitVerifier")

def verify_dataset(dataset_name="flickr"):
    # Dynamically resolve the project root from inside the src/ folder
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "data" / "splits" / dataset_name
    
    meta_path = splits_dir / "split_meta.parquet"
    labels_path = splits_dir / "labels.parquet"

    if not meta_path.exists() or not labels_path.exists():
        log.error(f"Missing split files in {splits_dir}")
        return

    log.info(f"Loading splits from {splits_dir}...")
    meta = pd.read_parquet(meta_path)
    labels = pd.read_parquet(labels_path)

    # 1. Label Isolation Check
    assert "label" not in meta.columns, "CRITICAL: 'label' column leaked into metadata!"
    assert "growth" not in meta.columns, "CRITICAL: 'growth' column leaked into metadata!"
    assert len(meta) == len(labels), "Row counts between metadata and labels do not match."
    assert set(meta["post_id"]) == set(labels["post_id"]), "Post IDs do not match between meta and labels."
    log.info("✓ Labels are perfectly isolated from metadata.")

    # 2. Extract splits
    train = meta[meta["split"] == "train"]
    val = meta[meta["split"] == "val"]
    test = meta[meta["split"] == "test"]

    log.info(f"Counts - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 3. Disjoint Sets Check (No Post in Multiple Splits)
    assert set(train["post_id"]).isdisjoint(set(val["post_id"])), "CRITICAL: Train and Val share posts!"
    assert set(train["post_id"]).isdisjoint(set(test["post_id"])), "CRITICAL: Train and Test share posts!"
    assert set(val["post_id"]).isdisjoint(set(test["post_id"])), "CRITICAL: Val and Test share posts!"
    log.info("✓ Splits are mutually exclusive (no shared posts).")

    # 4. Chronological Order Check
    if not train.empty and not val.empty:
        assert train["t0"].max() <= val["t0"].min(), "CRITICAL: Train set t0 overlaps with Val set t0!"
    if not val.empty and not test.empty:
        assert val["t0"].max() <= test["t0"].min(), "CRITICAL: Val set t0 overlaps with Test set t0!"
    log.info("✓ Splits respect strict chronological 't0' ordering.")

    # 5. Future Leakage Check (t_end vs t0)
    if not train.empty and not val.empty:
        assert train["t_end"].max() < val["t0"].min(), "CRITICAL: Train prediction window leaks into Val!"
    if not val.empty and not test.empty:
        assert val["t_end"].max() < test["t0"].min(), "CRITICAL: Val prediction window leaks into Test!"
    log.info("✓ No future data leakage (t_end strictly precedes the next split).")

    log.info("ALL CHECKS PASSED SUCCESSFULLY. The dataset is safe for modeling.")

if __name__ == "__main__":
    verify_dataset("flickr")