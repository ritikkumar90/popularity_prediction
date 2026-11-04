import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import load_config
from src.ml.dataset import PostGraphDataset
from src.ml.train import find_latest_hash_dir

# Assuming your simulator and selectors expose these or similar interfaces:
# from src.im.selectors import GreedySelector, CELFSelector
# from src.diffusion.simulator import ICSimulator

log = logging.getLogger(__name__)

def run_im_simulation():
    cfg = load_config()
    base_dir = cfg.run.base_path
    samples_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "samples"
    results_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = find_latest_hash_dir(samples_dir)
    test_dataset = PostGraphDataset(cfg, split="test", config_hash=config_hash)
    
    post_ids = test_dataset.index["post_id"].tolist()
    
    # Initialize output storage
    im_results = {
        "post_id": post_ids,
        "IM (k=1) seed by greedy": [],
        "IM (k=1) seed by celf": [],
        "IM avg(k=1, 2, 3) seed by greedy": [],
        "IM avg(k=1, 2, 3) seed by celf": []
    }
    
    log.info(f"Starting IM simulations for {len(post_ids)} test graphs...")
    
    for idx, post_id in enumerate(post_ids):
        # Extract the specific subgraph for this post
        data = test_dataset[idx]
        
        # NOTE: Replace the block below with your actual module calls
        # -----------------------------------------------------------
        # greedy_seeds = GreedySelector(data, k=3).select()
        # celf_seeds = CELFSelector(data, k=3).select()
        # 
        # sim = ICSimulator(data, n_runs=cfg.simulate.n_runs)
        # 
        # greedy_k1 = sim.evaluate(greedy_seeds[:1])
        # greedy_k3 = np.mean([sim.evaluate(greedy_seeds[:k]) for k in [1, 2, 3]])
        # 
        # celf_k1 = sim.evaluate(celf_seeds[:1])
        # celf_k3 = np.mean([sim.evaluate(celf_seeds[:k]) for k in [1, 2, 3]])
        # -----------------------------------------------------------
        
        # Placeholder mock execution (Remove this when wiring your real simulator)
        greedy_k1, celf_k1 = 0.0, 0.0
        greedy_k3, celf_k3 = 0.0, 0.0
        
        im_results["IM (k=1) seed by greedy"].append(greedy_k1)
        im_results["IM (k=1) seed by celf"].append(celf_k1)
        im_results["IM avg(k=1, 2, 3) seed by greedy"].append(greedy_k3)
        im_results["IM avg(k=1, 2, 3) seed by celf"].append(celf_k3)
        
        if (idx + 1) % 10 == 0:
            log.info(f"Simulated {idx + 1}/{len(post_ids)} graphs...")

    im_df = pd.DataFrame(im_results)
    out_path = results_dir / "im_predictions.csv"
    im_df.to_csv(out_path, index=False)
    log.info(f"Saved IM simulations to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_im_simulation()