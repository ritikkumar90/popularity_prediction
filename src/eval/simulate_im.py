import logging
import pandas as pd

from src.config import load_config
from src.diffusion.simulator import DiffusionRunner

log = logging.getLogger(__name__)

def run_im_simulation():
    cfg = load_config()
    base_dir = cfg.run.base_path
    results_dir = base_dir / cfg.run.output_dir / cfg.dataset_name / "results"
    
    log.info("Starting IM simulations using DiffusionRunner...")
    
    # Initialize the runner which loads fitted parameters and test subgraphs[cite: 3, 4]
    runner = DiffusionRunner(cfg)
    
    # run_all() evaluates Greedy and CELF across k=1,2,3 for all test posts
    sim_results_df = runner.run_all()
    
    # Map the output columns from spread_comparison.csv to the exact template format
    im_df = pd.DataFrame({
        "post_id": sim_results_df["post_id"],
        "IM (k=1) seed by greedy": sim_results_df["greedy_k1"],
        "IM (k=1) seed by celf": sim_results_df["celf_k1"],
        "IM avg(k=1, 2, 3) seed by greedy": sim_results_df["greedy_avg"],
        "IM avg(k=1, 2, 3) seed by celf": sim_results_df["celf_avg"]
    })
    
    out_path = results_dir / "im_predictions.csv"
    im_df.to_csv(out_path, index=False)
    log.info(f"Saved formatted IM simulations to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_im_simulation()