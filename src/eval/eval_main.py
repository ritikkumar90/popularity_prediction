import logging
from src.eval.predict_ml import run_ml_inference
from src.eval.simulate_im import run_im_simulation
from src.eval.build_report import generate_report

log = logging.getLogger(__name__)

def main():
    log.info("=== Phase 1: Machine Learning Inference ===")
    run_ml_inference()
    
    log.info("=== Phase 2: Influence Maximization Simulation ===")
    run_im_simulation()
    
    log.info("=== Phase 3: Building Evaluation Matrix ===")
    generate_report()
    
    log.info("=== Evaluation Pipeline Complete ===")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()