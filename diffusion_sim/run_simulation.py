
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('/content/drive/MyDrive/Colab Notebooks/PopularityPrediction')
from diffusion_sim.ic_model import build_social_graph, independent_cascade

def generate_expected_spread(data_dir, output_dir, num_test_photos, num_simulations):
    print("1. Building Frozen Social Graph...")
    graph = build_social_graph(data_dir)
    
    print("2. Loading favorites data...")
    fav_path = os.path.join(data_dir, 'raw', 'flickr-all-photo-favorite-markings.txt.gz')
    df_fav = pd.read_csv(fav_path, sep=r'\t| ', engine='python', header=None, nrows=100000)
    df_fav.columns = ['user_id', 'photo_id', 'timestamp']
    
    photo_counts = df_fav['photo_id'].value_counts()
    valid_photos = photo_counts[photo_counts > 10].index.tolist()[:num_test_photos]
    
    print(f"3. Running Monte Carlo Simulations ({num_simulations} runs per photo)...")
    results = []
    
    for i, pid in enumerate(valid_photos):
        photo_data = df_fav[df_fav['photo_id'] == pid].sort_values('timestamp')
        ground_truth = len(photo_data)
        seeds = photo_data['user_id'].head(3).tolist()
        
        # --- MONTE CARLO SIMULATION ---
        # Run the simulation multiple times to find the "Expected" spread
        simulated_spreads = []
        for _ in range(num_simulations):
            sim_popularity, _ = independent_cascade(graph, seeds, prob=0.015)
            simulated_spreads.append(sim_popularity)
            
        # The expected spread is the average of all the simulation runs
        expected_spread = np.mean(simulated_spreads)
        
        results.append({
            'photo_id': pid,
            'ic_expected_popularity': round(expected_spread, 2),
            'ground_truth': ground_truth,
            'seed_count': len(seeds)
        })
        
        if (i + 1) % 10 == 0:
            print(f"   -> Processed {i + 1}/{len(valid_photos)} photos")
            
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ic_simulation_results.csv')
    results_df.to_csv(save_path, index=False)
    
    print(f"\n✅ Simulation Complete! Results saved to {save_path}")
    print(results_df.head())

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    OUTPUT_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/diffusion_sim'
    
    generate_expected_spread(DATA_DIR, OUTPUT_DIR, num_test_photos=10, num_simulations=100)