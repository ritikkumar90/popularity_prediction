import os
import sys
import torch
import pandas as pd
import numpy as np

# Add local directories to path so we can import the models
sys.path.append('/content/drive/MyDrive/Colab Notebooks/PopularityPrediction')
from src.hgt_model import HGTPopularityPredictor
from diffusion_sim.ic_model import build_social_graph, independent_cascade

def run_comparison(data_dir, weights_dir, num_test=5, mc_runs=20, ic_prob=1.5):
    print("1. Re-deriving ID Mappings to link Flickr IDs to PyG nodes...")
    growth_path = os.path.join(data_dir, 'raw_mini', 'flickr-growth.txt.gz')
    df_growth = pd.read_csv(growth_path, sep=r'\t| ', engine='python', header=None, usecols=[0, 1])
    train_fav_path = os.path.join(data_dir, 'splits', 'train_favorites.csv')
    df_fav_train = pd.read_csv(train_fav_path)
    photo_path = os.path.join(data_dir, 'raw_mini', 'flickr-all-photos.txt.gz')
    df_photos = pd.read_csv(photo_path, sep=r'\t| ', engine='python', header=None, usecols=[0, 1])
    
    unique_photos = set(df_fav_train['photo_id']).union(set(df_photos[0]))
    photo_mapping = {raw_id: i for i, raw_id in enumerate(unique_photos)}

    print("2. Building Social Graph for Independent Cascade...")
    ic_graph = build_social_graph(data_dir)

    print("3. Loading Pre-Trained HGT Model...")
    processed_path = os.path.join(data_dir, 'processed', 'train_heterodata.pt')
    data = torch.load(processed_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = HGTPopularityPredictor(
        hidden_channels=64, out_channels=1, num_heads=4, num_layers=2,
        node_types=data.node_types, metadata=data.metadata()
    ).to(device)
    
    weights_path = os.path.join(weights_dir, 'hgt_best_model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("4. Selecting Test Photos...")
    fav_all_path = os.path.join(data_dir, 'raw_mini', 'flickr-all-photo-favorite-markings.txt.gz')
    df_fav_all = pd.read_csv(fav_all_path, sep=r'\t| ', engine='python', header=None)
    df_fav_all.columns = ['user_id', 'photo_id', 'timestamp']
    
    # Pick photos with at least 10 likes for a meaningful cascade
    photo_counts = df_fav_all['photo_id'].value_counts()
    valid_photos = photo_counts[photo_counts > 10].index.tolist()
    valid_photos = [p for p in valid_photos if p in photo_mapping][:num_test]

    print(f"5. Running Joint Evaluations...\n")
    results = []
    
    # Perform a single fast forward-pass on the mini-dataset for the HGT
    with torch.no_grad():
        hgt_preds = model(data.x_dict, data.edge_index_dict)
    
    for pid in valid_photos:
        # A. HGT Prediction extraction
        pyg_idx = photo_mapping[pid]
        hgt_pred_val = hgt_preds[pyg_idx].item()
        
        # B. Get Seeds & Ground Truth
        photo_data = df_fav_all[df_fav_all['photo_id'] == pid].sort_values('timestamp')
        ground_truth = len(photo_data)
        seeds = photo_data['user_id'].head(3).tolist()
        
        # C. IC Simulation Prediction
        simulated_spreads = []
        for _ in range(mc_runs):
            sim_pop, _ = independent_cascade(ic_graph, seeds, prob=ic_prob)
            simulated_spreads.append(sim_pop)
        ic_pred_val = np.mean(simulated_spreads)
        
        results.append({
            'Photo_ID': pid,
            'Ground_Truth': ground_truth,
            'HGT_Prediction': round(hgt_pred_val, 1),
            'IC_Prediction': round(ic_pred_val, 1)
        })

    print("🏆 --- FINAL MODEL COMPARISON --- 🏆")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    WEIGHTS_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/models/saved_weights'
    
    # Passed ic_prob=1.5 to match your successful degree-normalized run
    run_comparison(DATA_DIR, WEIGHTS_DIR, num_test=5, mc_runs=20, ic_prob=0.014)
