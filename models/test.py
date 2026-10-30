
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader
import os
import sys

# Add src to the Python path
sys.path.append('/content/drive/MyDrive/Colab Notebooks/PopularityPrediction')
from src.hgt_model import HGTPopularityPredictor

def test_model(data_dir, weights_dir):
    print("Loading graph data...")
    processed_path = os.path.join(data_dir, 'processed', 'train_heterodata.pt')
    data = torch.load(processed_path, weights_only=False)
    
    # Re-calculate Ground Truth
    edge_index = data['user', 'favorites', 'photo'].edge_index
    photo_popularity = degree(edge_index[1], num_nodes=data['photo'].num_nodes)
    data['photo'].y = photo_popularity.float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Trained HGT Model...")
    model = HGTPopularityPredictor(
        hidden_channels=64, 
        out_channels=1, 
        num_heads=4, 
        num_layers=2, 
        node_types=data.node_types, 
        metadata=data.metadata()
    ).to(device)

    # Load the saved weights
    weights_path = os.path.join(weights_dir, 'hgt_best_model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Use a NeighborLoader to sample a batch for testing
    loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_types},
        batch_size=32,          # Look at 32 photos at a time
        input_nodes='photo',
        shuffle=True
    )

    print("\n--- INFERENCE RESULTS ---")
    total_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            
            batch_size = batch['photo'].batch_size
            predictions = out[:batch_size]
            actuals = batch['photo'].y[:batch_size]
            
            # Print the first 5 examples from the first batch
            if num_batches == 0:
                print("Sample Predictions (First 5 Photos in Batch):")
                for i in range(min(5, batch_size)):
                    print(f"  Photo Node ID {batch['photo'].n_id[i].item()}: Predicted Likes = {predictions[i].item():.1f} | Actual Likes = {actuals[i].item():.0f}")
            
            # Calculate Mean Absolute Error (MAE)
            mae = F.l1_loss(predictions, actuals)
            total_mae += mae.item()
            num_batches += 1
            
    print(f"\nOverall Mean Absolute Error (MAE) across test batches: {total_mae/num_batches:.2f} likes")

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    WEIGHTS_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/models/saved_weights'
    
    test_model(DATA_DIR, WEIGHTS_DIR)