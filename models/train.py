import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader
import os
import sys

# Add src to the Python path so we can import the model
sys.path.append('/content/drive/MyDrive/Colab Notebooks/PopularityPrediction')
from src.hgt_model import HGTPopularityPredictor

def train_hgt(data_dir, weights_dir, epochs=10):
    print("Loading preprocessed HeteroData graph (Keeping on CPU)...")
    processed_path = os.path.join(data_dir, 'processed', 'train_heterodata.pt')
    
    # Load graph onto CPU RAM, not GPU
    data = torch.load(processed_path, weights_only=False)
    
    # Calculate Ground Truth (Popularity = Number of favorites per photo)
    edge_index = data['user', 'favorites', 'photo'].edge_index
    photo_popularity = degree(edge_index[1], num_nodes=data['photo'].num_nodes)
    data['photo'].y = photo_popularity.float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Setting up NeighborLoader for mini-batching...")
    # Sample 15 neighbors in the first hop, and 10 in the second hop for a 2-layer HGT
    loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_types},
        batch_size=256,         # Number of target photos per batch
        input_nodes='photo',    # We are predicting values for 'photo' nodes
        shuffle=True,
        num_workers=2           # Speed up data loading
    )

    print("Initializing HGT Model...")
    model = HGTPopularityPredictor(
        hidden_channels=64, 
        out_channels=1, 
        num_heads=4, 
        num_layers=2, 
        node_types=data.node_types, 
        metadata=data.metadata()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    
    print(f"Starting Training on {device}...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        num_batches = 0
        
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            # Move ONLY the small sampled mini-batch to the GPU
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # The loader puts the target nodes at the beginning of the batch.
            # We slice it so we only calculate loss on the target batch_size, 
            # ignoring the extra neighbors pulled in for context.
            batch_size = batch['photo'].batch_size
            
            # Calculate Loss (MSE for regression)
            loss = F.mse_loss(out[:batch_size], batch['photo'].y[:batch_size])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            # Add this print statement to track batch progress
            if batch_idx % 500 == 0 or batch_idx == total_batches:
              print(f"   -> Epoch {epoch} | Batch {batch_idx}/{total_batches} | Current Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch:03d}/{epochs:03d}, Avg Loss: {avg_loss:.4f}")

    # Save the trained model weights
    save_path = os.path.join(weights_dir, 'hgt_best_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Weights saved to {save_path}")

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    WEIGHTS_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/models/saved_weights'
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Reduced epochs to 10 for standard mini-batch training speed
    train_hgt(DATA_DIR, WEIGHTS_DIR, epochs=10)
