import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path

# from src.ml.dataset import get_dataloaders
from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders       # no config object needed

from src.ml.models.rgcn import PopularityRGCN

log = logging.getLogger(__name__)

# --- Default Configurations ---
DEFAULT_BASE_PATH = "."
DEFAULT_DATASET_NAME = "flickr"
DEFAULT_OUTPUT_DIR = "outputs"

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 50

def train_rgcn():
    parser = argparse.ArgumentParser(description="Train R-GCN for Popularity Prediction")
    
    # Environment Arguments
    parser.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH, help="Base path for the project")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, help="Target dataset name (e.g., flickr, digg)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory inside base_path")
    
    # R-GCN Hyperparameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Hidden dimension of the network")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of message passing layers")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Setup directories
    base_dir = Path(args.base_path)
    ml_dir = base_dir / args.output_dir / args.dataset_name / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    # get_dataloaders automatically finds the latest hash dir if config_hash is not provided
    train_loader, val_loader, test_loader = get_dataloaders(
        base_path=args.base_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check your splits and extraction.")

    # 2. Initialize Model using metadata from the first batch
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device) # Make sure it's on the correct device
    
    log.info(f"Initializing RGCN with hyperparams: hidden_channels={args.hidden_channels}, num_layers={args.num_layers}, lr={args.lr}, batch_size={args.batch_size}")
    
    model = PopularityRGCN(
        metadata=sample_batch.metadata(),
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)

    # --- THE FIX: Dummy forward pass to trigger PyG lazy initialization ---
    with torch.no_grad():
        model(sample_batch.x_dict, sample_batch.edge_index_dict)

    # 3. Setup Optimizer and Loss (Now it will track ALL parameters)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    model_save_path = ml_dir / "rgcn_best.pt"

    # 4. Training Loop
    log.info("Starting R-GCN training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            preds = model(batch.x_dict, batch.edge_index_dict)
            
            # Squeeze batch.y to prevent broadcasting mismatches
            loss = criterion(preds, batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(train_loader.dataset)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x_dict, batch.edge_index_dict)
                loss = criterion(preds, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                
                # Compute MAE on original scale
                true_vals = torch.expm1(batch.y.squeeze())
                pred_vals = torch.expm1(preds)
                val_mae += torch.abs(true_vals - pred_vals).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        log.info(f"Epoch {epoch:03d}/{args.epochs} | Train Loss (MSLE): {train_loss:.4f} | Val Loss (MSLE): {val_loss:.4f} | Val MAE: {val_mae:.2f}")

        # 6. Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            log.info(f" -> Saved new best model to {model_save_path}")

    # 7. Test Evaluation
    log.info("--- Training Complete. Evaluating on Test Set ---")
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))
    model.eval()
    
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict)
            test_loss += criterion(preds, batch.y.squeeze()).item() * batch.num_graphs
            
            true_vals = torch.expm1(batch.y.squeeze())
            pred_vals = torch.expm1(preds)
            test_mae += torch.abs(true_vals - pred_vals).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    
    print(f"\\n=== FINAL LOSS STATS ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: R-GCN (hidden={args.hidden_channels}, layers={args.num_layers}, lr={args.lr})")
    print(f"Validation Loss (MSLE): {best_val_loss:.4f}")
    print(f"Test Loss (MSLE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"========================\\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_rgcn()
