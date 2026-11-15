import torch
import torch.nn as nn

# Import your model and batch helper[cite: 2]
from src.ml.models.hgt import PopularityHGT, get_batch_dict
# Import your actual dataloader identical to train.py[cite: 3]
from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Use your exact data pipeline to get ONE correctly formatted batch[cite: 3]
print("Loading data via dataloader...")
train_loader, _, _ = get_dataloaders(
    base_path=".",
    dataset_name="digg",
    config_hash="1f3dea3067",
    batch_size=32,
    num_workers=0,
    output_dir="outputs"
)

# Grab the very first batch and ignore the rest of the dataset
single_batch = next(iter(train_loader)).to(device)

# 2. Initialize the model with ALL regularisation forced to 0.0[cite: 2]
model = PopularityHGT(
    metadata=single_batch.metadata(),
    dropout=0.0,       # Turn off head dropout[cite: 2]
    input_dropout=0.0, # Turn off input dropout[cite: 2]
    conv_dropout=0.0,  # Turn off message-passing dropout[cite: 2]
    edge_dropout=0.0,  # Turn off DropEdge[cite: 2]
).to(device)

# Perform the dummy forward pass to initialize the lazy Linear layers
model.eval()
with torch.no_grad():
    model(single_batch.x_dict, single_batch.edge_index_dict, get_batch_dict(single_batch))

# 3. Setup Optimizer and Loss
# Ensure weight decay is exactly 0.0 so the model is free to memorize
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
criterion = nn.MSELoss() 

# 4. The 500-step loop on the EXACT SAME batch
model.train() 
print("Starting single-batch overfit test...")
for epoch in range(501):
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass
    preds = model(single_batch.x_dict, single_batch.edge_index_dict, get_batch_dict(single_batch))
    loss = criterion(preds, single_batch.y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")