import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os

def load_hgt_graph(data_dir):
    print("Loading datasets...")
    
    # 1. Load Social Graph (User follows User) from raw/
    growth_path = os.path.join(data_dir, 'raw_mini', 'flickr-growth.txt.gz')
    df_growth = pd.read_csv(growth_path, sep=r'\t| ', engine='python', header=None, usecols=[0, 1])
    df_growth.columns = ['follower', 'followed']
    
    # 2. Load Train Favorites (User favorites Photo) from splits/
    train_fav_path = os.path.join(data_dir, 'splits', 'train_favorites.csv')
    df_fav = pd.read_csv(train_fav_path, usecols=['user_id', 'photo_id'])
    
    # 3. Load Photo Uploads (User uploads Photo) from raw/
    photo_path = os.path.join(data_dir, 'raw_mini', 'flickr-all-photos.txt.gz')
    df_photos = pd.read_csv(photo_path, sep=r'\t| ', engine='python', header=None, usecols=[0, 1])
    df_photos.columns = ['photo_id', 'uploader']
    
    print("Creating ID mappings...")
    # Extract unique IDs across all files
    unique_users = set(df_growth['follower']).union(set(df_growth['followed'])) \
                   .union(set(df_fav['user_id'])).union(set(df_photos['uploader']))
    unique_photos = set(df_fav['photo_id']).union(set(df_photos['photo_id']))
    
    # Create mapping dictionaries (Raw ID -> Contiguous Index 0, 1, 2...)
    user_mapping = {raw_id: i for i, raw_id in enumerate(unique_users)}
    photo_mapping = {raw_id: i for i, raw_id in enumerate(unique_photos)}
    
    num_users = len(user_mapping)
    num_photos = len(photo_mapping)
    print(f"Total Unique Users: {num_users:,} | Total Unique Photos: {num_photos:,}")
    
    print("Initializing PyG HeteroData...")
    data = HeteroData()
    
    # Initialize placeholder node features (vectors of 1s, dim=16)
    feat_dim = 16
    data['user'].x = torch.ones(num_users, feat_dim)
    data['photo'].x = torch.ones(num_photos, feat_dim)
    
    print("Adding edges...")
    # Edge: User follows User
    src_follows = [user_mapping[uid] for uid in df_growth['follower']]
    dst_follows = [user_mapping[uid] for uid in df_growth['followed']]
    data['user', 'follows', 'user'].edge_index = torch.tensor([src_follows, dst_follows], dtype=torch.long)
    
    # Edge: User favorites Photo
    src_fav = [user_mapping[uid] for uid in df_fav['user_id']]
    dst_fav = [photo_mapping[pid] for pid in df_fav['photo_id']]
    data['user', 'favorites', 'photo'].edge_index = torch.tensor([src_fav, dst_fav], dtype=torch.long)
    
    # Edge: Photo uploaded_by User (Reverse direction for PyG message passing)
    src_upload = [photo_mapping[pid] for pid in df_photos['photo_id']]
    dst_upload = [user_mapping[uid] for uid in df_photos['uploader']]
    data['photo', 'uploaded_by', 'user'].edge_index = torch.tensor([src_upload, dst_upload], dtype=torch.long)
    
    print(f"\nHeteroData object built successfully!\n{data}")
    return data, user_mapping, photo_mapping

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    data, _, _ = load_hgt_graph(DATA_DIR)
    
    # Save the PyG graph to disk
    processed_dir = os.path.join(DATA_DIR, 'processed')
    torch.save(data, os.path.join(processed_dir, 'train_heterodata.pt'))
    print(f"\nSaved PyG graph to {processed_dir}/train_heterodata.pt")