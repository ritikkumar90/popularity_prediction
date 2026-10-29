import pandas as pd
import os

def create_mini_data(data_dir, mini_dir, num_favs=5000):
    os.makedirs(mini_dir, exist_ok=True)
    raw_dir = os.path.join(data_dir, 'raw')
    
    print(f"1. Extracting first {num_favs} favorites...")
    fav_path = os.path.join(raw_dir, 'flickr-all-photo-favorite-markings.txt.gz')
    df_fav = pd.read_csv(fav_path, sep=r'\t| ', engine='python', header=None, nrows=num_favs)
    df_fav.columns = ['user_id', 'photo_id', 'timestamp']
    
    valid_users = set(df_fav['user_id'])
    valid_photos = set(df_fav['photo_id'])
    print(f"   -> Found {len(valid_users)} unique users and {len(valid_photos)} unique photos.")
    
    print("2. Filtering Social Graph (User-User) to match these users...")
    growth_path = os.path.join(raw_dir, 'flickr-growth.txt.gz')
    # Read in chunks to save RAM
    chunk_iter = pd.read_csv(growth_path, sep=r'\t| ', engine='python', header=None, chunksize=100000)
    mini_growth = []
    for chunk in chunk_iter:
        chunk.columns = ['follower', 'followed', 'timestamp']
        # Keep edge only if both users exist in our mini favorites list
        filtered_chunk = chunk[chunk['follower'].isin(valid_users) & chunk['followed'].isin(valid_users)]
        mini_growth.append(filtered_chunk)
    df_growth_mini = pd.concat(mini_growth)
    print(f"   -> Kept {len(df_growth_mini)} social connections.")
    
    print("3. Filtering Photo Metadata to match these photos...")
    photo_path = os.path.join(raw_dir, 'flickr-all-photos.txt.gz')
    chunk_iter = pd.read_csv(photo_path, sep=r'\t| ', engine='python', header=None, chunksize=100000)
    mini_photos = []
    for chunk in chunk_iter:
        chunk.columns = ['photo_id', 'uploader_user_id', 'timestamp', 'col3', 'col4', 'col5'][:len(chunk.columns)]
        filtered_chunk = chunk[chunk['photo_id'].isin(valid_photos)]
        mini_photos.append(filtered_chunk)
    df_photos_mini = pd.concat(mini_photos)
    print(f"   -> Kept {len(df_photos_mini)} photo metadata records.")
    
    print("\nSaving Mini Dataset...")
    df_fav.to_csv(os.path.join(mini_dir, 'flickr-all-photo-favorite-markings.txt.gz'), sep='\t', index=False, header=False, compression='gzip')
    df_growth_mini.to_csv(os.path.join(mini_dir, 'flickr-growth.txt.gz'), sep='\t', index=False, header=False, compression='gzip')
    df_photos_mini.to_csv(os.path.join(mini_dir, 'flickr-all-photos.txt.gz'), sep='\t', index=False, header=False, compression='gzip')
    print(f"Done! Mini dataset saved to {mini_dir}")

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    MINI_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw_mini'
    create_mini_data(DATA_DIR, MINI_DIR)