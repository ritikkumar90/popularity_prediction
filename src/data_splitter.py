
import pandas as pd
import os

def temporal_split(data_dir):
    print("Loading favorite markings for temporal splitting...")
    # fav_path = os.path.join(data_dir, 'raw', 'flickr-all-photo-favorite-markings.txt.gz')
    fav_path = os.path.join(data_dir, 'raw_mini', 'flickr-all-photo-favorite-markings.txt.gz')
    splits_dir = os.path.join(data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # Load the favorites log
    df_fav = pd.read_csv(fav_path, sep=r'\t| ', engine='python', header=None)
    df_fav.columns = ['user_id', 'photo_id', 'timestamp']
    
    # Ensure timestamp is numeric for sorting
    df_fav['timestamp'] = pd.to_numeric(df_fav['timestamp'], errors='coerce')
    df_fav = df_fav.dropna(subset=['timestamp'])
    
    print(f"Total favorite interactions: {len(df_fav):,}")
    
    # Sort chronologically
    df_fav = df_fav.sort_values(by='timestamp')
    
    # Calculate index cutoffs for 80/10/10 split
    n = len(df_fav)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    df_train = df_fav.iloc[:train_end]
    df_val = df_fav.iloc[train_end:val_end]
    df_test = df_fav.iloc[val_end:]
    
    print("\nSplit Complete:")
    print(f"Train: {len(df_train):,} edges (until timestamp {df_train['timestamp'].max()})")
    print(f"Val:   {len(df_val):,} edges (until timestamp {df_val['timestamp'].max()})")
    print(f"Test:  {len(df_test):,} edges (until timestamp {df_test['timestamp'].max()})")
    
    # Save the splits
    print("\nSaving splits to data/splits/...")
    df_train.to_csv(os.path.join(splits_dir, 'train_favorites.csv'), index=False)
    df_val.to_csv(os.path.join(splits_dir, 'val_favorites.csv'), index=False)
    df_test.to_csv(os.path.join(splits_dir, 'test_favorites.csv'), index=False)
    print("Done!")

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    temporal_split(DATA_DIR)