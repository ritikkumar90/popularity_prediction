import pandas as pd
import numpy as np
import os

def temporal_split_with_labels(data_dir, obs_days=7, horizon_days=30, purge=True):
    print("Loading favorite markings for temporal splitting...")
    fav_path = os.path.join(data_dir, 'raw', 'flickr-all-photo-favorite-markings.txt.gz')
    splits_dir = os.path.join(data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    df = pd.read_csv(fav_path, sep=r'\s+', header=None,
                     names=['user_id', 'photo_id', 'timestamp'],
                     dtype={'user_id': np.int64, 'photo_id': np.int64, 'timestamp': np.float64})
    df = df.dropna(subset=['timestamp'])

    OBS = obs_days * 86400
    HOR = horizon_days * 86400
    
    print("Calculating temporal windows and censoring bounds...")
    df['start'] = df.groupby('photo_id')['timestamp'].transform('min')  # proxy for upload time
    df['age'] = df['timestamp'] - df['start']
    crawl_end = df['timestamp'].max()

    photos = df.groupby('photo_id')['start'].first().reset_index()
    # drop censored photos (must have full observation + horizon window available in data)
    photos = photos[photos.start + OBS + HOR <= crawl_end]              

    print(f"Assigning target labels (favorites between day {obs_days} and {obs_days + horizon_days})...")
    in_hor = df[(df.age > OBS) & (df.age <= OBS + HOR)]
    counts = in_hor.groupby('photo_id').size()
    photos['future_popularity'] = photos['photo_id'].map(counts).fillna(0).astype(np.int64)

    print("Sorting and allocating to Train/Val/Test...")
    photos = photos.sort_values('start').reset_index(drop=True)
    n = len(photos)
    i_val, i_test = int(n * 0.8), int(n * 0.9)
    
    photos['split'] = np.where(photos.index < i_val, 'train',
                      np.where(photos.index < i_test, 'val', 'test'))

    if purge:
        print("Purging temporal overlap between splits...")
        val_start = photos.start.iloc[i_val]
        test_start = photos.start.iloc[i_test]
        end = photos.start + OBS + HOR
        
        # labels of earlier splits must not reach into the next split's period
        photos = photos[~(((photos.split == 'train') & (end > val_start)) |
                          ((photos.split == 'val') & (end > test_start)))]

    print("Extracting observation edges...")
    obs = df[(df.age <= OBS) & df.photo_id.isin(photos.photo_id)].copy()
    obs['split'] = obs['photo_id'].map(photos.set_index('photo_id')['split'])

    print("Saving splits to disk...")
    for s in ['train', 'val', 'test']:
        obs[obs.split == s][['user_id', 'photo_id', 'timestamp']].to_csv(
            os.path.join(splits_dir, f'{s}_obs_edges.csv'), index=False)
        
        lab = photos[photos.split == s][['photo_id', 'start', 'future_popularity']]
        lab.to_csv(os.path.join(splits_dir, f'{s}_target_labels.csv'), index=False)
        
        print(f"{s.capitalize()} Split: {len(lab):,} photos, {(lab.future_popularity == 0).mean():.1%} zero-label, median {lab.future_popularity.median():.0f}")
        
    print("\nTemporal splitting complete!")

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    temporal_split_with_labels(DATA_DIR, obs_days=7, horizon_days=30, purge=True)