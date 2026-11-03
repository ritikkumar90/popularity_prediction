import os
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

SPLITS = ['train', 'val', 'test']
BINS_H = [1, 6, 24, 72, 168]  # hours since a photo's first favorite

def read_cols(path, cols, names):
    df = pd.read_csv(path, sep=r'\s+', header=None, usecols=cols, dtype=np.int64)
    df.columns = names
    return df

def idx(ids, col):
    v = col.to_numpy()
    pos = np.searchsorted(ids, v)
    assert (ids[pos.clip(max=len(ids) - 1)] == v).all(), "ID not in mapping"
    return torch.from_numpy(pos)

def load_hgt_graph(data_dir):
    sp = os.path.join(data_dir, 'splits')
    obs = pd.concat([pd.read_csv(os.path.join(sp, f'{s}_obs_edges.csv'),
                     dtype={'user_id': np.int64, 'photo_id': np.int64, 'timestamp': np.float64})
                     for s in SPLITS], ignore_index=True)
    lab = pd.concat([pd.read_csv(os.path.join(sp, f'{s}_target_labels.csv'),
                     usecols=['photo_id', 'future_popularity'],
                     dtype={'photo_id': np.int64, 'future_popularity': np.float32}).assign(split=s)
                     for s in SPLITS], ignore_index=True)

    g = read_cols(os.path.join(data_dir, 'raw', 'flickr-growth.txt.gz'),
                  [0, 1], ['follower', 'followed']).drop_duplicates()
    g = g[g.follower != g.followed]
    p = read_cols(os.path.join(data_dir, 'raw', 'flickr-all-photos.txt.gz'),
                  [0, 1], ['photo_id', 'uploader'])
    p = p[p.photo_id.isin(lab.photo_id)]            # only photos that belong to a split

    user_ids = np.unique(np.concatenate([g.follower, g.followed, obs.user_id, p.uploader]))
    photo_ids = np.unique(lab.photo_id)
    n_users, n_photos = len(user_ids), len(photo_ids)
    print(f"Users: {n_users:,} | Photos: {n_photos:,}")

    data = HeteroData()
    data['user', 'follows', 'user'].edge_index = torch.stack(
        [idx(user_ids, g.follower), idx(user_ids, g.followed)])
    data['user', 'favorites', 'photo'].edge_index = torch.stack(
        [idx(user_ids, obs.user_id), idx(photo_ids, obs.photo_id)])
    data['photo', 'uploaded_by', 'user'].edge_index = torch.stack(
        [idx(photo_ids, p.photo_id), idx(user_ids, p.uploader)])

    # user features: follow out-/in-degree, favorites given
    fe = data['user', 'follows', 'user'].edge_index
    ve = data['user', 'favorites', 'photo'].edge_index
    ux = torch.stack([torch.bincount(fe[0], minlength=n_users),
                      torch.bincount(fe[1], minlength=n_users),
                      torch.bincount(ve[0], minlength=n_users)], dim=1).float()
    data['user'].x = torch.log1p(ux)

    # photo features: favorites within first 1h/6h/24h/3d/7d + time of day of first favorite
    st = obs.groupby('photo_id')['timestamp'].min()
    age_h = ((obs['timestamp'] - obs['photo_id'].map(st)) / 3600).to_numpy()
    pi = ve[1].numpy()
    px = np.zeros((n_photos, len(BINS_H) + 2), dtype=np.float32)
    for j, h in enumerate(BINS_H):
        px[:, j] = np.log1p(np.bincount(pi[age_h <= h], minlength=n_photos))
    si = idx(photo_ids, st.index.to_series()).numpy()
    hod = (st.to_numpy() % 86400) / 86400 * 2 * np.pi
    px[si, -2], px[si, -1] = np.sin(hod), np.cos(hod)
    data['photo'].x = torch.from_numpy(px)

    # labels: log1p target, NaN where unlabeled, plus split masks
    y = torch.full((n_photos,), float('nan'))
    y[idx(photo_ids, lab.photo_id)] = torch.log1p(torch.from_numpy(lab.future_popularity.to_numpy()))
    data['photo'].y = y
    for s in SPLITS:
        m = torch.zeros(n_photos, dtype=torch.bool)
        m[idx(photo_ids, lab.photo_id[lab.split == s])] = True
        data['photo'][f'{s}_mask'] = m

    return T.ToUndirected(merge=False)(data)        # reverse edges so photos get uploader/user info

if __name__ == "__main__":
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data'
    data = load_hgt_graph(DATA_DIR)
    
    processed_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    save_path = os.path.join(processed_dir, 'heterodata.pt')
    
    torch.save(data, save_path)
    print(f"\nSaved graph to {save_path}")
    print(data)