import pandas as pd
import os

def print_data_schemas(data_dir):
    print("================ DATA SCHEMAS ================\n")
    
    # 1. Social Graph (Growth)
    growth_path = os.path.join(data_dir, 'flickr-growth.txt.gz')
    if os.path.exists(growth_path):
        print("1. flickr-growth.txt.gz (User-User Social Graph)")
        # Reading first 5 rows, space-separated, no headers in original data
        df_growth = pd.read_csv(growth_path, sep='\t| ', engine='python', header=None, nrows=5)
        df_growth.columns = ['follower_user_id', 'followed_user_id', 'timestamp'][:len(df_growth.columns)]
        print(df_growth.head(), "\n")
    
    # 2. Favorite Markings (Cascades)
    fav_path = os.path.join(data_dir, 'flickr-all-photo-favorite-markings.txt.gz')
    if os.path.exists(fav_path):
        print("2. flickr-all-photo-favorite-markings.txt.gz (User-Photo Favorites)")
        df_fav = pd.read_csv(fav_path, sep='\t| ', engine='python', header=None, nrows=5)
        df_fav.columns = ['user_id', 'photo_id', 'timestamp'][:len(df_fav.columns)]
        print(df_fav.head(), "\n")
        
    # # 3. Photo Metadata (Uploads)
    # photo_path = os.path.join(data_dir, 'flickr-all-photos.txt.gz')
    # if os.path.exists(photo_path):
    #     print("3. flickr-all-photos.txt.gz (Photo Metadata)")
    #     df_photos = pd.read_csv(photo_path, sep='\t| ', engine='python', header=None, nrows=5)
    #     df_photos.columns = ['photo_id', 'uploader_user_id', 'timestamp'][:len(df_photos.columns)]
    #     print(df_photos.head(), "\n")
    # 3. Photo Metadata (Uploads)
    photo_path = os.path.join(data_dir, 'flickr-all-photos.txt.gz')
    if os.path.exists(photo_path):
        print("3. flickr-all-photos.txt.gz (Photo Metadata)")
        df_photos = pd.read_csv(photo_path, sep=r'\t| ', engine='python', header=None, nrows=5)
        
        # Let pandas use default column numbers (0 to 5) so we can see what is actually inside!
        print(df_photos.head(), "\n")

def print_meta_paths():
    print("================ HGT META-PATHS ================\n")
    print("To build your Heterogeneous Graph Transformer (HGT), you will use these explicit meta-paths:\n")
    
    print("Node Types:")
    print("  - 'User'")
    print("  - 'Photo'\n")
    
    print("Edge Types (Meta-paths):")
    print("  1. ('User', 'follows', 'User')     <- From flickr-growth")
    print("  2. ('User', 'favorites', 'Photo')  <- From favorite-markings (Target for Popularity)")
    print("  3. ('Photo', 'favorited_by', 'User') <- Reverse edge (Required for PyG message passing)")
    print("  4. ('User', 'uploads', 'Photo')    <- From flickr-all-photos")
    print("  5. ('Photo', 'uploaded_by', 'User')  <- Reverse edge\n")

if __name__ == "__main__":
    # Point to your specific data folder
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw'
    
    print_data_schemas(DATA_DIR)
    print_meta_paths()