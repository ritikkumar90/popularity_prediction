import pandas as pd
import os

def print_data_schemas(data_dir):
    print("================ DATA SCHEMAS ================\n")
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
        
    # Dynamically find and read whatever dataset files exist in the digg folder
    files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.csv', '.gz'))]
    
    if not files:
        print(f"No data files found inside {data_dir}")
        return
        
    for i, file_name in enumerate(sorted(files), 1):
        file_path = os.path.join(data_dir, file_name)
        print(f"{i}. {file_name}")
        
        try:
            # Read standard CSVs or space-separated TXT/GZ files
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=5)
            else:
                df = pd.read_csv(file_path, sep=r'\s+', header=None, nrows=5)
            print(df)
            print("\n")
        except Exception as e:
            print(f"  [Error reading file: {e}]\n")

def print_meta_paths():
    print("================ HGT META-PATHS ================\n")
    print("To build your Heterogeneous Graph Transformer (HGT) for Digg, you will use these explicit meta-paths:\n")
    
    print("Node Types:")
    print("  - 'User'")
    print("  - 'Story'\n")
    
    print("Edge Types (Meta-paths):")
    print("  1. ('User', 'follows', 'User')       <- From social/friends graph")
    print("  2. ('User', 'votes_for', 'Story')    <- From votes/actions (Target for Popularity)")
    print("  3. ('Story', 'voted_by', 'User')     <- Reverse edge (Required for PyG message passing)")
    print("  4. ('User', 'submits', 'Story')      <- From story metadata (if available)")
    print("  5. ('Story', 'submitted_by', 'User') <- Reverse edge\n")

if __name__ == "__main__":
    # Point to the specific digg data folder
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/PopularityPrediction/data/raw/digg'
    
    print_data_schemas(DATA_DIR)
    print_meta_paths()
