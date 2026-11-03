import abc
import logging
import pandas as pd
from pathlib import Path
from src.config import Config

log = logging.getLogger(__name__)

class BaseAdapter(abc.ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.active_cfg = cfg.active
        
        self.processed_dir = cfg.run.base_path / "data" / "processed" / self.dataset_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_dataframe(self, path_key: str, column_names: list[str]) -> pd.DataFrame:
        """Loads a raw dataset file using the path resolver and pandas kwargs from config."""
        path = self.cfg.path(path_key)
        if path is None:
            raise ValueError(f"Path for '{path_key}' is not defined in the config.")
            
        kwargs = self.active_cfg.read_csv_kwargs.copy()
        log.info(f"Loading {path_key} from {path}")
        
        df = pd.read_csv(path, **kwargs)
        
        # If the file has no header, apply the canonical names defined in config
        if kwargs.get("header") is None:
            df.columns = column_names
            
        return df

    @abc.abstractmethod
    def build_events(self) -> pd.DataFrame:
        """Returns a DataFrame with columns: ['post_id', 'user', 'time']"""
        pass

    @abc.abstractmethod
    def build_posts(self) -> pd.DataFrame:
        """Returns a DataFrame with columns: ['post_id', 'author', 't0']"""
        pass

    @abc.abstractmethod
    def build_edges(self) -> pd.DataFrame:
        """Returns a DataFrame with columns: ['source', 'target', 'created_at']"""
        pass

    def process_and_save(self):
        """Executes the adapter pipeline and saves canonical parquet files."""
        log.info(f"Processing dataset: {self.dataset_name} (Profile: {self.cfg.run.profile})")
        
        events_df = self.build_events()
        posts_df = self.build_posts()
        edges_df = self.build_edges()
        
        events_df.to_parquet(self.processed_dir / "events.parquet", index=False)
        posts_df.to_parquet(self.processed_dir / "posts.parquet", index=False)
        edges_df.to_parquet(self.processed_dir / "edges.parquet", index=False)
        
        log.info(f"Successfully saved canonical tables to {self.processed_dir}")