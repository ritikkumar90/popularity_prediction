import pandas as pd
from src.data.adapters.base import BaseAdapter

class DiggAdapter(BaseAdapter):
    """Adapter for Digg 2009."""
    
    def build_events(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("votes", self.config.columns["votes"])
        df.rename(columns={"post": "post_id"}, inplace=True)
        return df[["post_id", "user", "time"]].sort_values("time")

    def build_posts(self) -> pd.DataFrame:
        events = self.build_events()
        posts = events.groupby("post_id", as_index=False).first()
        posts.rename(columns={"user": "author", "time": "t0"}, inplace=True)
        return posts[["post_id", "author", "t0"]].sort_values("t0")

    def build_edges(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("edges", self.config.columns["edges"])
        if self.config.edge_direction == "friend_to_user":
            df.rename(columns={"friend": "source", "user": "target", "time": "created_at"}, inplace=True)
        else:
            df.rename(columns={"user": "source", "friend": "target", "time": "created_at"}, inplace=True)
        return df[["source", "target", "created_at"]].sort_values("created_at")