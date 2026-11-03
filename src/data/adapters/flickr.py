import pandas as pd
from src.data.adapters.base import BaseAdapter

class FlickrAdapter(BaseAdapter):
    def build_events(self) -> pd.DataFrame:
        # Load using canonical names defined in config: ['user', 'post', 'time']
        df = self.load_raw_dataframe("votes", self.active_cfg.columns.votes)
        
        # Convert time to seconds (Flickr's favorites time is in epoch days)
        if self.active_cfg.time_format != "unix_seconds":
            df["time"] = df["time"] * 86400 
            
        # Rename 'post' to 'post_id' for downstream canonical consistency
        df.rename(columns={"post": "post_id"}, inplace=True)
        
        return df[["post_id", "user", "time"]].sort_values("time")

    def build_posts(self) -> pd.DataFrame:
        events = self.build_events()
        # Author is the first person to interact (favorite) the photo
        posts = events.groupby("post_id", as_index=False).first()
        posts.rename(columns={"user": "author", "time": "t0"}, inplace=True)
        
        return posts[["post_id", "author", "t0"]].sort_values("t0")

    def build_edges(self) -> pd.DataFrame:
        # Load using canonical names defined in config: ['follower', 'followed', 'created_at']
        df = self.load_raw_dataframe("edges", self.active_cfg.columns.edges)
        
        # Convert YYYY-MM-DD to unix seconds for direct comparison with events
        df["created_at"] = pd.to_datetime(df["created_at"]).astype('int64') // 10**9
        
        # Dynamically rename based on config edge directions
        df.rename(columns={
            self.active_cfg.edge_source: "source", 
            self.active_cfg.edge_target: "target"
        }, inplace=True)
            
        return df[["source", "target", "created_at"]].sort_values("created_at")