"""Flickr adapter (Mislove et al., WWW 2009 Flickr data).

Raw files (header-less, whitespace separated):
  paths.votes   flickr-all-photo-favorite-markings : user, post, time
  paths.edges   flickr-growth                      : two user ids, date (YYYY-MM-DD)
  paths.owners  flickr-all-photos (optional)       : photo_id, upload_time, owner,
                                                     views, comments, favorites

Time
----
cfg time_format ('date:%Y-%m-%d') describes the EDGE dates. Favorite times in this
dataset are relative seconds since the crawl started on 2006-11-02. convert_event_time() 
adds the Unix timestamp for 2006-11-02 (1162425600) to align them with the edges. BaseAdapter 
does the rest (cleaning, de-duplication, sorting, edge lag).

Authors (cfg author_rule)
-------------------------
first_event : author = first user to favorite the photo (a proxy).
owner_file  : author = real photo owner from paths.owners.

The owners file also holds views / comments / favorites. Those are totals at crawl
time and include events after t_obs, so they are never read into any output table
(they would leak the label).
"""
import logging

import pandas as pd

from src.data.adapters.base import BaseAdapter

log = logging.getLogger(__name__)

# Column order of flickr-all-photos, as documented by the dataset authors.
PHOTO_COLUMNS = ["post_id", "upload_time", "owner", "views", "comments", "favorites"]
CRAWL_START_UNIX = 1162425600.0  # Unix timestamp for 2006-11-02


class FlickrAdapter(BaseAdapter):
    def convert_event_time(self, values: pd.Series) -> pd.Series:
        # Shift relative elapsed seconds to standard Unix time starting 2006-11-02
        return pd.to_numeric(values, errors="coerce").astype("float64") + CRAWL_START_UNIX

    def build_events(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("votes", self.active_cfg.columns.votes)
        df = df.rename(columns={"post": "post_id"})
        return df[["post_id", "user", "time"]]

    def build_posts(self, events: pd.DataFrame) -> pd.DataFrame:
        # `events` is cleaned and sorted by (time, post_id, user), so the first row
        # of each post is its earliest favorite. t0 = time of that first favorite.
        first = events.drop_duplicates("post_id", keep="first")
        posts = first.rename(columns={"user": "author", "time": "t0"})[["post_id", "author", "t0"]]

        if self.active_cfg.author_rule == "owner_file":
            owners = self._load_photo_owners()
            n0 = len(posts)
            # Inner merge keeps the id dtype intact; posts without a known owner are dropped.
            posts = posts.drop(columns="author").merge(owners, on="post_id", how="inner")
            log.info("Photo owners used as authors: %d of %d posts have a known owner", len(posts), n0)
            posts = posts[["post_id", "author", "t0"]]
        return posts

    def build_edges(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("edges", self.active_cfg.columns.edges)

        # created_at stays as the raw date string; BaseAdapter parses it with
        # time_format and adds the 1-day lag so an edge only becomes visible after its day.
        # Which raw column is source/target comes from config (information flows source -> target).
        df = df.rename(columns={
            self.active_cfg.edge_source: "source",
            self.active_cfg.edge_target: "target",
        })
        return df[["source", "target", "created_at"]]

    def _load_photo_owners(self) -> pd.DataFrame:
        df = self.load_raw_dataframe("owners", PHOTO_COLUMNS)
        # Keep only id and owner; the count columns would leak future information.
        owners = df[["post_id", "owner"]].drop_duplicates("post_id", keep="first")
        return owners.rename(columns={"owner": "author"})