import re
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from core.read_write import read_numpy_array
from core.config import load_config
from core.entities.data import MainConfig
# from core.vectorizers import TransEmbedder



class Ranker:
    def __init__(self, config: MainConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._logger.info("Load embeddings...")
        self.emb = read_numpy_array(self.config.data.emb_labse)
        self._logger.info("Load channels...")
        self.chans = read_numpy_array(self.config.data.ch_labse)
        meta = pd.read_csv(self.config.data.channels)
        meta = meta[meta.channel_id.isin(self.chans)]
        self.meta = meta
        self.channel_id2username = dict(zip(self.meta.channel_id, self.meta.link))
        ordered_chans = [self.channel_id2username[x] for x in self.chans]
        self.username2emb = dict(zip(ordered_chans, self.emb))

    def get_channel_embedding(self, username: str):
        username = self._preprocess_username(username)
        if username is None:
            return None

        if username in self.username2emb:
            emb = self.username2emb[username]
        else:
            emb = None  # TODO get embedding of new channel (parse and process)
        return username, emb

    def get_closest_channels(self, query: str, topn=5):
        username, emb = self.get_channel_embedding(query)
        if emb is None:
            return None

        cosine_similarities = linear_kernel(emb.reshape(1, -1), self.emb).squeeze()
        topn_idx = np.argsort(cosine_similarities)[-topn:]
        closest_channel_ids = self.chans[topn_idx]
        closest_usernames = [self.channel_id2username[i] for i in closest_channel_ids]
        df = self.meta[
            (self.meta.link.isin(closest_usernames)) &
            (self.meta.link != username) &
            (~self.meta.link.str.endswith("bot"))
        ]
        channels = df[["link", "title"]].values
        return channels

    def get_random_channels(self, n=5):
        df = self.meta.sample(n)
        channels = df[["link", "title"]].values
        return channels

    def get_channel_by_description(self, text: str):

        return None

    @staticmethod
    def _preprocess_username(username: str) -> str:
        """ can process such username formats
        - https://t.me/username
        - http://t.me/username
        - @username
        - t.me/username
        - username
        """
        match = re.search("(t\\.me/|@)([a-zA-Z0-9_]{5,32})", username)
        if match is None:
            match = re.match("(^[a-zA-Z0-9_]{5,32})\s?$", username)

        username = match if match is None else match.groups()[-1]
        return username


if __name__ == "__main__":
    PATH_TO_CONFIG = "config.yaml"
    config = load_config(PATH_TO_CONFIG)
    print(config)
    ranker = Ranker(config)

    username = "sliv_halyavy"
    top = ranker.get_closest_channels(username)
    if top is None:
        print("No embedding")
        exit(1)
    # df = ranker.meta[ranker.meta.link.isin(top)]
    print(top)
