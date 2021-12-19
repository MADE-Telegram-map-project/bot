import logging
import re
from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd
import pkg_resources
from sklearn.metrics.pairwise import cosine_similarity

from core.config import load_config
from core.entities.data import MainConfig
from core.read_write import read_numpy_array
from core.vectorizers import TransEmbedder
from core.web_parsing import parse_channel_web

SIM_CUTOFF_DEFAULT = 0.7
SIM_CUTOFF_DESCR = 0.5


class Ranker:
    def __init__(self, config: MainConfig, use_trans=True, sim_cutoff=SIM_CUTOFF_DEFAULT):
        self.config = config
        self.use_trans = use_trans
        self.sim_cutoff = sim_cutoff

        self.distance_func = cosine_similarity
        self._logger = logging.getLogger(__name__)
        self._logger.info("Load embeddings...")
        self.emb = read_numpy_array(
            pkg_resources.resource_filename(
                __name__, self.config.data.emb_full))
        self.chan_ids = read_numpy_array(
            pkg_resources.resource_filename(
                __name__, self.config.data.ch_full))
        self._logger.info("Load channels...")

        meta = pd.read_csv(pkg_resources.resource_filename(__name__, self.config.data.channels))
        meta = meta[meta.channel_id.isin(self.chan_ids)]
        self.meta = meta
        self.channel_id2username = dict(zip(self.meta.channel_id, self.meta.link))
        self.username2channel_id = {v: k for k, v in self.channel_id2username.items()}
        # ordered_chans = [self.channel_id2username[x] for x in self.chan_ids]
        # self.username2emb = dict(zip(ordered_chans, self.emb))

        self.username2similar = defaultdict(list)
        self._precalculate_sim_scores()
        self.transformer = None
        if use_trans:
            self._logger.info("Load transformer model...")
            self.transformer = TransEmbedder(messages_data=None, load_data=False)

    def _precalculate_sim_scores(self):
        cosine_similarities = self.distance_func(self.emb)

        res_sort = np.sort(cosine_similarities)
        res_argsort = np.argsort(cosine_similarities)
        mask = res_sort > self.sim_cutoff

        for i in range(len(cosine_similarities)):
            indexes_of_sim = res_argsort[i][mask[i]][:-1]
            indexes_of_sim = list(indexes_of_sim[::-1])  # descending
            similar_channel_ids = [self.chan_ids[x] for x in indexes_of_sim]
            channel_id = self.chan_ids[i]
            username = self.channel_id2username[channel_id]
            self.username2similar[username] = similar_channel_ids

    def get_channels_by_username(self, query: str, k=5):
        """ main func: search by username"""
        username = self._preprocess_username(query)
        if username is None:
            self._logger.warn("Preprocessed username is None")
            return None  # TODO status of unknown channel

        if username in self.username2similar:
            sim_chan_indexes = self.known_channel_processing(username, k)
        else:
            if not self.use_trans:
                return None  # TODO status of no loaded transformer
            sim_chan_indexes = self.unknown_channel_processing(username, k)

        if sim_chan_indexes is None:
            return None

        df = self.meta[
            (self.meta.channel_id.isin(sim_chan_indexes)) &
            (self.meta.link != username) &
            (~self.meta.link.str.endswith("bot"))
        ]
        channels = df[["link", "title"]].values
        return channels

    def get_channels_by_description(
            self, description: str, k=5, sim_cutoff=SIM_CUTOFF_DESCR):
        """ main func: search by description """
        if not self.use_trans:
            return None  # TODO status of no loaded transformer

        emb = self.description_vectorize([description])
        sim_chan_indexes = self.search_by_embedding(emb, sim_cutoff=sim_cutoff)
        if len(sim_chan_indexes) == 0:
            return None  # TODO status no channels for such descr, try to extend it

        sim_chan_indexes = sim_chan_indexes[:k]
        df = self.meta[
            (self.meta.channel_id.isin(sim_chan_indexes)) &
            (~self.meta.link.str.endswith("bot"))
        ]
        channels = df[["link", "title"]].values
        return channels

    def description_vectorize(self, description: List[str]) -> Union[np.ndarray, None]:
        assert isinstance(description, list)
        emb = self.transformer.description2vec(description)
        return emb

    def known_channel_processing(self, username: str, k=5) -> List[int]:
        """ return indexes of similar channels """
        sim_channel_ids = self.username2similar[username]
        sim_channel_ids_top = sim_channel_ids[:k]
        return sim_channel_ids_top

    def unknown_channel_processing(self, username: str, k=5) -> List[int]:
        emb = self.get_unk_channel_vec(username)
        if emb is None:
            # TODO channel noInfo status
            return None
        sim_chan_indexes = self.search_by_embedding(emb)
        self.username2similar[username] = sim_chan_indexes
        sim_chan_indexes = sim_chan_indexes[:k]  # TODO modify for more-button or not this channel will be in history when more-button will be pressed
        return sim_chan_indexes

    def get_unk_channel_vec(self, username: str):
        header, messages = parse_channel_web(username)
        messages.append(header)
        if len(messages) == 0:
            emb = None  # TODO status of channel that hasn't good description
        else:
            emb = self.description_vectorize(messages)
        return emb

    def search_by_embedding(self, emb: np.ndarray, sim_cutoff=None):
        assert hasattr(emb, "__len__") and len(emb) > 0
        cosine_similarities = self.distance_func(emb.reshape(1, -1), self.emb).squeeze()
        res_sort = np.sort(cosine_similarities)
        res_argsort = np.argsort(cosine_similarities)
        sim_cutoff = sim_cutoff or self.sim_cutoff
        mask = res_sort > sim_cutoff

        indexes_of_sim = res_argsort[mask]
        indexes_of_sim = list(indexes_of_sim[::-1])  # descending
        similar_channel_ids = [self.chan_ids[x] for x in indexes_of_sim]

        return similar_channel_ids

    def get_random_channels(self, n=5):
        df = self.meta.sample(n)
        channels = df[["link", "title"]].values
        return channels

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
            match = re.match("(^[a-zA-Z0-9_]{5,32})\\s?$", username)

        username = match if match is None else match.groups()[-1]
        return username


if __name__ == "__main__":
    PATH_TO_CONFIG = "config.yaml"
    config = load_config(PATH_TO_CONFIG)
    # print(config)
    ranker = Ranker(config)
    print(list(ranker.username2similar.keys())[:50])
    print("ranker loaded\n\n")

    lst = ["psychics", "https://t.me/latinapopacanski"]
    for username in lst:
        print(username)
        top = ranker.get_channels_by_username(username)
        if top is None:
            print("No embedding")
            exit(1)
        print(top)
        print("\n")

    descriptions = [
        "латинский язык и древний рим", "канал про медицину", "медицина", 
        "политика", "работа в париже", "канал про рыбалку",
    ]
    for descr in descriptions:
        print(descr)
        top = ranker.get_channels_by_description(descr)
        if top is None:
            print("No embedding")
            continue
            # exit(1)
        print(top)
