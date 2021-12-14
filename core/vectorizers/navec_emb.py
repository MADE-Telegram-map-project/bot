from multiprocessing import Pool
from typing import List

import numpy as np
from core.preprocessing import messages_generator, preprocess_message
from core.vectorizers.base import BaseEmbedder
from navec import Navec

DEFAULT_PATH_TO_MODEL = './messages2vec/models/navec_hudlit_v1_12B_500K_300d_100q.tar'


class NavecEmbedder(BaseEmbedder):
    def __init__(self, path_to_navec=DEFAULT_PATH_TO_MODEL, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_to_navec = path_to_navec
        self.navec = Navec.load(path_to_navec)

    def _preprocess_messages(self, messages: List[str]) -> List[List[str]]:
        cleaned_messages = []
        for m in messages:
            cleaned_m = preprocess_message(m).split()
            if len(cleaned_m) > 3:
                cleaned_messages.append(cleaned_m)
        return cleaned_messages

    def _message2vec(self, message: List[str]) -> np.ndarray:
        word_embs = [self.navec[w] for w in message if w in self.navec]
        if len(word_embs) == 0:
            return None
        message_emb = np.mean(word_embs, axis=0)
        return message_emb

    def channel2vec(self, messages: List[str]) -> np.ndarray:
        """ transform messages from channel to one embedding by averaging """
        mes_embs = []
        for m in messages:
            mvec = self._message2vec(m)
            if mvec is not None:
                mes_embs.append(mvec)
        channel_emb = np.mean(mes_embs, axis=0)
        return channel_emb


if __name__ == "__main__":
    # config = load_config("configs/base.yaml")
    path = "messages2vec/data/sorted_messages.csv"
    # messages = get_messages(path, 500, threads=12, tokenize=True, stemming=False)
    # mes = [x.message for x in messages]

    navemb = NavecEmbedder(messages_data=messages_generator(path))
    ch, embs = navemb.compute_embeddings(50)
    print(ch[:10])

    for x in embs[:10]:
        print(x[:5])
