from typing import Iterable, List

import numpy as np
import tqdm

from core.read_write import MessageData, messages_generator
from core.preprocessing import drop_links

MIN_MESSAGE_NUM = 20  # # min number of messages in one channel
MIN_WORD_NUM = 7  # min number of word in one message


class BaseEmbedder:
    """ output of _preprocess_messages will pass to channel2vec """
    def __init__(self, messages_data: Iterable[MessageData]):
        self.messages_data = messages_data
        self._get_cur()

    def _get_cur(self, cur_mes: MessageData = None):
        cur_mes = cur_mes or next(self.messages_data)
        self.cur_ch = cur_mes.channel_id
        self.cur_messages = [cur_mes.message]

    def _preprocess_messages(self, messages: List[str]) -> List[str]:
        """ template function, must be replaced by true preprocessing func """
        cleaned_messages = []
        for m in messages:
            cm = drop_links(m)
            if len(cm.split()) > MIN_WORD_NUM:
                cleaned_messages.append(cm)
        return cleaned_messages

    def channel2vec(self, messages):
        """ template function, must be replaced by true channel embedder """
        return np.random.random(100)

    def compute_embeddings(self, num_channel=-1):
        embeddings = []
        channels = []
        for i, (ch, messages) in tqdm.tqdm(enumerate(
                self.get_messages_from_channel()), total=16500):
            if i == num_channel:
                break
            messages = self._preprocess_messages(messages)
            if len(messages) < MIN_MESSAGE_NUM:
                continue
            emb = self.channel2vec(messages)
            if isinstance(emb, np.ndarray):
                channels.append(ch)
                embeddings.append(emb)

        return np.array(channels), np.array(embeddings)

    def get_messages_from_channel(self):
        """ generator of channel and its raw messages"""
        for mes_data in self.messages_data:
            if mes_data.channel_id == self.cur_ch:
                self.cur_messages.append(mes_data.message)
            else:
                channel = self.cur_ch
                messages = self.cur_messages.copy()
                self._get_cur(mes_data)
                yield channel, messages

        yield self.cur_ch, self.cur_messages


if __name__ == "__main__":
    path = "messages2vec/data/sorted_messages.csv"
    mgen = messages_generator(path, 11000)
    embr = BaseEmbedder(mgen)

    print(embr.compute_embeddings(10))
