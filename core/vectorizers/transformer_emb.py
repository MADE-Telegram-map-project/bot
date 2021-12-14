from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from core.preprocessing import drop_links
from core.vectorizers.base import BaseEmbedder
from core.preprocessing import messages_generator

DEFAULT_MODEL_NAME = "cointegrated/LaBSE-en-ru"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_MESSAGE_NUM = 20  # # min number of messages in one channel
MIN_WORD_NUM = 7  # min number of word in one message
MAX_TOKEN_LENGHT = 64


class TransEmbedder(BaseEmbedder):
    def __init__(self, model_name=DEFAULT_MODEL_NAME, device=DEVICE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

    def _preprocess_messages(self, messages: List[str]):
        cleaned_messages = []
        for m in messages:
            cm = drop_links(m)
            # TODO clean message from emodji
            # TODO split to sentences

            if len(cm.split()) > MIN_WORD_NUM:
                cleaned_messages.append(cm)
        if len(cleaned_messages) < MIN_MESSAGE_NUM:
            return []

        tokens = self._tokenize_text(cleaned_messages)
        return tokens

    def _tokenize_text(self, text: List[str]):
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGHT,
            return_tensors='pt',
        )
        for k, v in encoded_input.items():
            encoded_input[k] = v.to(self.device)

        return encoded_input

    def channel2vec(self, messages):
        with torch.no_grad():
            model_output = self.model(**messages)
        mes_embs = model_output.pooler_output
        mes_embs = torch.nn.functional.normalize(mes_embs)
        mes_embs = mes_embs.cpu().detach().numpy()
        channel_emb = np.mean(mes_embs, axis=0)
        return channel_emb
    
    def description2vec(self, text: str):

        pass

if __name__ == "__main__":
    path = "messages2vec/data/sorted_messages.csv"

    transmb = TransEmbedder(messages_data=messages_generator(path))
    ch, embs = transmb.compute_embeddings(50)
    print(ch[:10])

    for x in embs[:10]:
        print(x[:5])
