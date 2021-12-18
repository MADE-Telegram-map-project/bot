from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer

from core.preprocessing import drop_links, clear_emoji, split_sentences
from core.vectorizers.base import BaseEmbedder
from core.preprocessing import messages_generator

DEFAULT_MODEL_NAME = "cointegrated/LaBSE-en-ru"
NEW_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_MESSAGE_NUM = 20  # min number of messages in one channel
MIN_WORD_NUM = 7  # min number of word in one message
MAX_TOKEN_LENGHT = 64
BATCH_SIZE = 64


class TransEmbedder(BaseEmbedder):
    def __init__(self, model_name=DEFAULT_MODEL_NAME, device=DEVICE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model_name = model_name
        if model_name == DEFAULT_MODEL_NAME:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(device)
        elif model_name == NEW_MODEL_NAME:
            self.model = SentenceTransformer(model_name)

    def _preprocess_messages(
            self, messages: List[str],
            min_word_num=MIN_WORD_NUM,
            min_message_num=MIN_MESSAGE_NUM) -> BatchEncoding:
        cleaned_messages = []
        
        for m in messages:
            cm = drop_links(m)
            cm = clear_emoji(cm)
            sents = split_sentences(cm)

            for sentence in sents:
                if len(sentence.split(' ')) > min_word_num:
                    cleaned_messages.append(sentence)

        if len(cleaned_messages) < min_message_num:
            return []
        
        if self.model_name == DEFAULT_MODEL_NAME:
            tokens = self._tokenize_text(cleaned_messages)
        elif self.model_name == NEW_MODEL_NAME:
            tokens = cleaned_messages
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

    def channel2vec(self, tokens: BatchEncoding) -> np.ndarray:
        embs = self.apply_model(tokens)
        channel_emb = np.mean(embs, axis=0)
        return channel_emb

    def apply_model(self, sentences: BatchEncoding) -> np.ndarray:
        if self.model_name == DEFAULT_MODEL_NAME:
            with torch.no_grad():
                model_output = self.model(**sentences)
            embs = model_output.pooler_output
            embs = torch.nn.functional.normalize(embs)
            embs = embs.cpu().detach().numpy()
        elif self.model_name == NEW_MODEL_NAME:
            embs = self.model.encode(sentences, batch_size=BATCH_SIZE)

        return embs

    def description2vec(self, text: str):
        description = [text]
        tokens = self._preprocess_messages(description, 1, 1)
        channel_emb = self.channel2vec(tokens)
        return channel_emb


if __name__ == "__main__":
    path = "data/sorted_messages.csv"

    transmb = TransEmbedder(messages_data=messages_generator(path))
    ch, embs = transmb.compute_embeddings(10)
    print(ch[:10])

    for x in embs[:10]:
        print(x[:5])

    demb = transmb.description2vec("Носорог съел собянина")
    print(demb.shape)
