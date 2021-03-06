from typing import List
import random

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

MIN_WORD_NUM = 7        # min number of word in one message
MIN_MESSAGE_NUM = 20    # min number of messages in one channel
MIN_SENTENCE_NUM = 50
MAX_SENTENCES_NUM = 500
MAX_TOKEN_LENGHT = 64
BATCH_SIZE = 128


class TransEmbedder(BaseEmbedder):
    def __init__(self, model_name=NEW_MODEL_NAME, device=DEVICE, *args, **kwargs):
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
            min_message_num=MIN_MESSAGE_NUM,
            min_sentence_num=MIN_SENTENCE_NUM,
            max_sentence_num=MAX_SENTENCES_NUM):
        assert isinstance(messages, list), "messages must be list of strings"
        sentences = []
        n_messages = 0
        for m in messages:
            sents = split_sentences(m)
            _n_sentences = 0
            for sentence in sents:
                if len(sentence.split(' ')) >= min_word_num:
                    sentences.append(sentence)
                    _n_sentences += 1

            if _n_sentences >= 1:
                n_messages += 1
        if n_messages < min_message_num:
            return []
        
        random.shuffle(sentences)
        cleaned_sentences = [] 
        for sent in sentences:
            if len(cleaned_sentences) > max_sentence_num:
                break
            cs = drop_links(sent)
            cs = clear_emoji(cs)
            cleaned_sentences.append(cs)
        
        if len(cleaned_sentences) < min_sentence_num:
            return []
        elif len(cleaned_sentences) > max_sentence_num:
            cleaned_sentences = random.sample(cleaned_sentences, k=max_sentence_num)
        
        if self.model_name == DEFAULT_MODEL_NAME:
            tokens = self._tokenize_text(cleaned_sentences)
        elif self.model_name == NEW_MODEL_NAME:
            tokens = cleaned_sentences
        else:
            raise ValueError("No such model")
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

    def description2vec(self, description: List[str], max_sentence_num=20, min_word_num=1):
        assert isinstance(description, list), "description must be list of strings"
        sentences = self._preprocess_messages(
            description,
            min_word_num=min_word_num,
            min_message_num=1,
            min_sentence_num=1,
            max_sentence_num=max_sentence_num,
        )
        if len(sentences) == 0:
            channel_emb = None
        elif len(sentences) == 1 and len(sentences[0].strip()) == 0:
            channel_emb = None
        else:
            channel_emb = self.channel2vec(sentences)
        return channel_emb


if __name__ == "__main__":
    path = "data/sorted_messages.csv"

    transmb = TransEmbedder(messages_data=messages_generator(path), 
                            model_name=NEW_MODEL_NAME)
    ch, embs = transmb.compute_embeddings(10)
    print(ch[:10])

    for x in embs[:10]:
        print(x[:5])

    demb = transmb.description2vec("?????????????? ???????? ????????????????")
    print(demb.shape)
