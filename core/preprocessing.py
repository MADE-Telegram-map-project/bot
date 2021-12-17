""" read and write messages """

import re
from typing import List, Optional
from functools import partial
from multiprocessing import Pool

import demoji
import nltk
from nltk import download
from nltk.corpus import stopwords as SW
from nltk.stem import SnowballStemmer

from  .read_write import MessageData, messages_generator

download("stopwords")
stop_words = set(SW.words('english')).union(SW.words('russian'))
stemmer_ru = SnowballStemmer("russian")
stemmer_en = SnowballStemmer("english")

LINK_REGEX = {
    "(http://|https://)?t\\.me/joinchat/[a-zA-Z0-9_-]{5,32}": " ",
    "(http://|https://)?t\\.me/[a-zA-Z0-9_]{5,32}": " ",
    "@[a-z0-9_A-Z]+": " ",
    r'<img[^<>]+(>|$)': " image_token ",
    r'<[^<>]+(>|$)': " ",
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+': " url_token "
}


def stem_word(token: str) -> str:
    """ apply english and russan stemmers to a token """
    token = stemmer_ru.stem(token)
    token = stemmer_en.stem(token)
    return token


def stem_messages(tokens: List[str]) -> List[str]:
    return [stem_word(w) for w in tokens]


def tokenize_message(doc: str, stemming=True) -> List[str]:
    tokens = doc.split()
    if stemming:
        tokens = stem_messages(tokens)
    tokens = [t for t in tokens if t not in stop_words and len(t)]
    return tokens


def drop_links(text: str) -> str:
    for link_regex, replace in LINK_REGEX.items():
        text = re.sub(link_regex, replace, text)  # drop joinchat links

    return text


def preprocess_message(text: str, is_lower: bool = True) -> str:
    """Tokenize, clean up input document string

    https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32
    """
    assert isinstance(text, str)

    if is_lower:
        text = text.lower()

    try:
        text = drop_links(text)
        text = re.sub("[^a-zа-я\\s-]", "", text, flags=re.IGNORECASE)  # drop everything except
        text = re.sub("\\s+", " ", text)
    except BaseException:
        pass
    return text.strip()


def process_one_message(
        md: MessageData, pfunc,
        tokenize=True, stemming=True
    ) -> MessageData:
    message = pfunc(md.message)
    if tokenize:
        message = tokenize_message(message, stemming=stemming)
    return MessageData(md.message_id, md.channel_id, message)


class HeaderTextExtractor:
    def __init__(self, entity_regex: Optional[str] = None):
        if entity_regex is None:
            self.regex = re.compile("|".join(list(LINK_REGEX.keys()) + ["|@[a-z0-9_]+"]), re.IGNORECASE | re.UNICODE)
        else:
            self.regex = re.compile(entity_regex, re.UNICODE)

    def __call__(self, text: str) -> str:
        main_lines = []

        for line in map(str.strip, text.splitlines()):
            if line:
                match = self.regex.match(line)
                # first line with mention
                if match is not None and not main_lines:
                    main_lines.append(line)
                elif match is None:
                    main_lines.append(line)

        return demoji.replace(drop_links(" ".join(main_lines)), "").strip()

def get_messages(
        path: str,
        max_num: int = -1,
        threads=4,
        chunksize=None,
        cleaning_func=preprocess_message,
        tokenize=True,
        stemming=True,
) -> List[MessageData]:
    """
    return cleaned messages

    params:
        - path - path to messages raw table
        - max_num - max number of messages to return (-1 means all)
        - cleaning_func - function, that process and clean message
        - tokenize - if True return tokens of messages, else just cleaned messages
    """
    messages_reader = messages_generator(path, max_num)
    with Pool(threads) as p:
        messages = p.map(
            partial(process_one_message, pfunc=cleaning_func,
            tokenize=tokenize, stemming=stemming),
            messages_reader,
            chunksize
        )
    return messages


def clear_emoji(message: str):
    return demoji.replace(message)


def split_sentences(message: str):
    sentences = []
    for sentence in nltk.sent_tokenize(message):
        sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    path = "data/raw_16k/Messages.csv"

    messages = get_messages(path, 1000, threads=12, tokenize=True, stemming=False)
    print(len(messages))
    print(messages[:10])
