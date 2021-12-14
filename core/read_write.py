import csv
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
import scipy.sparse

MessageData = namedtuple("MessageData", ["message_id", "channel_id", "message"])
dtypes = {
    "message_id": np.int64,
    "channel_id": np.int64,
    "views": np.int32,
    "forwards": np.int32,
    "replies_cnt": np.int32,
    "fwd_from_channel_id": "Int64",
    "fwd_from_message_id": "Int64",
}


def read_messages_full(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtypes)
    return df


def read_messages_metadata(path: str) -> pd.DataFrame:
    usecols = list(dtypes.keys()) + ["date"]
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    return df


def messages_generator(path: str, max_num=-1):
    """ read messages and return with indexes in MessageData object 
    sorted by channel id
    """
    with open(path) as fin:
        reader = csv.reader(fin, delimiter=',')
        header = next(reader)
        for i in range(len(header)):
            if header[i] == "message":
                mes_idx = i
            elif header[i] == "message_id":
                mid_idx = i
            elif header[i] == "channel_id":
                cid_idx = i

        for i, row in enumerate(reader):
            if i == max_num:
                break
            message_id, channel_id = int(row[mid_idx]), int(row[cid_idx])
            message = row[mes_idx]
            if message is None or message == "":
                continue
            yield MessageData(message_id, channel_id, message)


def read_messages_df(path: str, format="pickle"):
    """ read messages dataframe

    params:
        - format - pickle or csv
    """
    if format == "pickle":
        df = pd.read_pickle(path)
    elif format == "csv":
        df = pd.read_csv(path)
    return df


def write_messages_df(df: pd.DataFrame, path: str, format="pickle"):
    """ write messages dataframe

    params:
        - format - pickle or csv
    """
    if format == "pickle":
        df.to_pickle(path)
    elif format == "csv":
        df.to_csv(path, index=None)


def read_sparse(path):
    X = scipy.sparse.load_npz(path)
    return X


def write_sparse(X, path):
    scipy.sparse.save_npz(path, X)


def write_numpy_array(array: np.ndarray, path: str):
    assert isinstance(array, np.ndarray)
    with open(path, 'wb') as fout:
        np.save(fout, array)


def read_numpy_array(path: str, **kwargs):
    with open(path, 'rb') as fin:
        array = np.load(fin, **kwargs)
    return array


if __name__ == "__main__":
    path = "data/raw_16k/Messages.csv"

    messages = messages_generator(path, 10000)
    
    m = next(messages)
    print(m.channel_id, m.message_id)
    # cur_ch = m.channel_id
    dct = defaultdict(int)
    
    for m in messages:
        dct[m.channel_id] += 1
        # if m.channel_id != cur_ch:
            # print(m.channel_id, m.message_id)
            # cur_ch = m.channel_id
    print(dct)
