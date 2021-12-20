import os

from core.read_write import write_numpy_array
from core.preprocessing import messages_generator
from core.vectorizers.transformer_emb import TransEmbedder, NEW_MODEL_NAME

EMB_NAME = "emb_MiniLM_v1"
NUMBER_OF_CHANNELS = -1  # -1 means all

if __name__ == "__main__":
    path_to_messages = "data/sorted_messages.csv"
    emb_data = TransEmbedder(
        model_name=NEW_MODEL_NAME,
        messages_data=messages_generator(path_to_messages),
    )
    ch, embs = emb_data.compute_embeddings(NUMBER_OF_CHANNELS)
    print(embs.shape)
    
    _dir = "core/data/{}".format(EMB_NAME)
    os.makedirs(_dir, exist_ok=True)

    path_to_channels = os.path.join(_dir, "channels.npy")
    path_to_emb = os.path.join(_dir, "embeddings.npy")

    write_numpy_array(ch, path_to_channels)
    write_numpy_array(embs, path_to_emb)
