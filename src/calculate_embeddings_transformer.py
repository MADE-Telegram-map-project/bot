from core.read_write import write_numpy_array
from core.preprocessing import messages_generator
from core.vectorizers.transformer_emb import TransEmbedder


if __name__ == "__main__":
    path_to_messages = "messages2vec/data/sorted_messages.csv"
    emb_data = TransEmbedder(messages_data=messages_generator(path_to_messages))
    ch, embs = emb_data.compute_embeddings()

    path_to_channels = "messages2vec/data/channels_LaBSE.npy"
    path_to_emb = "messages2vec/data/embeddings_LaBSE.npy"
    write_numpy_array(ch, path_to_channels)
    write_numpy_array(embs, path_to_emb)
