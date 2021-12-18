from core.config import load_config
from core.ranking_model import Ranker
from core.read_write import messages_generator

PATH_TO_CONFIG = "config.yaml"
config = load_config(PATH_TO_CONFIG)


def test_get_closest_channel():
    ranker = Ranker(config)

    username = "sliv_halyavy"
    top = ranker.get_closest_channels(username)
    if top is None:
        print("No embedding")
        exit(1)
    print(top)


def _test_check_new_trans():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    sentences = ["This is an example sentence",
                 "Each sentence is converted"]

    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    print("model loaded")
    embeddings = model.encode(sentences)
    print(embeddings.shape)


def _test_trans_emb():
    from core.vectorizers.transformer_emb import TransEmbedder, DEFAULT_MODEL_NAME, NEW_MODEL_NAME

    path = "data/sorted_messages.csv"
    transmb = TransEmbedder(model_name=DEFAULT_MODEL_NAME,
                            messages_data=messages_generator(path))
    ch, embs = transmb.compute_embeddings(10)
    print(ch[:10])

    for x in embs[:10]:
        print(x[:5])

    demb = transmb.description2vec("Носорог съел собянина")
    print(demb.shape)

    transmb = TransEmbedder(messages_data=None, load_data=False, model_name=NEW_MODEL_NAME)
    demb = transmb.description2vec("Носорог съел собянина")
    print(demb.shape)


def test_descr2vec_in_ranker():
    from core.ranking_model import Ranker

    r = Ranker(config)
    sim = r.get_channels_by_description("Носорог съел собянина")
    print(sim)


test_descr2vec_in_ranker()
