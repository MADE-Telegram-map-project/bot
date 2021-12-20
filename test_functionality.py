import pytest
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from core.config import load_config
from core.ranking_model import Ranker
from core.read_write import messages_generator
from core.vectorizers.transformer_emb import TransEmbedder, DEFAULT_MODEL_NAME, NEW_MODEL_NAME

PATH_TO_CONFIG = "config.yaml"
config = load_config(PATH_TO_CONFIG)

DESCRIPTION = ["Носорог съел собянина"]


@pytest.fixture(scope="session")
def local_ranker() -> Ranker:
    ranker = Ranker(config)
    return ranker


def test_trans_emb():
    path = "data/sorted_messages.csv"
    transmb = TransEmbedder(model_name=NEW_MODEL_NAME,
                            messages_data=messages_generator(path))
    n = 10
    ch, embs = transmb.compute_embeddings(n)
    assert len(embs) <= n


def test_transembedder_wo_data_loading():
    transmb = TransEmbedder(
        messages_data=None, load_data=False, model_name=NEW_MODEL_NAME)
    demb = transmb.description2vec(DESCRIPTION)
    assert hasattr(demb, "shape"), f"demb is {demb}"


@pytest.mark.parametrize(
    "username",
    [
        pytest.param("sliv_halyavy"),
        pytest.param("https://t.me/latinapopacanski"),
        pytest.param("t.me/cutterpool"),
    ]
)
def test_get_closest_channel(local_ranker, username):
    top = local_ranker.get_channels_by_username(username)
    assert top is not None


@pytest.mark.parametrize(
    "descr",
    [
        pytest.param("Носорог съел собянина"),
        pytest.param("Трейдинг"),
        pytest.param("моя жизнь не так сильно переплетается с с/х. Так объясните, зачем замедлять поток воды?"),
    ]
)
def test_descr2vec_in_ranker(local_ranker, descr):
    sim = local_ranker.get_channels_by_description(descr)
    print(sim)
