from core.config import load_config
from core.ranking_model import Ranker


def main1():
    PATH_TO_CONFIG = "config.yaml"
    config = load_config(PATH_TO_CONFIG)
    print(config)
    ranker = Ranker(config)
    
    username = "sliv_halyavy"
    top = ranker.get_closest_channels(username)
    if top is None:
        print("No embedding")
        exit(1)
    # df = ranker.meta[ranker.meta.link.isin(top)]
    print(top)


def main2():
    from sentence_transformers import SentenceTransformer
    sentences = ["This is an example sentence", 
                 "Each sentence is converted"]

    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    print("model loaded")
    embeddings = model.encode(sentences)
    print(embeddings.shape)


if __name__ == "__main__":
    main2()
