from core.entities.data import MainConfig
from core.config import load_config
from core.ranking_model import Ranker

if __name__ == "__main__":
    path_to_config = "config.test.yaml"
    config: MainConfig = load_config(path_to_config)
    ranker = Ranker(config)
