from omegaconf import OmegaConf
from yaml import load

from core.entities import MainConfig


def load_config(path: str) -> MainConfig:
    base_config = OmegaConf.load(path)
    config = OmegaConf.merge(base_config, base_config)
    schema = OmegaConf.structured(MainConfig)
    config = OmegaConf.merge(schema, config)
    config: MainConfig = OmegaConf.to_object(config)
    return config
