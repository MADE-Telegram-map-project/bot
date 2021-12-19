from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DataPath:
    dir: str
    ch_full: str
    emb_full: str
    channels: str


@dataclass
class BotConfig:
    token: str


@dataclass
class MainConfig:
    data: DataPath = MISSING
    bot: BotConfig = MISSING
