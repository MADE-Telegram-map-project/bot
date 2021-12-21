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


@dataclass
class OneSimilarChannel:
    channel_id: int
    similarity: float


class SimilarChannels:
    def __init__(self, indexes, similarities, channel_id=None):
        assert len(indexes) == len(similarities), "indexes and similarities mist be same lenght"
        self.indexes = indexes
        self.similarities = similarities
        self.channel_id = channel_id

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, items):
        if isinstance(items, int):
            return OneSimilarChannel(self.indexes[items], self.similarities[items])
        elif isinstance(items, slice):
            return SimilarChannels(self.indexes[items], self.similarities[items], self.channel_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self[0]}, ...)"
