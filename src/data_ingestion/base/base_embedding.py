from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension"""
        pass
