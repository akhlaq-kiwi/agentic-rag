from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        pass
