from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDB(ABC):
    @abstractmethod
    def insert(self, records: List[Dict[str, Any]]):
        """Insert a list of records into the DB"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5):
        """Search top_k records by similarity"""
        pass

    @abstractmethod
    def hybrid_search(self, query: str, query_embedding, top_k: int = 5, alpha: float = 0.5):
        """Hybrid search: combine vector + keyword scores"""
        pass

    @abstractmethod
    def init_schema(self):
        """Initialize required tables/collections"""
        pass
