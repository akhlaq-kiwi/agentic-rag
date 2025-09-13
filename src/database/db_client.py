from src.logger import get_logger
from typing import Dict, List, Any
from .base.base_db import BaseDB

logger = get_logger("DB Client", "logs/db.log")

class DBClient:
    def __init__(self, db: BaseDB):
        self.db = db

    def insert(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Inserting {len(chunks)} chunks with {self.db.__class__.__name__}")
        try:
            self.db.insert(chunks)
            logger.info(f"Inserted {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.exception(f"Failed to insert {chunks}")
            raise

    def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5, alpha: float = 0.5):
        logger.info(f"Searching for {query} with {self.db.__class__.__name__}")
        try:
            results = self.db.hybrid_search(query, query_embedding, top_k, alpha)
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.exception(f"Failed to search for {query}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5):
        logger.info(f"Searching for {query_embedding} with {self.db.__class__.__name__}")
        try:
            results = self.db.search(query_embedding, top_k)
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.exception(f"Failed to search for {query_embedding}")
            raise
        