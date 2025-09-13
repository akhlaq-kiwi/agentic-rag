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