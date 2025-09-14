from src.logger import get_logger
from typing import Dict, List, Any
from .base.base_embedding import BaseEmbedding

logger = get_logger("Document Indexer", "logs/ingestion.log")

class DocumentIndexer:
    def __init__(self, embedder: BaseEmbedding):
        self.embedder = embedder

    def index(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Indexing {len(chunks)} chunks with {self.embedder.__class__.__name__}")
        try:
            
            embedding = self.embedder.embed(chunks)
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embedding[i]
            logger.info(f"Indexed {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(e)
            logger.exception(f"Failed to index")
            raise