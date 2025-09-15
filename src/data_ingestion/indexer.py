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
            # Check if embedder returns enhanced chunks (hybrid) or just embeddings (legacy)
            embedding_result = self.embedder.embed(chunks)
            
            # Handle hybrid embedding adapters that return enhanced chunks
            if (isinstance(embedding_result, list) and len(embedding_result) > 0 and 
                isinstance(embedding_result[0], dict) and "embedding_type" in embedding_result[0]):
                
                logger.info(f"Processing hybrid embeddings for {len(embedding_result)} chunks")
                return embedding_result
            
            # Handle legacy embedding adapters that return just embedding vectors
            else:
                for i, chunk in enumerate(chunks):
                    chunk["embedding"] = embedding_result[i]
                logger.info(f"Indexed {len(chunks)} chunks with legacy embeddings")
                return chunks
                
        except Exception as e:
            logger.exception(f"Failed to index chunks: {e}")
            raise