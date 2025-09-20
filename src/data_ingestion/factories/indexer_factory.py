from src.logger import get_logger
from ..base.base_indexer import BaseIndexer
from ..adaptors.index_adapter import LlamaIndexAdapter

logger = get_logger("Indexer Factory", "logs/ingestion.log")


class IndexerFactory:
    """Factory class for creating indexer instances."""
    
    @staticmethod
    def get_indexer(name: str, **kwargs) -> BaseIndexer:
        """
        Create an indexer instance based on the specified name.
        
        Args:
            name: Name of the indexer to create ('llamaindex', 'pgvector', etc.)
            **kwargs: Additional configuration parameters for the indexer
            
        Returns:
            BaseIndexer instance
            
        Raises:
            ValueError: If the indexer name is not recognized
        """
        name = name.lower()
        logger.debug(f"Initializing indexer: {name}")
        
        if name in ["llamaindex"]:
            return LlamaIndexAdapter(**kwargs)
        else:
            logger.error(f"Unknown indexer: {name}")
            raise ValueError(f"Unknown indexer: {name}")