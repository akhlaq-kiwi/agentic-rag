from src.logger import get_logger
from .base.base_extractor import BaseExtractor
from typing import Union, Dict, List
from pathlib import Path

logger = get_logger("Document Ingestor", "logs/ingestion.log")

class DocumentIngestor:
    def __init__(self, extractor: BaseExtractor):
        self.extractor = extractor

    def ingest(self, source: Union[str, Path], export_format: str = "text") -> Union[str, Dict]:
        logger.info(f"Ingesting {source} with {self.extractor.__class__.__name__}")
        try:
            return self.extractor.extract(source, export_format)
        except Exception as e:
            logger.exception(f"Failed to ingest {source}")
            raise

    def batch_ingest(self, sources: List[Union[str, Path]], export_format: str = "text") -> List[Union[str, Dict]]:
        logger.info(f"Batch ingesting {len(sources)} documents")
        return [self.ingest(src, export_format) for src in sources]