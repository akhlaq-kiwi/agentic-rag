import os, json
from src.logger import get_logger
from typing import Union, Dict, List, Any
from pathlib import Path
from .base.base_chunker import BaseChunker

logger = get_logger("Document Chunker", "logs/ingestion.log")

class DocumentChuncker:
    def __init__(self, chunker: BaseChunker):
        self.chunker = chunker

    def chunk(self, sources: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        logger.info(f"Chunking {sources} with {self.chunker.__class__.__name__}")
        try:
            
            chunks = []
            for source in sources:
                md_text = Path(source).read_text()
                md_chunks = self.chunker.split(md_text, {"source": str(source)})
                chunks.extend(md_chunks)
            logger.info(f"Chunked {sources} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.exception(f"Failed to chunk {sources}")
            raise