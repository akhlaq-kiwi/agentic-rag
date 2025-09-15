import os, json
from src.logger import get_logger
from typing import Union, Dict, List, Any, Callable, Optional
from pathlib import Path
from .base.base_chunker import BaseChunker

logger = get_logger("Document Chunker", "logs/ingestion.log")

class DocumentChuncker:
    def __init__(self, chunker: BaseChunker):
        self.chunker = chunker

    def chunk(self, sources: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Legacy batch chunking method"""
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

    def chunk_and_process_streaming(
        self, 
        sources: List[Union[str, Path]], 
        embedder,
        db_client,
        chunk_callback: Optional[Callable] = None
    ) -> int:
        """
        Stream processing: chunk -> embed -> save immediately for each chunk
        
        Args:
            sources: List of source files to process
            embedder: Embedding adapter (should be hybrid or compatible)
            db_client: Database client for saving chunks
            chunk_callback: Optional callback function called for each processed chunk
            
        Returns:
            Total number of chunks processed
        """
        logger.info(f"Starting streaming chunk processing for {len(sources)} sources")
        total_chunks = 0
        
        try:
            for i, source in enumerate(sources, 1):
                logger.info(f"Processing source {i}/{len(sources)}: {source}")
                
                # Read and chunk the document
                md_text = Path(source).read_text()
                md_chunks = self.chunker.split(md_text, {"source": str(source)})
                
                logger.info(f"Generated {len(md_chunks)} chunks from {source}")
                
                # Process each chunk individually
                for j, chunk in enumerate(md_chunks, 1):
                    try:
                        logger.info(f"Processing chunk {j}/{len(md_chunks)} from {Path(source).name}")
                        
                        # Check if chunk already exists in database
                        if db_client.check_exists(chunk["text"], chunk.get("metadata", {})):
                            logger.info(f"⏭️  Chunk {j} already exists, skipping...")
                            if chunk_callback:
                                chunk_callback(None, j, len(md_chunks), source, skipped=True)
                            continue
                        
                        # Embed the chunk (hybrid or regular)
                        if hasattr(embedder, 'embed') and callable(embedder.embed):
                            embedded_chunks = embedder.embed([chunk])
                            
                            # Handle both hybrid and legacy embedding formats
                            if (isinstance(embedded_chunks, list) and len(embedded_chunks) > 0 and 
                                isinstance(embedded_chunks[0], dict) and "embedding_type" in embedded_chunks[0]):
                                # Hybrid embedding format
                                processed_chunk = embedded_chunks[0]
                            else:
                                # Legacy embedding format
                                chunk["embedding"] = embedded_chunks[0]
                                processed_chunk = chunk
                        else:
                            raise ValueError("Invalid embedder provided")
                        
                        # Save to database immediately
                        db_client.insert([processed_chunk])
                        total_chunks += 1
                        
                        logger.info(f"✓ Chunk {j} embedded and saved successfully")
                        
                        # Call optional callback
                        if chunk_callback:
                            chunk_callback(processed_chunk, j, len(md_chunks), source, skipped=False)
                            
                    except Exception as e:
                        logger.error(f"Failed to process chunk {j} from {source}: {e}")
                        # Continue with next chunk instead of failing completely
                        continue
                
                logger.info(f"✓ Completed processing {source}")
            
            logger.info(f"Streaming processing completed. Total chunks processed: {total_chunks}")
            return total_chunks
            
        except Exception as e:
            logger.exception(f"Failed during streaming chunk processing")
            raise