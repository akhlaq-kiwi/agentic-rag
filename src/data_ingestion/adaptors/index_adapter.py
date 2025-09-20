from typing import List, Union, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import re

from src.logger import get_logger
from ..base.base_indexer import BaseIndexer
from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_USER, 
    DATABASE_PASSWORD, DATABASE_NAME, OLLAMA_BASE_URL, DIM
)

# LlamaIndex imports
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

logger = get_logger("LlamaIndex Adapter", "logs/ingestion.log")


class LlamaIndexAdapter(BaseIndexer):
    """
    Adapter that implements BaseIndexer interface using LlamaIndex with PostgreSQL+pgvector.
    Contains all indexing logic internally.
    """
    
    # Regex for parsing page metadata from Docling markdown
    META_RE = re.compile(
        r"<!--\s*Metadata\(Page No\.\s*:\s*(\d+),\s*Filename:\s*(.+?)\)\s*-->",
        re.IGNORECASE
    )
    
    def __init__(
        self,
        mode: str = "page",
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        min_chars: int = 50,
        embed_model: str = "nomic-embed-text:v1.5",
        table_name: str = "vectors",
        collection_name: str = "default_collection",
        context_llm: str = "gemma3:4b",
        enable_context_enrichment: bool = True,
        context_timeout: int = 10,  # Timeout in seconds for context generation
        **kwargs
    ):
        """
        Initialize the LlamaIndex adapter.
        
        Args:
            mode: Chunking mode ('page', 'paragraph', 'sentences')
            chunk_size: Size of chunks for sentence splitting
            chunk_overlap: Overlap between chunks
            min_chars: Minimum characters per chunk
            embed_model: Ollama embedding model name
            table_name: PostgreSQL table name for vectors
            collection_name: Collection name for organizing documents
            context_llm: Ollama context model name
            enable_context_enrichment: Whether to use LLM for context enrichment
            context_timeout: Timeout in seconds for LLM context generation
            **kwargs: Additional configuration options
        """
        self.mode = mode
        self.embed_model = embed_model
        self.table_name = table_name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars
        self.enable_context_enrichment = enable_context_enrichment
        self.context_timeout = context_timeout
        
        # Initialize sentence splitter for sentence mode
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Set up LlamaIndex embedding model
        Settings.embed_model = OllamaEmbedding(
            model_name=embed_model, 
            base_url=OLLAMA_BASE_URL
        )

        # Initialize LLM with timeout settings
        Settings.llm = Ollama(
            model=context_llm,
            base_url=OLLAMA_BASE_URL,
            request_timeout=300  # Short timeout for initialization test
        )
        
        # Initialize PostgreSQL vector store
        self.vector_store = PGVectorStore.from_params(
            database=DATABASE_NAME,
            host=DATABASE_HOST,
            password=DATABASE_PASSWORD,
            port=DATABASE_PORT,
            user=DATABASE_USER,
            schema_name="public",
            table_name=table_name,
            embed_dim=DIM,
            hybrid_search=False
        )
        
        logger.info(f"LlamaIndexAdapter initialized with mode={mode}, embed_model={embed_model}")
        
        # Test LLM connection if context enrichment is enabled
        if self.enable_context_enrichment:
            try:
                logger.info("Testing LLM connection...")
                # Use a very simple prompt with shorter timeout
                test_response = Settings.llm.complete("Hi")
                if test_response and test_response.text:
                    logger.info("LLM connection test successful")
                else:
                    raise ValueError("Empty response from LLM")
            except Exception as e:
                logger.warning(f"LLM connection test failed: {e}")
                logger.info("Context enrichment will be disabled due to LLM connection issues")
                self.enable_context_enrichment = False
        
        # If context enrichment is still enabled after test, reconfigure LLM with proper timeout
        if self.enable_context_enrichment:
            Settings.llm = Ollama(
                model=context_llm,
                base_url=OLLAMA_BASE_URL,
                request_timeout=self.context_timeout
            )
    
    def _split_pages(self, text: str, fallback_file_name: str) -> List[Tuple[int, str, str]]:
        """Split markdown text into pages based on metadata markers."""
        matches = list(self.META_RE.finditer(text))
        if not matches:
            # Whole file is page 1 if no markers
            return [(1, fallback_file_name, text.strip())]
        
        out: List[Tuple[int, str, str]] = []
        for i, m in enumerate(matches):
            page_no = int(m.group(1))
            file_name = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                out.append((page_no, file_name, body))
        return out
    
    def _create_nodes(self, md_path: Union[str, Path], doc_id: Optional[str] = None) -> List[TextNode]:
        """Create LlamaIndex TextNodes from a page-marked Markdown file."""
        md_path = Path(md_path)
        file_stem = md_path.stem
        text = md_path.read_text(encoding="utf-8")

        # Parse page blocks
        pages = self._split_pages(text, fallback_file_name=md_path.name)

        nodes: List[TextNode] = []
        for page_no, file_name, page_text in pages:
            chunks: List[str]
            if self.mode == "page":
                chunks = [page_text.strip()]
            elif self.mode == "paragraph":
                chunks = [p.strip() for p in re.split(r"\n{2,}", page_text) if p.strip()]
            else:  # sentences
                # Feed one Document per page to SentenceSplitter, then collect node texts
                doc = Document(text=page_text)
                sentence_nodes = self.splitter.get_nodes_from_documents([doc])
                chunks = [n.get_content().strip() for n in sentence_nodes if n.get_content().strip()]

            for i, content in enumerate(chunks, start=1):
                if len(content) < self.min_chars:
                    continue
                
                # Base metadata
                meta = {"file_name": file_name, "page_no": page_no}
                
                # Conditionally enrich content and metadata with LLM context
                if self.enable_context_enrichment:
                    enriched_content, enriched_meta = self._enrich_with_context(content, meta)
                else:
                    enriched_content, enriched_meta = content, meta
                
                # Construct TextNode with (possibly enriched) content and metadata
                node = TextNode(
                    text=enriched_content, 
                    metadata=enriched_meta,
                    id_=f"{doc_id or file_stem}_{page_no}_{i}"  # Create unique node ID
                )
                nodes.append(node)

        return nodes
    
    def _enrich_with_context(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Use LLM to add context to text content and metadata."""
        try:
            # Truncate content for faster processing
            truncated_content = content[:300] if len(content) > 300 else content
            
            # Create a more concise context prompt
            context_prompt = f"""Summarize key topics in 1 sentence:

            Text: {truncated_content}
            File: {metadata.get('file_name', 'Unknown')}

            Summary:"""
            
            # Get context from LLM with timeout handling
            response = Settings.llm.complete(context_prompt)
            context_summary = response.text.strip()
            
            # Validate response
            if not context_summary or len(context_summary) < 10:
                raise ValueError("Invalid or empty context summary")
            
            # Enrich the content with context (more concise format)
            enriched_content = f"[Context: {context_summary}]\n\n{content}"
            
            # Enrich metadata with additional context
            enriched_metadata = metadata.copy()
            enriched_metadata.update({
                "context_summary": context_summary,
                "content_length": len(content),
                "has_context_enrichment": True
            })
            
            logger.debug(f"Added context to chunk from {metadata.get('file_name', 'Unknown')}, page {metadata.get('page_no', 'Unknown')}")
            
            return enriched_content, enriched_metadata
            
        except Exception as e:
            logger.warning(f"Failed to enrich content with context: {e}")
            # Return original content with basic metadata enhancement
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                "content_length": len(content),
                "has_context_enrichment": False,
                "enrichment_failed": True
            })
            return content, enhanced_metadata
    
    def _upsert_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Embed TextNodes and upsert into PostgreSQL vector store."""
        logger.info(f"Creating embeddings for {len(nodes)} nodes...")
        
        if self.enable_context_enrichment:
            enriched_count = sum(1 for node in nodes if node.metadata.get("has_context_enrichment", False))
            logger.info(f"Context enrichment enabled: {enriched_count}/{len(nodes)} nodes enriched")
        
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        # Build index from nodes → triggers embedding + upsert
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        logger.info("Embeddings created and saved to PostgreSQL successfully!")
        return index
    
    def create_index(self, sources: List[Union[str, Path]], **kwargs) -> Any:
        """
        Create an index from processed document sources.
        
        Args:
            sources: List of paths to processed markdown documents
            **kwargs: Additional configuration options
            
        Returns:
            VectorStoreIndex object
        """
        try:
            # Validate and filter sources
            validated_sources = self.filter_supported_sources(sources)
            
            if not validated_sources:
                raise ValueError("No valid sources found for indexing")
            
            logger.info(f"Creating index from {len(validated_sources)} sources")
            
            # Process all sources and create nodes
            all_nodes = []
            for source in validated_sources:
                logger.info(f"Processing {source}")
                nodes = self._create_nodes(source)
                all_nodes.extend(nodes)
                logger.info(f"Created {len(nodes)} nodes from {source}")
            
            logger.info(f"Total nodes created: {len(all_nodes)}")
            
            # Create embeddings and save to PostgreSQL vector store
            index = self._upsert_nodes(all_nodes)
            
            logger.info("Index creation completed successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = {
                "table_name": self.table_name,
                "collection_name": self.collection_name,
                "embed_model": self.embed_model,
                "mode": self.mode,
                "status": "active",
                "database": DATABASE_NAME,
                "host": DATABASE_HOST
            }
            
            # You could add actual database queries here to get real statistics
            # e.g., document count, vector count, etc.
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats for this indexer.
        
        Returns:
            List of supported file extensions
        """
        return ['.md']  # Currently only supports markdown with page metadata