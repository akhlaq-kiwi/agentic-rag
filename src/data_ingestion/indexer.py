# li_chunk_index.py
from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from src.data_ingestion.base.base_indexer import BaseIndexer


class DocumentIndexer:
    """
    Create LlamaIndex TextNodes from a page-marked Markdown file.
    Keeps ONLY metadata {file_name, page_no}.
    Granularity:
      - mode="page":      1 chunk per page (default)
      - mode="paragraph": split each page by blank lines
      - mode="sentences": sentence windows via SentenceSplitter
    """
    def __init__(self, indexer: BaseIndexer):
        self.indexer = indexer

    def index(self, sources: List[Union[str, Path]], **kwargs) -> Any:
        index = self.indexer.create_index(sources, **kwargs)
        print(index)
        return index

class PgVectorIndexer:
    """
    Embed TextNodes with a local Ollama embedding model and upsert into Postgres+pgvector.
    """
    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        embed_model: str = "nomic-embed-text",
        pg_params: Optional[Dict[str, str]] = None,
        schema_name: str = "public",
        table_name: str = "li_vectors",
        collection_name: str = "default_collection",
    ):
        ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(model_name=embed_model, base_url=ollama_base_url)

        self.vector_store = PGVectorStore.from_params(
            database=(pg_params or {}).get("database", os.getenv("PGDATABASE", "rag")),
            host=(pg_params or {}).get("host", os.getenv("PGHOST", "localhost")),
            password=(pg_params or {}).get("password", os.getenv("PGPASSWORD", "postgres")),
            port=int((pg_params or {}).get("port", os.getenv("PGPORT", "5432"))),
            user=(pg_params or {}).get("user", os.getenv("PGUSER", "postgres")),
            schema_name=schema_name,
            table_name=table_name,
            collection_name=collection_name,
            # embed_dim inferred automatically from embedding model
        )

    def upsert_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        # Build index from nodes → triggers embedding + upsert
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        return index
