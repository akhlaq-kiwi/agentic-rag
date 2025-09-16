#!/usr/bin/env python3
"""
LlamaIndex-based embedding script for processing markdown files with hybrid search support
Creates and populates a 'vectors' table in PostgreSQL
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_USER, 
    DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
)

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import TextNode
import requests
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaIndexEmbedder:
    """Process markdown files using LlamaIndex for hybrid search compatible embeddings"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/"
        self.embedding_model = "nomic-embed-text:v1.5"
        self.chunk_size = 500  # Tokens per chunk
        self.chunk_overlap = 100  # Token overlap for context continuity
        
        # Initialize components
        self.embed_model = None
        self.vector_store = None
        self.storage_context = None
        self.node_parser = None
        
    def setup_database(self):
        """Create vectors table with proper structure for hybrid search"""
        try:
            logger.info("🔧 Setting up vectors table...")
            
            conn = psycopg2.connect(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD
            )
            
            with conn.cursor() as cur:
                # Create pgvector extension if not exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Drop existing table to ensure clean schema
                cur.execute(f"DROP TABLE IF EXISTS data_vectors CASCADE;")
                
                # Create vectors table with dense and sparse vector support
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS data_vectors (
                        id BIGSERIAL PRIMARY KEY,
                        text TEXT,
                        metadata_ JSONB,
                        node_id VARCHAR,
                        dense_embedding vector({DIM}),
                        sparse_embedding JSONB,
                        embedding vector({DIM})
                    );
                """)
                
                # Create indexes for hybrid search
                # HNSW index for dense vector similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_dense_embedding 
                    ON data_vectors USING hnsw (dense_embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                
                # HNSW index for backward compatibility
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_embedding 
                    ON data_vectors USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                
                # GIN index for sparse embeddings
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_sparse_embedding 
                    ON data_vectors USING gin(sparse_embedding);
                """)
                
                # GIN index for metadata filtering
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_metadata 
                    ON data_vectors USING gin(metadata_);
                """)
                
                # Create index on node_id for faster lookups
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_node_id 
                    ON data_vectors(node_id);
                """)
                
                # Add full-text search column and index
                cur.execute("""
                    ALTER TABLE data_vectors ADD COLUMN IF NOT EXISTS text_search_tsv tsvector 
                    GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, ''))) STORED;
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_vectors_text_search_tsv 
                    ON data_vectors USING gin(text_search_tsv);
                """)
                
                conn.commit()
                logger.info("✅ Vectors table and indexes created successfully")
                
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False

    def get_context_with_gemma(self, text: str) -> str:
        """Use Ollama Gemma-2b to generate context summary for a chunk."""
        try:
            gemma = Ollama(model="gemma:2b", base_url=self.ollama_url, request_timeout=60.0)
            prompt = (
                "Given the following text, generate a short contextual summary "
                "that captures its key theme in 1 sentences of max 5 words, factual and concise.\n\n"
                f"Text:\n{text}\n\n"
                "Summary:"
            )
            response = gemma.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"⚠️ Gemma context generation failed: {e}")
            return ""

    def warm_up_ollama(self):
        """Pre-warm Ollama embedding model"""
        try:
            logger.info("🔥 Warming up Ollama embedding model...")
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": "test"},
                timeout=30
            )
            if response.status_code == 200:
                logger.info("✅ Ollama model warmed up successfully")
                return True
            else:
                logger.warning(f"⚠️ Ollama warm-up returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not warm up Ollama: {e}")
            return False
    
    def get_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding for text using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embedding', [])
            else:
                logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating dense embedding: {e}")
            return []
    
    def get_sparse_embedding(self, text: str, vocab_size: int = 30000) -> Dict[str, float]:
        """
        Generate sparse embedding using TF-IDF and keyword extraction
        Returns a dictionary with token indices and their weights
        """
        try:
            # Clean and tokenize text
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            tokens = text.split()
            
            # Calculate term frequencies
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            # Create sparse representation with TF scores
            sparse_vector = {}
            
            for token, count in token_counts.items():
                # Simple TF score (can be enhanced with IDF)
                tf_score = count / total_tokens
                
                # Hash token to create consistent index
                token_hash = hash(token) % vocab_size
                
                # Store with both token and hash for debugging
                sparse_vector[str(token_hash)] = tf_score
            
            # Add important keywords with higher weights
            important_patterns = [
                r'\b(policy|procedure|guideline|rule|regulation)\b',
                r'\b(employee|staff|worker|personnel)\b',
                r'\b(benefit|compensation|salary|wage)\b',
                r'\b(vacation|leave|time off|holiday)\b',
                r'\b(performance|evaluation|review|assessment)\b',
                r'\b(safety|security|compliance|legal)\b'
            ]
            
            for pattern in important_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    for match in matches:
                        token_hash = hash(match) % vocab_size
                        current_weight = sparse_vector.get(str(token_hash), 0.0)
                        sparse_vector[str(token_hash)] = max(current_weight, 0.8)  # Boost important terms
            
            return sparse_vector
            
        except Exception as e:
            logger.error(f"Error generating sparse embedding: {e}")
            return {}
    
    def initialize_components(self):
        """Initialize LlamaIndex components for embedding and storage"""
        try:
            logger.info("🔧 Initializing LlamaIndex components...")
            
            # Initialize Ollama embedding model
            self.embed_model = OllamaEmbedding(
                model_name=self.embedding_model,
                base_url=self.ollama_url,
                request_timeout=120.0
            )
            logger.info("✅ Ollama embedding model initialized")
            
            # Initialize PostgreSQL vector store pointing to 'vectors' table
            self.vector_store = PGVectorStore.from_params(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD,
                table_name="vectors",  # Using 'vectors' table as requested
                embed_dim=DIM,
                hybrid_search=True,  # Enable hybrid search
                text_search_config="english",  # Full-text search configuration
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                }
            )
            logger.info("✅ PostgreSQL vector store initialized with hybrid search on 'vectors' table")
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Initialize node parser with sentence-aware splitting
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[.!?]"
            )
            logger.info("✅ Node parser initialized with sentence splitting")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def process_markdown_file(self, file_path: Path) -> int:
        """Process a single markdown file using LlamaIndex"""
        logger.info(f"📄 Processing: {file_path.name}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"⚠️ Empty file: {file_path.name}")
                return 0
            
            # Create LlamaIndex Document with metadata
            document = Document(
                text=content,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "file_size": len(content),
                    "processed_at": datetime.now().isoformat(),
                    "file_type": "markdown",
                    #"context": self.get_context_with_gemma(content)
                }
            )
            
            # Parse document into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([document])
            logger.info(f"📋 Created {len(nodes)} nodes from {file_path.name}")
            
            # Add additional metadata to each node
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(nodes),
                    "char_count": len(node.text),
                    "chunk_id": f"{file_path.name}_{i}"
                })
            
            # Generate both dense and sparse embeddings for each node
            logger.info("🔄 Generating dense and sparse embeddings...")
            
            # Process nodes with custom embedding logic
            conn = psycopg2.connect(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD
            )
            
            try:
                with conn.cursor() as cur:
                    for i, node in enumerate(nodes):
                        logger.info(f"  🔄 Processing node {i+1}/{len(nodes)}")
                        
                        # Generate dense embedding using Ollama
                        dense_embedding = self.get_dense_embedding(node.text)
                        if not dense_embedding:
                            logger.error(f"Failed to generate dense embedding for node {i}")
                            continue
                        
                        # Generate sparse embedding using TF-IDF
                        sparse_embedding = self.get_sparse_embedding(node.text)
                        
                        # Insert into database with both embeddings
                        cur.execute("""
                            INSERT INTO data_vectors (
                                text, metadata_, node_id, 
                                dense_embedding, sparse_embedding, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            node.text,
                            json.dumps(node.metadata),
                            node.node_id,
                            dense_embedding,  # Dense vector
                            json.dumps(sparse_embedding),  # Sparse vector as JSON
                            dense_embedding  # Backward compatibility
                        ))
                
                conn.commit()
                logger.info(f"✅ Successfully stored {len(nodes)} nodes with dense and sparse embeddings")
                
            except Exception as e:
                logger.error(f"❌ Error storing embeddings: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
            
            logger.info(f"✅ Successfully processed {len(nodes)} nodes from {file_path.name}")
            return len(nodes)
            
        except Exception as e:
            logger.error(f"❌ Error processing file {file_path.name}: {e}")
            return 0
    
    def process_all_markdown_files(self, processed_dir: str = "data/processed/markdown"):
        """Process all markdown files in the processed directory"""
        processed_path = Path(processed_dir)
        
        if not processed_path.exists():
            logger.error(f"❌ Processed directory not found: {processed_path}")
            return
        
        # Find all markdown files
        md_files = list(processed_path.glob("*.md"))
        
        if not md_files:
            logger.warning(f"⚠️ No markdown files found in {processed_path}")
            return
        
        logger.info(f"🔍 Found {len(md_files)} markdown files to process")
        
        # Setup database table
        if not self.setup_database():
            logger.error("❌ Failed to setup database")
            return
        
        # Initialize LlamaIndex components
        if not self.initialize_components():
            logger.error("❌ Failed to initialize LlamaIndex components")
            return
        
        # Warm up Ollama
        self.warm_up_ollama()
        
        # Process each file
        total_nodes = 0
        start_time = time.time()
        
        for md_file in md_files:
            nodes_processed = self.process_markdown_file(md_file)
            total_nodes += nodes_processed
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"📊 Total nodes processed: {total_nodes}")
        logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        if total_nodes > 0:
            logger.info(f"📈 Average time per node: {elapsed_time/total_nodes:.2f} seconds")
        
        # Verify database content
        self.verify_database_content()
    
    def verify_database_content(self):
        """Verify the vectors table has been populated correctly"""
        try:
            conn = psycopg2.connect(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD
            )
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count total vectors
                cur.execute("SELECT COUNT(*) as total FROM data_vectors;")
                total = cur.fetchone()['total']
                
                # Count vectors with dense embeddings
                cur.execute("SELECT COUNT(*) as with_dense FROM data_vectors WHERE dense_embedding IS NOT NULL;")
                with_dense = cur.fetchone()['with_dense']
                
                # Count vectors with sparse embeddings
                cur.execute("SELECT COUNT(*) as with_sparse FROM data_vectors WHERE sparse_embedding IS NOT NULL;")
                with_sparse = cur.fetchone()['with_sparse']
                
                # Count vectors with embeddings (backward compatibility)
                cur.execute("SELECT COUNT(*) as with_embeddings FROM data_vectors WHERE embedding IS NOT NULL;")
                with_embeddings = cur.fetchone()['with_embeddings']
                
                # Count vectors with FTS (check if column exists first)
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'data_vectors' AND column_name = 'text_search_tsv';
                """)
                has_fts_column = cur.fetchone() is not None
                
                if has_fts_column:
                    cur.execute("SELECT COUNT(*) as with_fts FROM data_vectors WHERE text_search_tsv IS NOT NULL;")
                    with_fts = cur.fetchone()['with_fts']
                else:
                    with_fts = 0
                
                # Get sample vector
                cur.execute("SELECT id, substring(text, 1, 100) as text_preview, metadata_ FROM data_vectors LIMIT 1;")
                sample = cur.fetchone()
                
                logger.info("📊 Vectors Table Verification:")
                logger.info(f"  📄 Total vectors: {total}")
                logger.info(f"  🟢 With dense embeddings: {with_dense}")
                logger.info(f"  🔵 With sparse embeddings: {with_sparse}")
                logger.info(f"  🔢 With embeddings (legacy): {with_embeddings}")
                logger.info(f"  🔍 With full-text search: {with_fts}")
                
                if sample:
                    logger.info(f"  📝 Sample vector ID {sample['id']}: '{sample['text_preview']}...'")
                    logger.info(f"  📋 Sample metadata: {sample['metadata_']}")
                
                # Check hybrid search readiness
                if with_dense > 0 and with_sparse > 0:
                    logger.info("  ✅ Vectors table is ready for enhanced hybrid search (dense + sparse + text)!")
                elif with_embeddings > 0:
                    logger.info("  ✅ Vectors table is ready for basic hybrid search!")
                else:
                    logger.warning("  ⚠️ Vectors table may not be fully ready for hybrid search")
                
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")

def main():
    """Main function"""
    logger.info("🚀 Starting LlamaIndex document embedding process...")
    logger.info("📦 Target table: 'data_vectors' (hybrid search compatible)")
    
    embedder = LlamaIndexEmbedder()
    embedder.process_all_markdown_files()
    
    logger.info("✅ Document embedding process completed!")

if __name__ == "__main__":
    main()