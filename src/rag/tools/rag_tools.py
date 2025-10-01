import os
import psycopg2
import psycopg2.extras
import json
import requests
from crewai.tools import tool
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
from typing import List
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD,
    DATABASE_NAME, DATABASE_TABLE, OLLAMA_BASE_URL, EMBEDDING_LLM, DIM
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a pretrained ColBERT reranker
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

logger = logging.getLogger(__name__)

def rerank(query: str, nodes: List[NodeWithScore], top_k: int = 5) -> List[NodeWithScore]:
    """Rerank candidate nodes using ColBERT."""
    docs = [n.node.text for n in nodes]
    inputs = [(query, d) for d in docs]

    encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**encodings).logits.squeeze(-1)

    # Attach scores back to nodes
    scored_nodes = []
    for node, score in zip(nodes, scores.tolist()):
        node.score = score  # overwrite similarity score with reranker score
        scored_nodes.append(node)

    # Sort by reranker score
    reranked = sorted(scored_nodes, key=lambda x: x.score, reverse=False)
    return reranked[:top_k]


@tool("pg_retriever_tool")
def pg_retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from pgvector using LlamaIndex VectorStore.
    
    Args:
        query: The search query string.
    
    Returns:
        A formatted string containing retrieved chunks.
    """
    if not query or query.strip() == "":
        return "Error: Empty query provided."
    
    try:
        # Configure embedding model
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_LLM,
            base_url=OLLAMA_BASE_URL,
        )

        # Connect to pgvector
        vector_store = PGVectorStore.from_params(
            database=DATABASE_NAME,
            host=DATABASE_HOST,
            password=DATABASE_PASSWORD,
            port=int(DATABASE_PORT),
            user=DATABASE_USER,
            schema_name="public",
            table_name=DATABASE_TABLE,
            hybrid_search=True,
            embed_dim=768,  # Ensure this matches your embedding dimension
        )

        # Create retriever
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=15)

        # Perform search
        nodes: List[NodeWithScore] = retriever.retrieve(query)

        results = rerank(query, nodes, top_k=5)

        if not results:
            return "No relevant documents found for this query."

        # Format results
        formatted_chunks: List[str] = []

        for i, r in enumerate(results, 1):
            node = r.node
            score = r.score
            # Handle score being a list or a float
            if isinstance(score, list):
                score_value = score[0] if score and isinstance(score[0], (float, int)) else None
            elif isinstance(score, (float, int)):
                score_value = score
            else:
                score_value = None

            file_name = node.metadata.get("file_name", node.metadata.get("source", "Unknown source"))
            page_no = node.metadata.get("page_no", "")
            page_info = f" (Page {page_no})" if page_no else ""

            if score_value is not None:
                relevance_str = f"{score_value:.3f}"
            else:
                relevance_str = "N/A"

            formatted_chunk = f"""**Document Chunk {i}** (Relevance: {relevance_str})\nSource: {file_name}{page_info}\n\nContent:\n{node.text.strip()}"""
            formatted_chunks.append(formatted_chunk)

        return "\n\n" + ("\n" + "="*80 + "\n\n").join(formatted_chunks)

    except Exception as e:
        logger.error(f"Error in pg_retriever_tool: {str(e)}")
        return f"Error retrieving documents: {str(e)}"


def get_ollama_embedding(text: str) -> List[float]:
    """Get embedding from Ollama for the given text."""
    try:
        # Use the embedding model from config
        embedding_model = EMBEDDING_LLM  # Default fallback
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": embedding_model,
                "prompt": text
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            print(f"DEBUG: Generated embedding of dimension {len(embedding)}")
            return embedding
        else:
            print(f"ERROR: Ollama embedding request failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"ERROR: Failed to get embedding from Ollama: {e}")
        return []


def perform_hybrid_search(conn, query: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, Dict, float]]:
    """Perform hybrid search combining vector similarity and text search with reranking."""
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check which table exists and has data
            table_name = f"data_{DATABASE_TABLE}"
            
            # Check if the configured table exists and has data
            cur.execute(f"""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (table_name,))
            
            if cur.fetchone()['count'] == 0:
                # Fallback to other possible table names
                for fallback_table in ['vectors', 'data_vectors']:
                    cur.execute(f"""
                        SELECT COUNT(*) as count 
                        FROM information_schema.tables 
                        WHERE table_name = %s
                    """, (fallback_table,))
                    
                    if cur.fetchone()['count'] > 0:
                        table_name = fallback_table
                        print(f"DEBUG: Using fallback table: {table_name}")
                        break
            
            # Check if table has any data
            cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = cur.fetchone()['count']
            print(f"DEBUG: Table {table_name} has {row_count} rows")
            
            if row_count == 0:
                return []
            
            # Get more candidates for reranking (2x the final top_k)
            candidate_limit = min(top_k * 3, 20)  # Get 3x more candidates, max 20
            
            # Perform vector similarity search
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Try different column names for embedding
            embedding_columns = ['embedding', 'dense_embedding']
            search_query = None
            
            for emb_col in embedding_columns:
                try:
                    # Check if column exists
                    cur.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = %s
                    """, (table_name, emb_col))
                    
                    if cur.fetchone():
                        search_query = f"""
                            SELECT 
                                text,
                                metadata_,
                                {emb_col} <=> %s::vector as distance
                            FROM {table_name}
                            WHERE {emb_col} IS NOT NULL
                            ORDER BY distance ASC
                            LIMIT %s
                        """
                        break
                except Exception as e:
                    print(f"DEBUG: Column {emb_col} check failed: {e}")
                    continue
            
            if not search_query:
                print("ERROR: No suitable embedding column found")
                return []
            
            print(f"DEBUG: Executing search query with embedding dimension {len(query_embedding)}")
            cur.execute(search_query, (embedding_str, candidate_limit))
            
            candidates = []
            for row in cur.fetchall():
                content = row['text'] or ""
                metadata = row['metadata_'] or {}
                distance = float(row['distance'])
                similarity_score = 1.0 - distance  # Convert distance to similarity
                
                candidates.append((content, metadata, similarity_score))
            
            print(f"DEBUG: Found {len(candidates)} candidates from vector search")
            
            # Rerank candidates using multiple criteria
            reranked_results = rerank_results(query, candidates, top_k)
            
            print(f"DEBUG: Reranked to {len(reranked_results)} final results")
            return reranked_results
            
    except Exception as e:
        print(f"ERROR: Hybrid search failed: {e}")
        return []


def rerank_results(query: str, candidates: List[Tuple[str, Dict, float]], top_k: int) -> List[Tuple[str, Dict, float]]:
    """Rerank search results using multiple scoring criteria."""
    try:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        reranked_candidates = []
        
        for content, metadata, vector_score in candidates:
            content_lower = content.lower()
            
            # 1. Vector similarity score (base score)
            final_score = vector_score * 0.4
            
            # 2. Exact phrase matching boost
            if query_lower in content_lower:
                final_score += 0.3
                print(f"DEBUG: Exact phrase match found in chunk")
            
            # 3. Keyword overlap score
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            if len(query_words) > 0:
                keyword_score = word_overlap / len(query_words)
                final_score += keyword_score * 0.2
            
            # 4. Content length penalty (prefer more substantial content)
            content_length = len(content.strip())
            if content_length > 100:  # Prefer chunks with substantial content
                final_score += 0.05
            elif content_length < 50:  # Penalize very short chunks
                final_score -= 0.1
            
            # 5. Metadata relevance boost
            if metadata:
                # Boost if query matches file name or page context
                file_name = metadata.get('file_name', '').lower()
                if any(word in file_name for word in query_words):
                    final_score += 0.05
                    print(f"DEBUG: Query word found in filename: {file_name}")
            
            # 6. Position-based scoring (earlier results get slight boost)
            position_penalty = candidates.index((content, metadata, vector_score)) * 0.01
            final_score -= position_penalty
            
            reranked_candidates.append((content, metadata, final_score))
        
        # Sort by final score (descending) and return top_k
        reranked_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Log reranking changes
        print("DEBUG: Reranking results:")
        for i, (content, metadata, score) in enumerate(reranked_candidates[:top_k]):
            original_idx = next(j for j, (c, m, s) in enumerate(candidates) if c == content)
            print(f"  Rank {i+1}: Original pos {original_idx+1}, Score: {score:.3f}")
        
        return reranked_candidates[:top_k]
        
    except Exception as e:
        print(f"ERROR: Reranking failed: {e}")
        # Fallback to original vector scores
        return candidates[:top_k]