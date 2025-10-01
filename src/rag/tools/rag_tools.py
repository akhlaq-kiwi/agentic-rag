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
    DATABASE_NAME, DATABASE_TABLE, OLLAMA_BASE_URL, EMBEDDING_LLM, DIM,
    RETRIEVAL_INITIAL_TOP_K, RETRIEVAL_FINAL_TOP_K, 
    RETRIEVAL_SIMILARITY_CUTOFF, RERANKING_DIVERSITY_WEIGHT
)

from sentence_transformers import CrossEncoder
import torch

# Load a proper cross-encoder reranker model
# This model is specifically trained for reranking tasks
try:
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    logger = logging.getLogger(__name__)
    logger.info("✅ Reranker model loaded successfully")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to load reranker model: {e}")
    reranker_model = None

# Simple cache to prevent duplicate tool calls within the same session
_TOOL_CALL_CACHE = {}
_CACHE_MAX_SIZE = 10

# Call counter to enforce single tool call per query execution
_TOOL_CALL_COUNTER = {}
_MAX_CALLS_PER_QUERY = 1

def rerank(query: str, nodes: List[NodeWithScore], top_k: int = 5, diversity_weight: float = 0.3) -> List[NodeWithScore]:
    """
    Rerank candidate nodes using cross-encoder model with diversity.
    
    Args:
        query: Search query
        nodes: List of candidate nodes from vector search
        top_k: Number of top results to return
        diversity_weight: Weight for diversity (0.0 = no diversity, 1.0 = max diversity)
    
    Returns:
        List of reranked nodes with updated scores
    """
    if not nodes:
        return []
    
    if reranker_model is None:
        # Fallback: use original vector similarity scores
        logger.warning("Reranker model not available, using vector similarity scores")
        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score else 0, reverse=True)
        return sorted_nodes[:top_k]
    
    try:
        # Prepare query-document pairs for reranking
        query_doc_pairs = [[query, node.node.text] for node in nodes]
        
        # Get reranking scores from cross-encoder
        rerank_scores = reranker_model.predict(query_doc_pairs)
        
        # Normalize scores to 0-1 range
        if len(rerank_scores) > 1:
            min_score = float(np.min(rerank_scores))
            max_score = float(np.max(rerank_scores))
            if max_score > min_score:
                rerank_scores = (rerank_scores - min_score) / (max_score - min_score)
        
        # Attach reranked scores to nodes
        scored_nodes = []
        for node, score in zip(nodes, rerank_scores):
            node.score = float(score)
            scored_nodes.append(node)
        
        # Apply diversity: penalize chunks from same document
        if diversity_weight > 0:
            scored_nodes = apply_diversity_penalty(scored_nodes, diversity_weight)
        
        # Sort by final score (descending - higher is better)
        reranked = sorted(scored_nodes, key=lambda x: x.score, reverse=True)
        
        logger.info(f"Reranked {len(nodes)} nodes, returning top {top_k}")
        return reranked[:top_k]
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}, falling back to original scores")
        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score else 0, reverse=True)
        return sorted_nodes[:top_k]


def apply_diversity_penalty(nodes: List[NodeWithScore], diversity_weight: float) -> List[NodeWithScore]:
    """
    Apply diversity penalty to avoid returning too many chunks from the same document.
    
    Args:
        nodes: List of scored nodes
        diversity_weight: Penalty weight (0.0 = no penalty, 1.0 = max penalty)
    
    Returns:
        Nodes with adjusted scores for diversity
    """
    document_counts = {}
    adjusted_nodes = []
    
    for node in nodes:
        # Get document identifier (file_name or source)
        doc_id = node.node.metadata.get("file_name", node.node.metadata.get("source", "unknown"))
        
        # Count how many times we've seen this document
        count = document_counts.get(doc_id, 0)
        document_counts[doc_id] = count + 1
        
        # Apply penalty: reduce score for repeated documents
        # First chunk from doc: no penalty
        # Second chunk: small penalty
        # Third+ chunk: larger penalty
        penalty = diversity_weight * (count * 0.15)  # 15% penalty per repeat
        adjusted_score = node.score * (1 - penalty)
        
        node.score = max(0, adjusted_score)  # Ensure score doesn't go negative
        adjusted_nodes.append(node)
    
    return adjusted_nodes


@tool("pg_retriever_tool")
def pg_retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from pgvector using LlamaIndex VectorStore.
    
    This tool should be called ONCE per query. Do not retry if results are returned.
    
    Args:
        query (str): The search query string. Pass the query directly as a string, 
                     NOT as a dictionary or nested object. 
                     Example: "What is information security?"
    
    Returns:
        str: A formatted string containing retrieved chunks with source information.
    """
    # Handle case where agent passes dict instead of string
    if isinstance(query, dict):
        logger.warning(f"pg_retriever_tool received dict instead of string: {query}")
        # Try to extract query from common dict formats
        if 'query' in query:
            query = query['query']
        elif 'description' in query:
            query = query['description']
        elif 'text' in query:
            query = query['text']
        else:
            # Get first value if dict has any values
            query = next(iter(query.values())) if query else None
    
    # Validate input
    if not query:
        logger.warning("pg_retriever_tool called with None query")
        return "Error: Query parameter is required. Please provide a valid search query as a string."
    
    query = str(query).strip()
    if not query:
        logger.warning("pg_retriever_tool called with empty query")
        return "Error: Empty query provided. Please provide a valid search query as a string."
    
    # Log tool invocation for debugging
    logger.info(f"pg_retriever_tool invoked with query: '{query[:100]}...'")
    
    # Check cache to prevent duplicate calls
    cache_key = query.lower().strip()
    if cache_key in _TOOL_CALL_CACHE:
        logger.info(f"Returning cached result for query: '{query[:50]}...'")
        return _TOOL_CALL_CACHE[cache_key]
    
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

        # Create retriever with optimized parameters
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
        # Use configurable retrieval parameters
        retriever = index.as_retriever(
            similarity_top_k=RETRIEVAL_INITIAL_TOP_K,
            similarity_cutoff=RETRIEVAL_SIMILARITY_CUTOFF
        )

        # Perform search
        nodes: List[NodeWithScore] = retriever.retrieve(query)
        
        logger.info(f"Retrieved {len(nodes)} initial candidates from vector search")

        # Rerank with diversity to get best chunks
        results = rerank(query, nodes, top_k=RETRIEVAL_FINAL_TOP_K, diversity_weight=RERANKING_DIVERSITY_WEIGHT)

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

        result = "\n\n" + ("\n" + "="*80 + "\n\n").join(formatted_chunks)
        logger.info(f"pg_retriever_tool returning {len(results)} chunks")
        
        # Cache the result
        _TOOL_CALL_CACHE[cache_key] = result
        # Limit cache size
        if len(_TOOL_CALL_CACHE) > _CACHE_MAX_SIZE:
            # Remove oldest entry (first key)
            oldest_key = next(iter(_TOOL_CALL_CACHE))
            del _TOOL_CALL_CACHE[oldest_key]
        
        return result

    except Exception as e:
        logger.error(f"Error in pg_retriever_tool: {str(e)}", exc_info=True)
        return f"Error retrieving documents: {str(e)}. This is a system error, not a query issue. Do not retry the tool call."


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