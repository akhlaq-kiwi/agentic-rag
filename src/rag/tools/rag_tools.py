import os
import psycopg2
import psycopg2.extras
import json
import requests
from crewai.tools import tool
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, EMBEDDING_LLM, OLLAMA_BASE_URL, DIM, DATABASE_TABLE
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@tool("pg_retriever_tool")
def pg_retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from pgvector using direct PostgreSQL queries with hybrid search.
    
    This is a simplified, faster implementation that directly queries PostgreSQL
    without using LlamaIndex's query engine to avoid timeouts.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing the retrieved chunks
    """
    print(f"DEBUG: Tool received query parameter: {repr(query)}")
    
    # Validate input
    if not query or query.strip() == "":
        return "Error: Empty query provided."
    
    if query in ["The search query to find relevant documents"]:
        return "Error: Tool received schema placeholder instead of actual query."
    
    query = query.strip()
    
    try:
        # Get query embedding from Ollama
        query_embedding = get_ollama_embedding(query)
        if not query_embedding:
            return "Error: Failed to generate query embedding."
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            database=DATABASE_NAME,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD
        )
        
        # Perform hybrid search
        results = perform_hybrid_search(conn, query, query_embedding, top_k=5)
        
        conn.close()
        
        if not results:
            return "No relevant documents found for this query."
        
        # Format results with proper source attribution
        formatted_chunks = []
        
        for i, (content, metadata, score) in enumerate(results, 1):
            # Extract source information
            source_info = "Unknown source"
            page_info = ""
            
            if metadata:
                # Extract file name
                file_name = metadata.get('file_name', metadata.get('source', ''))
                if file_name:
                    source_info = file_name
                
                # Extract page number
                page_no = metadata.get('page_no', '')
                if page_no:
                    page_info = f" (Page {page_no})"
            
            # Format each chunk with clear source attribution
            formatted_chunk = f"""**Document Chunk {i}** (Relevance: {score:.3f})
Source: {source_info}{page_info}

Content:
{content.strip()}"""
            
            formatted_chunks.append(formatted_chunk)
        
        # Join all chunks with clear separators
        result_text = "\n\n" + ("\n" + "="*80 + "\n\n").join(formatted_chunks)
        
        print(f"DEBUG: Retrieved {len(results)} chunks with source metadata")
        print(result_text)
        return result_text
        
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