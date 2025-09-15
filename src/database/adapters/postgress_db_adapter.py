import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any
from ..base.base_db import BaseDB
import json
from src.config import DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT, DIM

class PostgresDbAdapter(BaseDB):
    def __init__(self, dim: int = 768):
        self.db_url = f"dbname={DATABASE_NAME} user={DATABASE_USER} password={DATABASE_PASSWORD} host={DATABASE_HOST} port={DATABASE_PORT}"
        self.dim = dim
        self.conn = psycopg2.connect(self.db_url)
        self.init_schema()  

    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    context TEXT,
                    embedding VECTOR({self.dim}),
                    sparse_embedding JSONB,
                    embedding_type TEXT DEFAULT 'dense',
                    dedup_key TEXT GENERATED ALWAYS AS (md5(content || metadata::text)) STORED,
                    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (dedup_key)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents USING GIN (fts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_sparse ON documents USING GIN (sparse_embedding);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);")
            self.conn.commit()

    def check_exists(self, content: str, metadata: dict) -> bool:
        """Check if a document already exists based on dedup_key"""
        with self.conn.cursor() as cur:
            # Use the same logic as the database's generated column
            # Convert metadata dict to JSON string exactly as PostgreSQL does
            metadata_json = json.dumps(metadata, separators=(',', ':'), sort_keys=True) if metadata else '{}'
            
            cur.execute(
                "SELECT 1 FROM documents WHERE md5(content || metadata::text) = md5(%s || %s) LIMIT 1",
                (content, metadata_json)
            )
            result = cur.fetchone() is not None
            
            # Debug logging
            if result:
                print(f"DEBUG: Found existing chunk for content: {content[:50]}...")
            else:
                print(f"DEBUG: No existing chunk found for content: {content[:50]}...")
            return result

    def insert(self, records: List[Dict[str, Any]]):
        with self.conn.cursor() as cur:
            values = []
            for r in records:
                # Handle both hybrid and legacy embedding formats
                if "embedding_type" in r and r["embedding_type"] == "hybrid":
                    # Hybrid embedding with both dense and sparse
                    values.append((
                        r["text"], 
                        json.dumps(r.get("metadata", {})), 
                        r.get("context", ""),  # Store context separately
                        r["dense_embedding"],
                        json.dumps(r.get("sparse_embedding", [])),
                        "hybrid"
                    ))
                else:
                    # Legacy dense-only embedding
                    values.append((
                        r["text"], 
                        json.dumps(r.get("metadata", {})), 
                        r.get("context", ""),  # Store context separately
                        r["embedding"],
                        json.dumps([]),  # Empty sparse embedding
                        "dense"
                    ))
            
            execute_values(
                cur,
                """
                INSERT INTO documents (content, metadata, context, embedding, sparse_embedding, embedding_type)
                VALUES %s
                ON CONFLICT (dedup_key) DO NOTHING
                """,
                values
            )
            self.conn.commit()

    def search(self, query_embedding: List[float], top_k: int = 5):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata, (embedding <#> %s::vector) AS distance
                FROM documents
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, top_k)
            )
            return cur.fetchall()

    def hybrid_search(self, query: str, query_embedding, top_k: int = 5, alpha: float = 0.5, sparse_embedding: List[float] = None):
        """
        Enhanced hybrid search: combine dense vector + sparse vector + keyword scores
        alpha = weight for dense vector (0.0 = pure keyword+sparse, 1.0 = pure dense vector)
        """
        # Normalize embedding type
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        with self.conn.cursor() as cur:
            if sparse_embedding is not None:
                # Advanced hybrid search with sparse embeddings
                cur.execute(
                    """
                    WITH sparse_scores AS (
                        SELECT id,
                            content,
                            metadata,
                            embedding,
                            (embedding <#> %s::vector) AS dense_distance,
                            ts_rank(fts, plainto_tsquery('english', %s)) AS keyword_score,
                            CASE 
                                WHEN sparse_embedding != 'null'::jsonb AND sparse_embedding != '[]'::jsonb 
                                THEN (
                                    SELECT SUM((sparse_val::numeric) * (query_val::numeric))
                                    FROM jsonb_array_elements_text(sparse_embedding) WITH ORDINALITY AS t1(sparse_val, idx)
                                    JOIN jsonb_array_elements_text(%s::jsonb) WITH ORDINALITY AS t2(query_val, idx2) 
                                        ON t1.idx = t2.idx2
                                    WHERE sparse_val::numeric > 0 AND query_val::numeric > 0
                                ) 
                                ELSE 0 
                            END AS sparse_score
                        FROM documents
                        WHERE fts @@ plainto_tsquery('english', %s) OR embedding_type = 'hybrid'
                    )
                    SELECT id, content, metadata, dense_distance, keyword_score, sparse_score,
                        (
                            %s * dense_distance + 
                            (1 - %s) * 0.5 * (1 - keyword_score) + 
                            (1 - %s) * 0.5 * (1 - LEAST(sparse_score / NULLIF(GREATEST(sparse_score, 0.1), 0), 1))
                        ) AS hybrid_score
                    FROM sparse_scores
                    ORDER BY hybrid_score ASC
                    LIMIT %s;
                    """,
                    (query_embedding, query, json.dumps(sparse_embedding), query, alpha, alpha, alpha, top_k)
                )
            else:
                # Fallback to original FTS + dense vector hybrid search
                cur.execute(
                    """
                    SELECT id,
                        content,
                        metadata,
                        (embedding <#> %s::vector) AS vector_distance,
                        ts_rank(fts, plainto_tsquery('english', %s)) AS keyword_score,
                        ((1 - %s) * (1 - ts_rank(fts, plainto_tsquery('english', %s))) +
                            %s * (embedding <#> %s::vector)) AS hybrid_score
                    FROM documents
                    WHERE fts @@ plainto_tsquery('english', %s)
                    ORDER BY hybrid_score ASC
                    LIMIT %s;
                    """,
                    (query_embedding, query, alpha, query, alpha, query_embedding, query, top_k)
                )
            return cur.fetchall()
