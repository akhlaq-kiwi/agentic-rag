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
                    embedding VECTOR({self.dim}),
                    dedup_key TEXT GENERATED ALWAYS AS (md5(content || metadata::text)) STORED,
                    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    UNIQUE (dedup_key)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents USING GIN (fts);")
            self.conn.commit()

    def insert(self, records: List[Dict[str, Any]]):
        with self.conn.cursor() as cur:
            values = [
                (r["text"], json.dumps(r.get("metadata", {})), r["embedding"].tolist())
                for r in records
            ]
            execute_values(
                cur,
                """
                INSERT INTO documents (content, metadata, embedding)
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

    def hybrid_search(self, query: str, query_embedding, top_k: int = 5, alpha: float = 0.5):
        """
        Hybrid search: combine vector + keyword scores
        alpha = weight for vector (0.0 = pure keyword, 1.0 = pure vector)
        """
        # ✅ Normalize embedding type
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
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
