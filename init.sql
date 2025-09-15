-- Initialize database with pgvector extension and documents table
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for hybrid RAG storage
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    context TEXT,
    embedding vector(768),
    sparse_embedding JSONB,
    embedding_type TEXT DEFAULT 'dense',
    dedup_key TEXT GENERATED ALWAYS AS (md5(content || metadata::text)) STORED,
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dedup_key)
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index for full-text search (hybrid search)
CREATE INDEX IF NOT EXISTS documents_fts_idx 
ON documents USING GIN (fts);

-- Create index for sparse embeddings
CREATE INDEX IF NOT EXISTS documents_sparse_idx 
ON documents USING GIN (sparse_embedding);

-- Create index on metadata for filtering
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING GIN (metadata);

-- Create index on dedup_key for efficient deduplication
CREATE INDEX IF NOT EXISTS documents_dedup_idx 
ON documents (dedup_key);
