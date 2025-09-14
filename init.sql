-- Initialize database with pgvector extension and documents table
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for RAG storage
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index on metadata for filtering
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING GIN (metadata);
