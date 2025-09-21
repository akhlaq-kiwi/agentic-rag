-- Initialize database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create LlamaIndex-compatible vectors table
-- LlamaIndex PGVectorStore requires specific column names:
-- id: BIGSERIAL PRIMARY KEY
-- text: TEXT (not 'content')
-- metadata_: JSONB (with underscore)
-- node_id: VARCHAR
-- embedding: vector(768)

CREATE TABLE IF NOT EXISTS vectors (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    metadata_ JSONB,
    node_id VARCHAR,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS vectors_embedding_idx
ON vectors USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for metadata filtering
CREATE INDEX IF NOT EXISTS vectors_metadata_idx
ON vectors USING GIN (metadata_);

-- Create index on node_id for fast lookups
CREATE INDEX IF NOT EXISTS vectors_node_id_idx
ON vectors (node_id);

-- Add full-text search column to existing vectors table (if missing)
DO $$
BEGIN
    -- Check if text_search_tsv column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'vectors'
        AND column_name = 'text_search_tsv'
    ) THEN
        -- Add the column with generated tsvector
        ALTER TABLE vectors ADD COLUMN text_search_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, ''))) STORED;

        -- Create GIN index for full-text search
        CREATE INDEX IF NOT EXISTS vectors_text_search_idx
        ON vectors USING GIN (text_search_tsv);
    END IF;
END $$;

-- Also ensure data_vectors table exists for backward compatibility (if needed)
CREATE TABLE IF NOT EXISTS data_vectors (
    id BIGSERIAL PRIMARY KEY,
    text TEXT,
    metadata_ JSONB,
    node_id VARCHAR,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add text_search_tsv column to data_vectors table if missing
DO $$
BEGIN
    -- Check if text_search_tsv column exists in data_vectors
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'data_vectors'
        AND column_name = 'text_search_tsv'
    ) THEN
        -- Add the column with generated tsvector
        ALTER TABLE data_vectors ADD COLUMN text_search_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, ''))) STORED;

        -- Create GIN index for full-text search
        CREATE INDEX IF NOT EXISTS data_vectors_text_search_idx
        ON data_vectors USING GIN (text_search_tsv);
    END IF;
END $$;
