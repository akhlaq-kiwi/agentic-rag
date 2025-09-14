# Agentic RAG Setup Guide

## Prerequisites

1. **PostgreSQL with pgvector extension**
   ```bash
   # Install PostgreSQL if not already installed
   brew install postgresql
   
   # Install pgvector extension
   brew install pgvector
   ```

2. **Ollama for local LLM**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull required models
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```

## Installation Steps

1. **Clone and setup environment**
   ```bash
   cd /Applications/MAMP/htdocs/agentic-rag
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirement.txt
   ```

2. **Configure environment variables**
   ```bash
   # Copy example env file
   cp .env.example .env
   
   # Edit .env file with your database credentials
   nano .env
   ```

3. **Setup PostgreSQL database**
   ```sql
   -- Create database
   CREATE DATABASE test;
   
   -- Connect to database
   \c test;
   
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   
   -- Create documents table
   CREATE TABLE IF NOT EXISTS documents (
       id SERIAL PRIMARY KEY,
       content TEXT,
       metadata JSONB,
       embedding vector(768)
   );
   
   -- Create index for similarity search
   CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
   ```

4. **Ingest documents**
   ```bash
   # Place your PDF documents in data/raw/
   # Run the ingestion script
   python main.py
   ```

5. **Start the RAG server**
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # In another terminal, start the FastAPI server
   python run_server.py
   ```

## API Usage

Once the server is running, you can query the RAG system:

```bash
# Test the API
curl "http://localhost:8000/query?question=What%20is%20procurement%20manual"
```

Or visit the interactive API docs at: `http://localhost:8000/docs`

## Troubleshooting

1. **Ollama connection error**: Make sure Ollama is running (`ollama serve`)
2. **Database connection error**: Check PostgreSQL is running and credentials in .env are correct
3. **Embedding dimension mismatch**: Ensure DIM=768 in .env matches your embedding model
4. **No documents found**: Run `python main.py` to ingest documents first

## Architecture

- **CrewAI Agents**: Three specialized agents work together
  - Retriever Agent: Fetches relevant chunks from pgvector
  - RAG Orchestrator: Formats retrieved context
  - LLM Generator: Generates final answers using Llama3

- **Vector Store**: PostgreSQL with pgvector for semantic search
- **Embeddings**: Nomic-embed-text via Ollama
- **LLM**: Llama3:8b via Ollama
