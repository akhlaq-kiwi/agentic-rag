# Agentic RAG Docker Setup

Complete containerized setup for the Agentic RAG system with all components.

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| **postgres** | 5434 | PostgreSQL with pgvector for document storage |
| **ollama** | 11434 | Local LLM inference (Gemma 2B) |
| **rag-api** | 8000 | FastAPI RAG service with CrewAI agents |
| **open-webui** | 3000 | Chat interface for RAG system |
| **phoenix** | 6006 | Arize Phoenix for observability & tracing |
| **ingestion** | - | Data ingestion pipeline (run-once) |
| **evaluation** | - | RAGAS evaluation service (run-once) |

## Quick Start

### 1. Start Core Services
```bash
# Start all persistent services
docker-compose up -d postgres ollama rag-api open-webui phoenix

# Check service health
docker compose ps
```

### 2. Setup Ollama Models
```bash
# Pull required models
docker compose exec ollama ollama pull gemma:2b
docker compose exec ollama ollama pull nomic-embed-text
```

### 3. Run Data Ingestion
```bash
# Ingest documents from data/raw/ folder
docker compose --profile ingestion up ingestion
```

### 4. Access Services
- **Chat Interface**: http://localhost:3000 (OpenWebUI)
- **RAG API**: http://localhost:8000/docs (FastAPI docs)
- **Phoenix Dashboard**: http://localhost:6006 (Observability)
- **Database**: localhost:5434 (PostgreSQL)

## Advanced Usage

### Run Evaluation

⚠️ **Prerequisites**: RAGAS evaluation requires an OpenAI API key.

#### 1. Setup OpenAI API Key
```bash
# Set your OpenAI API key (required for RAGAS evaluation)
export OPENAI_API_KEY=your-openai-api-key-here

# Or add to .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env

# Test API connectivity (optional but recommended)
python test_openai_connection.py
```

#### 2. Run Evaluation
```bash
# Run RAGAS evaluation with default small dataset
OPENAI_API_KEY=$OPENAI_API_KEY docker compose --profile evaluation up evaluation

# Run evaluation with large dataset
EVALUATION_DATASET=large_dataset.jsonl OPENAI_API_KEY=$OPENAI_API_KEY docker compose --profile evaluation up evaluation

# Run evaluation with custom dataset
EVALUATION_DATASET=my_custom_dataset.jsonl OPENAI_API_KEY=$OPENAI_API_KEY docker compose --profile evaluation up evaluation

# View evaluation results
ls -la evaluation_results/

# View specific evaluation result
cat evaluation_results/ragas_evaluation_small_dataset_*.json
```

#### 3. Troubleshooting Evaluation
If you encounter OpenAI API errors:
```bash
# Test your API key
python test_openai_connection.py

# Check evaluation logs
docker compose logs evaluation

# Common issues:
# - Invalid API key: Verify your key at https://platform.openai.com/api-keys
# - Insufficient credits: Check your OpenAI account balance
# - Rate limiting: Wait and retry, or upgrade your plan
```

### Development Mode
```bash
# Use override for development (live code reload)
docker compose -f docker-compose.yaml -f docker-compose.override.yml up -d
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag-api
docker compose logs -f phoenix
```

## Configuration

### Environment Variables
- **Production**: Uses `.env.docker` (container networking)
- **Development**: Uses `.env` (local networking)

### Data Persistence
- `postgres_data`: Database storage
- `ollama_data`: Model storage
- `open_webui_data`: Chat history
- `phoenix_data`: Observability data

### Evaluation Datasets
Available datasets in `data/evaluation_data/`:
- `small_dataset.jsonl`: Quick evaluation with 10-20 questions
- `large_dataset.jsonl`: Comprehensive evaluation with 100+ questions

### Evaluation Results
- Results are automatically saved to `evaluation_results/` directory
- Each run creates a timestamped JSON file with metrics and metadata
- Results include RAGAS metrics: faithfulness, answer relevancy, context precision/recall

## Troubleshooting

### Service Dependencies
Services start in order with health checks:
1. PostgreSQL (with pgvector)
2. Ollama (with models)
3. RAG API (depends on DB + Ollama)
4. OpenWebUI (depends on RAG API)

### Common Issues

**Ollama models not found:**
```bash
docker-compose exec ollama ollama list
docker-compose exec ollama ollama pull gemma:2b
```

**Database connection issues:**
```bash
docker-compose exec postgres pg_isready -U postgres
```

**Memory issues with Gemma:**
- Gemma 2B requires ~3GB RAM
- Use `gemma:2b` for lower memory usage
- Monitor with `docker stats`

### Reset Everything
```bash
# Stop all services and remove data
docker-compose down -v
docker-compose up -d
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │────│    RAG API      │────│   PostgreSQL    │
│   (Chat UI)     │    │  (CrewAI Agents)│    │   (pgvector)    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 5434    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐    ┌─────────────────┐
                    │     Ollama      │    │  Arize Phoenix  │
                    │   (LLM/Embed)   │    │ (Observability) │
                    │  Port: 11434    │    │   Port: 6006    │
                    └─────────────────┘    └─────────────────┘
```

## RAGAS Evaluation Metrics

The evaluation service measures:
- **Faithfulness**: Answer accuracy to retrieved context
- **Answer Relevancy**: Relevance of answer to question
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of relevant context
- **Answer Correctness**: Factual correctness
- **Answer Similarity**: Semantic similarity to ground truth

## Phoenix Observability

Monitor your RAG system with:
- LLM call traces and latencies
- Agent conversation flows
- Embedding similarity scores
- Error tracking and debugging
- Performance metrics and analytics
