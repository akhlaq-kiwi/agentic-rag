import os
from dotenv import load_dotenv

# Load env file
load_dotenv()

EXTRACTOR = os.getenv("EXTRACTOR", "docling")
EXPORT_FORMAT = os.getenv("EXPORT_FORMAT", "markdown")
INDEXER = os.getenv("INDEXER", "llamaindex")
SOURCE_DATA_PATH = os.getenv("SOURCE_DATA_PATH", "data")
RAW_DATA_PATH = f"{SOURCE_DATA_PATH}/raw/"
EVALUATION_DATA_PATH = f"{SOURCE_DATA_PATH}/evaluation_data/"
EVALUATION_RESULTS_PATH = f"{EVALUATION_DATA_PATH}/evaluation_results/"
PROCESSED_DATA_PATH = f"{SOURCE_DATA_PATH}/processed/"
ENABLE_OCR=bool(os.getenv("ENABLE_OCR", "false"))
ENABLE_TABLES=bool(os.getenv("ENABLE_TABLES", "false"))

DATABASE_HOST=os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT=int(os.getenv("DATABASE_PORT", 5432))
DATABASE_USER=os.getenv("DATABASE_USER", "postgres")
DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "secret")
DATABASE_NAME=os.getenv("DATABASE_NAME", "test")
DATABASE_TABLE=os.getenv("DATABASE_TABLE", "vectors")
DATABASE_DIM=int(os.getenv("DATABASE_DIM", 384))
DIM=int(os.getenv("DIM", 768))
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_LLM=os.getenv("EMBEDDING_LLM", "text-embedding-ada-002")
LLM=os.getenv("LLM", "llama3.2")
PHOENIX_COLLECTOR_ENDPOINT=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
PHOENIX_PROJECT_NAME=os.getenv("PHOENIX_PROJECT_NAME", "default")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")


REDIS_HOST=os.getenv("REDIS_HOST", "localhost")
REDIS_PORT=int(os.getenv("REDIS_PORT", 6379))
REDIS_DB=int(os.getenv("REDIS_DB", 0))
CONVERSATION_HISTORY_LIMIT=int(os.getenv("CONVERSATION_HISTORY_LIMIT", 5))
REDIS_PORT=6379



