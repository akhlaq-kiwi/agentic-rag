import os
from dotenv import load_dotenv

# Load env file
load_dotenv()

EXTRACTOR = os.getenv("EXTRACTOR", "docling")
EXPORT_FORMAT = os.getenv("EXPORT_FORMAT", "text")
SOURCE_DATA_PATH = os.getenv("SOURCE_DATA_PATH", "data")
RAW_DATA_PATH = f"{SOURCE_DATA_PATH}/raw/"
PROCESSED_DATA_PATH = f"{SOURCE_DATA_PATH}/processed/"
ENABLE_OCR=bool(os.getenv("ENABLE_OCR", "false"))
ENABLE_TABLES=bool(os.getenv("ENABLE_TABLES", "false"))

DATABASE_HOST=os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT=int(os.getenv("DATABASE_PORT", 5432))
DATABASE_USER=os.getenv("DATABASE_USER", "postgres")
DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "secret")
DATABASE_NAME=os.getenv("DATABASE_NAME", "test")
DATABASE_DIM=int(os.getenv("DATABASE_DIM", 384))




