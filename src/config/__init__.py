import os
from dotenv import load_dotenv

# Load env file
load_dotenv()

EXTRACTOR = os.getenv("EXTRACTOR", "docling")
EXPORT_FORMAT = os.getenv("EXPORT_FORMAT", "text")
SOURCE_DATA_PATH = os.getenv("SOURCE_DATA_PATH", "data")
RAW_DATA_PATH = f"{SOURCE_DATA_PATH}/raw/"
PROCESSED_DATA_PATH = f"{SOURCE_DATA_PATH}/processed/"

