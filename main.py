from src.config import EXTRACTOR, EXPORT_FORMAT, RAW_DATA_PATH, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT

from src.logger import get_logger

from src.data_ingestion import DocumentIngestor
from src.data_ingestion import ExtractorFactory

logger = get_logger("Ingestion")

if __name__ == "__main__":
    extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    ingestor = DocumentIngestor(extractor)
    processed_files = ingestor.ingest_dir(f"{RAW_DATA_PATH}", export_format=EXPORT_FORMAT)
   
    



