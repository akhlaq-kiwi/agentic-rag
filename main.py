from src.config import EXTRACTOR, EXPORT_FORMAT, RAW_DATA_PATH

from src.logger import get_logger

from src.data_ingestion import DocumentIngestor
from src.data_ingestion import ExtractorFactory

logger = get_logger("Ingestion")

if __name__ == "__main__":
    extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    ingestor = DocumentIngestor(extractor)
    content = ingestor.ingest(f"{RAW_DATA_PATH}/HR Bylaws.pdf", export_format=EXPORT_FORMAT)
    print(content[:100000])