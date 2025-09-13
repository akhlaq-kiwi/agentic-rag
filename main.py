from src.data_ingestion.chuncker import DocumentChuncker
from src.data_ingestion.factories.chunker_factory import ChunkerFactory
from src.config import EXTRACTOR, EXPORT_FORMAT, RAW_DATA_PATH

from src.logger import get_logger

from src.data_ingestion import DocumentIngestor
from src.data_ingestion import ExtractorFactory

logger = get_logger("Ingestion")

if __name__ == "__main__":
    extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    ingestor = DocumentIngestor(extractor)
    processed_files = ingestor.ingest_dir(f"{RAW_DATA_PATH}", export_format=EXPORT_FORMAT)
    print(processed_files)

    chunker = ChunkerFactory.get_chunker("markdown")
    chuncker = DocumentChuncker(chunker=chunker)
    chunks = chuncker.chunk(processed_files)
    print(chunks)
