from src.config import EXTRACTOR, EXPORT_FORMAT, INDEXER, RAW_DATA_PATH, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT

from src.logger import get_logger

from src.data_ingestion import DocumentIngestor
from src.data_ingestion import ExtractorFactory
from src.data_ingestion.factories.indexer_factory import IndexerFactory
from src.data_ingestion.indexer import DocumentIndexer

logger = get_logger("Ingestion")

if __name__ == "__main__":
    # Step 1: Extract and process documents
    extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    ingestor = DocumentIngestor(extractor)
    processed_files = ingestor.ingest_dir(f"{RAW_DATA_PATH}", export_format=EXPORT_FORMAT)
    logger.info(f"Processed {len(processed_files)} files: {processed_files}")
    
    # Step 2: Create vector index
    _indexer = IndexerFactory.get_indexer('llamaindex', 
                                         enable_context_enrichment=False,  # Set to False to disable
                                         context_timeout=30, 
                                         chunk_size=50, chunk_overlap=10,
                                         min_chars=50,
                                         mode="sentence"
                                        )  # Increase timeout if needed
    indexer = DocumentIndexer(_indexer)
    index = indexer.index(processed_files)
    logger.info("Document indexing completed successfully!")
    
    # Step 3: Get index statistics
    stats = _indexer.get_index_stats()
    logger.info(f"Index statistics: {stats}")

