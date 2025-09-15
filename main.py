from src.data_ingestion.chuncker import DocumentChuncker
from src.data_ingestion.factories.chunker_factory import ChunkerFactory
from src.config import EXTRACTOR, EXPORT_FORMAT, RAW_DATA_PATH, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT

from src.logger import get_logger

from src.data_ingestion import DocumentIngestor
from src.data_ingestion import ExtractorFactory
from src.data_ingestion.factories.embedding_factory import EmbedderFactory
from src.data_ingestion import DocumentIndexer
from src.database.factories.db_factory import DBFactory
from src.database.db_client import DBClient

logger = get_logger("Ingestion")

if __name__ == "__main__":
    extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    ingestor = DocumentIngestor(extractor)
    processed_files = ingestor.ingest_dir(f"{RAW_DATA_PATH}", export_format=EXPORT_FORMAT)
    #print(processed_files)

    # Initialize components for streaming processing
    chunker = ChunkerFactory.get_chunker("markdown")
    chuncker = DocumentChuncker(chunker=chunker)
    
    # Use hybrid search adapter for better retrieval accuracy
    embedder = EmbedderFactory.get_embedder(
        "hybrid",
        dense_model_name="nomic-embed-text",
        context_model="gemma:2b",
        base_url="http://ollama:11434",
        use_context_enrichment=True,
        max_features=5000
    )
    
    db_client = DBClient(DBFactory.get_db("postgres"))
    
    # Stream processing: chunk -> embed -> save immediately
    def progress_callback(chunk, chunk_num, total_chunks, source, skipped=False):
        """Optional callback to show progress"""
        from pathlib import Path
        source_name = Path(source).name
        if skipped:
            logger.info(f"Progress: {chunk_num}/{total_chunks} chunks from {source_name} (SKIPPED - already exists)")
        else:
            logger.info(f"Progress: {chunk_num}/{total_chunks} chunks from {source_name} (PROCESSED)")
    
    total_processed = chuncker.chunk_and_process_streaming(
        sources=processed_files,
        embedder=embedder,
        db_client=db_client,
        chunk_callback=progress_callback
    )
    
    logger.info(f"Streaming ingestion completed! Total chunks processed: {total_processed}")


    # db_client = DBClient(DBFactory.get_db("postgres"))
    # embedder = EmbedderFactory.get_embedder("sbert", model_name="intfloat/e5-base-v2")
    # query = "introduction to procurement manual"
    # query_emb = embedder.embed([query])[0].tolist()

    # results = db_client.hybrid_search(query, query_emb, top_k=5, alpha=0.6)

    # for r in results:
    #     print(r[0], r[1][:60], r[2])  # id, content snippet, metadata



