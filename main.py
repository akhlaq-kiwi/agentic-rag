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
    # extractor = ExtractorFactory.get_extractor(EXTRACTOR)
    # ingestor = DocumentIngestor(extractor)
    # processed_files = ingestor.ingest_dir(f"{RAW_DATA_PATH}", export_format=EXPORT_FORMAT)
    # #print(processed_files)

    # chunker = ChunkerFactory.get_chunker("markdown")
    # chuncker = DocumentChuncker(chunker=chunker)
    # chunks = chuncker.chunk(processed_files)
    # print(chunks)

    # embedder = EmbedderFactory.get_embedder("sbert")
    # indexer = DocumentIndexer(embedder=embedder)
    # indexed_chunks = indexer.index(chunks)
    # print(indexed_chunks[0])

    # db_client = DBClient(DBFactory.get_db("postgres"))
    # db_client.insert(indexed_chunks)


    db_client = DBClient(DBFactory.get_db("postgres"))
    embedder = EmbedderFactory.get_embedder("sbert")
    query = "Tell me about Compliance with Applicable Legislations"
    query_emb = embedder.embed([query])[0].tolist()

    results = db_client.hybrid_search(query, query_emb, top_k=5, alpha=0.6)

    for r in results:
        print(r[0], r[1][:60], r[2])  # id, content snippet, metadata



