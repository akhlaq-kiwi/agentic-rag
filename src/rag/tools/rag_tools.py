from crewai.tools import tool
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
from src.database.factories.db_factory import DBFactory
from src.database.db_client import DBClient
from src.data_ingestion.factories.embedding_factory import EmbedderFactory
import logging

logger = logging.getLogger(__name__)

@tool
def pg_retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from pgvector using hybrid search (dense + sparse + keyword).
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing the retrieved chunks
    """
    try:
        # Initialize hybrid embedder for query processing
        embedder = EmbedderFactory.get_embedder(
            "hybrid",
            dense_model_name="nomic-embed-text",
            context_model="gemma:2b",
            base_url=OLLAMA_BASE_URL,
            use_context_enrichment=False,  # Skip context for queries
            max_features=5000
        )
        
        # Connect to database
        db = DBFactory.get_db("postgres")
        db_client = DBClient(db)
        
        # Transform query to get both dense and sparse embeddings
        query_embeddings = embedder.transform_query(query)
        
        # Perform hybrid search
        results = db_client.db.hybrid_search(
            query=query,
            query_embedding=query_embeddings["dense_embedding"],
            sparse_embedding=query_embeddings["sparse_embedding"],
            top_k=5,
            alpha=0.7  # 70% dense vector, 30% sparse + keyword
        )
        
        # Format results
        if not results:
            return "No relevant documents found for the query."
            
        formatted_results = []
        for i, result in enumerate(results, 1):
            doc_id, content, metadata, *scores = result
            
            # Extract metadata info if available
            meta_info = ""
            if metadata:
                import json
                try:
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    source = meta_dict.get('source', 'Unknown')
                    section = meta_dict.get('section', '')
                    meta_info = f" (Source: {source}" + (f", Section: {section}" if section else "") + ")"
                except:
                    pass
            
            # Format score information
            score_info = ""
            if len(scores) >= 3:
                score_info = f" [Dense: {scores[0]:.3f}, Keyword: {scores[1]:.3f}, Sparse: {scores[2]:.3f}]"
            elif len(scores) >= 1:
                score_info = f" [Score: {scores[0]:.3f}]"
            
            formatted_results.append(f"[Chunk {i}{meta_info}]{score_info}\n{content}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in pg_retriever_tool: {str(e)}")
        return f"Error retrieving documents: {str(e)}"
