from crewai.tools import tool
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
import logging

logger = logging.getLogger(__name__)

@tool
def pg_retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from pgvector using semantic similarity.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing the retrieved chunks
    """
    try:
        # Initialize embedding model
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text", 
            base_url=OLLAMA_BASE_URL
        )
        
        # Connect to vector store
        vector_store = PGVectorStore.from_params(
            database=DATABASE_NAME,
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            table_name='documents',
            embed_dim=DIM,
        )
        
        # Create index and retriever
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        retriever = index.as_retriever(similarity_top_k=5)
        
        # Retrieve results
        results = retriever.retrieve(query)
        
        # Format results
        if not results:
            return "No relevant documents found for the query."
            
        formatted_results = []
        for i, r in enumerate(results, 1):
            content = r.node.get_content()
            score = r.score if hasattr(r, 'score') else 'N/A'
            formatted_results.append(f"[Chunk {i} - Score: {score}]\n{content}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in pg_retriever_tool: {str(e)}")
        return f"Error retrieving documents: {str(e)}"
