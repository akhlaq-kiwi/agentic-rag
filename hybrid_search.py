from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from typing import List, Dict, Any
import json

# Import database configuration
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, OLLAMA_BASE_URL

# Configure settings
Settings.llm = Ollama(model="llama3:8b", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL,
    request_timeout=60.0
)

class VectorSearch:
    def __init__(self):
        """Initialize the hybrid search engine with LlamaIndex components"""
        self.vector_store = PGVectorStore.from_params(
            database=DATABASE_NAME,
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            table_name="vectors",
            embed_dim=768,  # Dimension for nomic-embed-text
            hybrid_search=True,
            text_search_config="english"
        )
        
        # Create storage context and index
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = VectorStoreIndex([], storage_context=storage_context)
        
        # Create retriever with hybrid search
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
            vector_store_query_mode="hybrid"
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using PostgreSQL's native hybrid search
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text, metadata, and scores
        """
        try:
            # Update top_k for the retriever
            self.retriever.similarity_top_k = top_k
            
            # Get nodes using the retriever
            nodes = self.retriever.retrieve(query)
            
            print(nodes)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

def print_results(results: List[Dict[str, Any]]):
    """Print search results in a formatted way"""
    if not results:
        print("No results found.")
        return
    
    print(f"\n🔍 Found {len(results)} results:")
    print("=" * 120)
    
    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        text = result.get("text", "")[:200] + "..." if result.get("text") else "No text"
        metadata = result.get("metadata", {})
        
        print(f"\n{i}. {text}")
        print(f"   Score: {score:.4f}")
        if metadata:
            print(f"   Metadata: {json.dumps(metadata, indent=2)}")
        print("-" * 120)

if __name__ == "__main__":
    # Initialize the search engine
    print("Initializing hybrid search engine...")
    searcher = VectorSearch()
    
    # Example search
    query = "shall commit to start working for their employer with"
    print(f"\nSearching for: '{query}'")
    
    # Perform hybrid search
    results = searcher.search(query=query, top_k=5)
    
    # Print results
    #print_results(results)