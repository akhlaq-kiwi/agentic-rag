import os
import requests
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from crewai.tools import tool
from typing import Dict, Union, Any
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, OLLAMA_BASE_URL

os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

def warm_up_ollama(base_url: str, model_name: str) -> bool:
    """Pre-warm Ollama model to avoid cold start delays"""
    try:
        # Send a small embedding request to warm up the model
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not warm up Ollama model: {e}")
        return False

@tool("Document Retrieval Tool")  
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """Retrieves relevant context from a collection of policy and standards documents. Use this tool to search for information in policy documents, manuals, and standards."""
    try:
        # Debug: Log what we actually receive
        print(f"DEBUG: Tool received query parameter: {repr(query)}")
        
        # Handle CrewAI's parameter passing - extract actual query from different formats
        search_query = None
        
        if isinstance(query, str):
            search_query = query
        elif isinstance(query, dict):
            # CrewAI passes: {"description": "actual query", "type": "str"}
            if "description" in query:
                search_query = query["description"]
            elif "query" in query:
                search_query = query["query"]
            else:
                # Fallback: convert dict to string
                search_query = str(query)
        else:
            search_query = str(query)
        
        # Validate we have a proper query string
        if not search_query or not isinstance(search_query, str):
            return "Error: No valid search query provided."
        
        # Check if we got a placeholder description instead of real query
        if search_query in ["The search query to find relevant documents", ""]:
            return "Error: Tool received schema placeholder instead of actual query."
        
        search_query = search_query.strip()
        print(f"DEBUG: Extracted search query: {repr(search_query)}")
        
        if not DATABASE_HOST:
            return "Error: DATABASE_URL environment variable not set."

        db_url_parts = urlparse(DATABASE_HOST)

        # Debug print parsed connection details (password masked)
        print(f"DEBUG: Parsed Postgres connection details - "
              f"host: {DATABASE_HOST}, "
              f"port: {DATABASE_PORT}, "
              f"database: {DATABASE_NAME}, "
              f"user: {DATABASE_USER}, "
              f"password: {'******' if DATABASE_PASSWORD else None}")
        
        # Initialize the vector store with connection details and hybrid search enabled
        # Check for contextual table first, fallback to regular table
        contextual_table = "vectors"
        regular_table = "vectors"
        
        # Try contextual table first
        try:
            vector_store = PGVectorStore.from_params(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD,
                table_name=contextual_table,  # Use contextual table if available
                embed_dim=768, # Dimension for nomic-embed-text
                hybrid_search=True,
                text_search_config="english"
            )
            print(f"DEBUG: Using contextual table: {contextual_table}")
        except Exception as e:
            print(f"DEBUG: Contextual table not available, using regular table: {e}")
            vector_store = PGVectorStore.from_params(
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                database=DATABASE_NAME,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD,
                table_name=regular_table,  # Fallback to regular table
                embed_dim=768, # Dimension for nomic-embed-text
                hybrid_search=True,
                text_search_config="english"
            )

        # Pre-warm the Ollama model to avoid cold start delays
        warm_up_ollama(OLLAMA_BASE_URL, "nomic-embed-text:v1.5")
        
        # Configure Ollama LLM and embedding models
        llm = Ollama(
            model="gemma:2b",
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0
        )
        
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0  # Increased timeout for slower connections
        )

        # Set global LlamaIndex settings to use Ollama
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Create a LlamaIndex VectorStoreIndex object from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Create a query engine with hybrid search mode - increased retrieval for maximum tokens
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=2,  # Increased from 5 to utilize maximum token capacity
            sparse_top_k=2       # Increased from 5 to get more comprehensive context
        )
        
        # Query using hybrid search (combines vector + text search)
        print("Querying using hybrid search...", search_query)
        response = query_engine.query(search_query)
        print("Response:", response)
        retrieved_nodes = response.source_nodes
        
        if not retrieved_nodes:
            return "No relevant documents found for this query."
        
        # Format the retrieved context with source metadata and contextual information
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()
            
            # Extract source file information from metadata
            source_info = "Unknown source"
            context_info = ""
            page_info = ""
            
            if hasattr(node, 'metadata') and node.metadata:
                # File information
                file_name = node.metadata.get('source_file', node.metadata.get('file_name', 'Unknown file'))
                file_path = node.metadata.get('file_path', '')
                if file_path:
                    source_info = f"Source: {os.path.basename(file_path)}"
                else:
                    source_info = f"Source: {file_name}"
                
                # Contextual information (if available)
                context = node.metadata.get('context', '')
                if context:
                    context_info = f"\nContext: {context}"
                
                # Page number information (if available)
                page_num = node.metadata.get('page_number', '')
                if page_num:
                    page_info = f" (Page {page_num})"
            
            formatted_chunk = f"**Document Chunk {i}**\n{source_info}{page_info}{context_info}\n\nContent:\n{content}"
            formatted_chunks.append(formatted_chunk)
        
        context = "\n\n" + "="*50 + "\n\n".join(formatted_chunks)
        
        return context
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"