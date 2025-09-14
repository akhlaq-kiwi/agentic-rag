from crewai.tools import tool
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
vector_store = PGVectorStore.from_params(
        database=DATABASE_NAME,
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
        table_name='documents',
        embed_dim=DIM,
    )
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

@tool
def pg_retriever_tool(query: str):
    """Retrieve relevant chunks from pgvector using semantic similarity."""
    retriever = index.as_retriever(similarity_top_k=5)
    results = retriever.retrieve(query)
    return [r.node.get_content() for r in results]
