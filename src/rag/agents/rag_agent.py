# src/agents/rag_agents.py
from crewai import Agent, Task
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM
from ..tools.base_tool import PgRetrieverTool

def create_rag_agents():
    # --- Embedding + Vector Store ---
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    vector_store = PGVectorStore.from_params(
        database=DATABASE_NAME,
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
        table_name='documents',
        embed_dim=768,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    retriever_tool = PgRetrieverTool(index.as_retriever(similarity_top_k=5))

    retriever_agent = Agent(
        name="Retriever",
        role="Fetch relevant context",
        goal="Retrieve the most relevant chunks for a query from pgvector",
        backstory="You are a retrieval agent that understands embeddings and hybrid search.",
        tools=[retriever_tool],
    )

    rag_agent = Agent(
        name="RAG Orchestrator",
        role="Build RAG context",
        goal="Format retrieved docs into a coherent context for LLM answering",
        backstory="You organize raw chunks into a concise context with metadata like page, section, etc.",
    )

    llm_agent = Agent(
        name="LLM Generator",
        role="Generate answers",
        goal="Answer user queries using LLaMA 3 with given context",
        backstory="You are an expert assistant providing well-structured answers.",
        tools=[Ollama(model="llama3:8b", base_url="http://localhost:11434")],
    )

    return retriever_agent, rag_agent, llm_agent

