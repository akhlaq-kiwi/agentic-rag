# src/agents/rag_agents.py
from crewai import Agent
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
from ..tools.rag_tools import pg_retriever_tool
import os

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

def create_rag_agents():
    # CrewAI uses LiteLLM under the hood
    # For Ollama, use the format: ollama/model_name
    llm = "ollama/gemma:2b"
    
    # Set base URL for Ollama (without /api suffix as LiteLLM adds it)
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

    retriever_agent = Agent(
        name="Retriever",
        role="Fetch relevant context",
        goal="Retrieve the most relevant chunks for a query from pgvector",
        backstory="You are a retrieval agent that understands embeddings and hybrid search.",
        tools=[pg_retriever_tool],
        llm=llm,
        verbose=True
    )

    rag_agent = Agent(
        name="RAG Orchestrator",
        role="Build RAG context",
        goal="Format retrieved docs into a coherent context for LLM answering",
        backstory="You organize raw chunks into a concise context with metadata like page, section, etc.",
        llm=llm,
        verbose=True
    )

    llm_agent = Agent(
        name="LLM Generator",
        role="Generate answers",
        goal="Answer user queries using LLaMA 3 with given context",
        backstory="You are an expert assistant providing well-structured answers based on the retrieved context.",
        llm=llm,
        verbose=True
    )

    return retriever_agent, rag_agent, llm_agent

