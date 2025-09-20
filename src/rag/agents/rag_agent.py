# src/agents/rag_agents.py
from crewai import Agent
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL, LLM
from src.rag.tools.rag_tools import pg_retriever_tool
import os

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

def create_rag_agents():
    # CrewAI uses LiteLLM under the hood
    # For Ollama, use the format: ollama/model_name
    llm = Ollama(
        model=LLM,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0
    )
    
    # Set base URL for Ollama (without /api suffix as LiteLLM adds it)
    os.environ["OLLAMA_API_BASE"] = "http://ollama:11434"

    retriever_agent = Agent(
        name="Document Retriever",
        role="Retrieve relevant documents",
        goal="Find and retrieve the most relevant document chunks for the user query",
        backstory="You are an expert document retrieval specialist with access to a comprehensive document database.",
        tools=[pg_retriever_tool],
        llm=llm,
        verbose=True
    )

    answer_agent = Agent(
        name="Answer Generator",
        role="Generate comprehensive answers",
        goal="Answer user queries directly using the retrieved document chunks as context",
        backstory="""You are an expert assistant that provides accurate, comprehensive answers based on retrieved document context. 
        Your task is to:
        1. Analyze the retrieved document chunks
        2. Extract relevant information to answer the user's question
        3. Provide a clear, well-structured response
        4. Cite sources when appropriate
        5. If the retrieved context doesn't contain enough information, clearly state what is missing""",
        llm=llm,
        verbose=True
    )

    return retriever_agent, answer_agent
