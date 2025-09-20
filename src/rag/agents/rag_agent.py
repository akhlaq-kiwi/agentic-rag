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
    # Initialize LLM with Ollama - zero temperature for deterministic responses
    llm = Ollama(
        model=LLM,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=0.0  # No creativity - strict adherence to context
    )
    
    # Set base URL for Ollama (without /api suffix as LiteLLM adds it)
    os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

    query_router = Agent(
        name="Query Router",
        role="Route and classify user queries",
        goal="Determine if a query is a greeting, requires document retrieval, or is out of scope",
        backstory="""You are a query classification expert. Your job is to:
        1. Identify greetings (hello, hi, good morning, how are you, etc.) and respond warmly
        2. Identify questions that need document retrieval (factual questions, policy questions, etc.)
        3. Classify the query type and route appropriately
        
        For greetings, respond with: "GREETING: [friendly greeting response]"
        For document questions, respond with: "RETRIEVE: [original query]"
        For unclear queries, respond with: "CLARIFY: Please ask a specific question about the documents."
        """,
        llm=llm,
        verbose=True
    )

    # Create a simple greeting handler without tools
    greeting_handler = Agent(
        name="Greeting Handler",
        role="Handle greetings and simple interactions",
        goal="Respond to greetings and clarification requests without using any tools",
        backstory="""You are a friendly assistant that handles social interactions and routing decisions.
        
        Your job is to:
        1. Check the routing decision from the query router
        2. For GREETING or CLARIFY routing decisions, respond appropriately without using any tools
        3. For RETRIEVE routing decisions, pass the query to the document retrieval system
        
        You have no tools available and should not attempt to use any.""",
        tools=[],  # No tools for greeting handler
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create a dedicated document retriever with tools
    document_retriever = Agent(
        name="Document Retriever",
        role="Retrieve relevant documents from the database",
        goal="Search and retrieve relevant document chunks using the pg_retriever_tool",
        backstory="""You are a document retrieval specialist. Your only job is to search for and retrieve relevant document chunks when given a specific query to search for.""",
        tools=[pg_retriever_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    answer_agent = Agent(
        name="Answer Generator",
        role="Generate responses strictly based on retrieved documents",
        goal="Provide document-based answers with zero creativity and strict source adherence",
        backstory="""You are a document-based assistant with ZERO creativity and STRICT adherence to retrieved content.

        CRITICAL RULES - NO EXCEPTIONS:
        1. For GREETING queries: Use the exact greeting from the routing decision
        2. For CLARIFY queries: Use the exact clarification message from routing
        3. For RETRIEVE queries: ONLY use information explicitly stated in the retrieved document chunks
        
        DOCUMENT-BASED RESPONSE REQUIREMENTS:
        - ALWAYS check the retrieved document chunks first
        - If ANY relevant information exists in the chunks, use it to answer
        - Quote directly from retrieved documents when possible
        - Only say "I don't have enough information" if the retrieved chunks are completely empty or irrelevant
        - Always cite the exact source document name and page number
        - Never paraphrase beyond what's explicitly stated
        - Never make logical inferences beyond what's written
        - If chunks contain partial information, provide what's available and cite sources
        
        FORBIDDEN BEHAVIORS:
        - Adding context not in the documents
        - Making assumptions or inferences
        - Using general knowledge
        - Creative interpretation
        - Expanding on document content
        """,
        llm=llm,
        verbose=True
    )

    return query_router, greeting_handler, document_retriever, answer_agent
