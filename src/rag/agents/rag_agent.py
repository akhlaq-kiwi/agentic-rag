# src/agents/rag_agents.py - OPTIMIZED VERSION
from crewai import Agent
from llama_index.llms.ollama import Ollama
from src.config import OLLAMA_BASE_URL, LLM
from src.rag.tools.rag_tools import pg_retriever_tool
import os

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

def create_rag_agents():
    """Create optimized RAG agents - reduced from 4 to 2 agents for speed."""
    
    # Initialize LLM with faster settings
    llm = Ollama(
        model=f"ollama/{LLM}",
        base_url=OLLAMA_BASE_URL,
        request_timeout=60.0,  # Reduced from 120s
        temperature=0.0
    )
    
    # AGENT 1: Smart Retriever (combines routing + retrieval)
    smart_retriever = Agent(
        name="Smart Retriever",
        role="Intelligent document retrieval with built-in query handling",
        goal="Handle all queries efficiently - greetings directly, document questions via retrieval",
        backstory="""You are an efficient assistant that handles queries intelligently:

        FOR SIMPLE GREETINGS (hi, hello, good morning, how are you):
        - Respond directly with a friendly greeting
        - DO NOT use any tools
        - Example: "Hello! I'm here to help you with document questions."

        FOR DOCUMENT QUESTIONS (policies, procedures, specific information):
        - Use pg_retriever_tool to search for relevant information
        - Retrieve the most relevant document chunks
        
        FOR UNCLEAR QUERIES:
        - Ask for clarification politely
        - DO NOT use tools for clarification requests
        
        Be smart about when to use tools vs. direct responses.""",
        tools=[pg_retriever_tool],
        llm=llm,
        verbose=False,  # Reduced verbosity for speed
        allow_delegation=False
    )

    # AGENT 2: Answer Generator (streamlined)
    answer_generator = Agent(
        name="Answer Generator",
        role="Generate accurate responses from retrieved documents",
        goal="Provide precise, document-based answers with source citations",
        backstory="""You generate responses based on retrieved documents with these rules:

        IF NO RETRIEVAL WAS PERFORMED (greetings, clarifications):
        - Use the direct response from the Smart Retriever
        
        IF DOCUMENTS WERE RETRIEVED:
        - Extract relevant information from the retrieved chunks
        - Quote directly from documents when possible
        - Always cite source: "According to [Document], page [X]: [information]"
        - If no relevant info found, say "I don't have information about that in the available documents."
        
        STRICT REQUIREMENTS:
        - Only use information explicitly stated in retrieved documents
        - Never add external knowledge or assumptions
        - Keep responses concise and factual
        - Always provide source citations""",
        llm=llm,
        verbose=False,  # Reduced verbosity for speed
        allow_delegation=False
    )

    return smart_retriever, answer_generator
