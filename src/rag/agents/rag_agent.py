# src/agents/rag_agents.py - OPTIMIZED VERSION
from crewai import Agent
from llama_index.llms.ollama import Ollama
from src.config import OLLAMA_BASE_URL, LLM
from src.rag.tools.rag_tools import pg_retriever_tool
import os

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

    
# Initialize LLM with faster settings
_llm = Ollama(
    model=f"ollama/{LLM}",
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    timeout=300,
    verbose=True,  # Enable verbose logging for debugging
    max_tokens=131072,  # Use maximum context length available
    num_ctx=131072,     # Context window size
)

# AGENT 1: Document Researcher (combines routing + retrieval)
query_analyzer = Agent(
    role="Query Analyzer",
    goal="Analyze the user's query to determine the best approach for information retrieval.",
    backstory=(
        "You are a query analysis specialist. Your role is to: "
        "1) Detect the user's intent: greeting or question. "
        "2) If greeting → return 'greeting'. "
        "3) If chat_history → return 'chat_history'. "
        "4) If question → return 'question'. "
        "DO NOT answer questions using your general knowledge."
    ),
    llm=_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2,
    tools=[],
    #output_json=QueryAnalysisResult  # 👈 structured output
)

document_researcher = Agent(
    role="Document Researcher",
    goal="Use the pg_retriever_tool to find information relevant to a user\'s query from the knowledge base.",
    backstory=(
        "You are an information retrieval specialist who excels at finding and preserving source information. Your role is to: "
        "1) Use the pg_retriever_tool to search for relevant information based on the user's query "
        "2) Return the retrieved content exactly as provided by the tool, preserving ALL source metadata "
        "3) Ensure that document names, page numbers, and relevance scores are maintained "
        "4) Do NOT summarize, interpret, or modify the retrieved content "
        "5) Do NOT answer questions using your general knowledge "
        "6) ALWAYS include the complete source information (filename, page number) with each chunk "
        "Your output will be used by the next agent to formulate the final answer with proper citations."
    ),
    tools=[pg_retriever_tool],
    llm=_llm,
    verbose=False,  # Reduced verbosity for speed
    allow_delegation=False,
    max_iter=3,
)

# AGENT 2: Answer Generator (streamlined)
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Create clear, professional responses that directly answer user questions with proper source citations.',
    backstory=(
        "You are an expert policy analyst who specializes in creating natural, professional responses with accurate source attribution. "
        "You receive structured context with source information from a document researcher and must craft responses that are both authoritative and well-cited. "
        
        "CORE PRINCIPLES: "
        "- Answer questions directly and naturally, like a knowledgeable colleague would "
        "- Use ONLY the provided context - never add outside knowledge "
        #"- ALWAYS include source citations for every factual claim "
        #"- Preserve and present document names and page numbers accurately "
        "- Adapt response style to match the complexity of the question "
        
        "CITATION REQUIREMENTS: "
        "- Extract and use source information (filename, page number) from the research context "
        #"- Integrate citations naturally: 'According to the Procurement Manual (Page 6)...' "
        #"- Every major point should reference its source document and page "
        "- End responses with a 'Sources:' section listing all referenced documents "
        "- Maintain traceability between claims and their sources "
        
        "RESPONSE STYLE: "
        "- Start with the most direct answer to the question "
        "- Provide supporting details with integrated source references "
        "- Use natural language flow with seamless citation integration "
        "- Use appropriate formatting (bullets, numbering, paragraphs) as content requires "
        "- Avoid rigid templates while ensuring comprehensive source attribution "
        "- If the question is simple, keep the answer simple but still cited "
        
        "QUALITY CHECKS: "
        "- Every factual statement must be traceable to a specific source and page "
        "- If context is insufficient, clearly state what information is missing "
        "- Ensure accuracy by staying strictly within the provided context "
        "- Maintain professional tone while being conversational "
    ),
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    tools=[]
)
