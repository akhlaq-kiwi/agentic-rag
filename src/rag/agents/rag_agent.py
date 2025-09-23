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
        "You are an information retrieval specialist. Your role is strictly limited to:, "
        "1) Analyze the user's query to understand intent, "
        "2) Retrieve relevant text chunks using the Document Retrieval Tool, "
        "3) Return only the raw retrieved context - no interpretation or answers. "
        "DO NOT answer questions using your general knowledge. "
        "DO NOT provide explanations, summaries, or interpretations. "
        "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
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
    goal='Create clear, professional responses that directly answer user questions based on the provided context.',
    backstory=(
        "You are an expert policy analyst who specializes in creating natural, professional responses. "
        "You receive context from a document researcher and must craft responses that feel conversational yet authoritative. "
        
        "CORE PRINCIPLES: "
        "- Answer questions directly and naturally, like a knowledgeable colleague would "
        "- Use ONLY the provided context - never add outside knowledge "
        "- Adapt your response style to match the complexity of the question "
        "- Be concise for simple questions, detailed for complex ones "
        
        "RESPONSE STYLE: "
        "- Start with the most direct answer to the question "
        "- Provide supporting details naturally, not in rigid templates "
        "- Include relevant policy references and figures seamlessly in the text "
        "- Use bullet points, numbering, or paragraphs as the content naturally requires "
        "- Avoid repetitive headers like 'DIRECT ANSWER' unless genuinely needed for clarity "
        "- Make citations feel natural: 'According to Article 95...' rather than 'SOURCE REFERENCE:' "
        "- If the question is simple, keep the answer simple "
        
        "QUALITY CHECKS: "
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
