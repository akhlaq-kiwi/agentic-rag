# src/agents/rag_agents.py - OPTIMIZED VERSION
import os
from crewai import Agent
from llama_index.llms.ollama import Ollama
from src.config import OLLAMA_BASE_URL, LLM
from src.rag.tools.rag_tools import pg_retriever_tool
from src.prompts.prompt_manager import get_prompt


# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
insight_synthesizer_backstory_prompt = ""
try:
    insight_synthesizer_backstory_prompt = get_prompt("insight_synthesizer_backstory")
    insight_synthesizer_backstory_prompt = insight_synthesizer_backstory_prompt.format().messages[0]['content']
except Exception as e:
    print(e)
    insight_synthesizer_backstory_prompt = "Response as human readable answer based on the context provided"
    
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
        "2) If greeting -> return 'greeting'. "
        "3) If chat_history -> return 'chat_history'. "
        "4) If question -> return 'question'. "
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
    max_iter=1,
)

# AGENT 2: Answer Generator (streamlined)
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Create clear, professional responses that directly answer user questions with proper source citations.',
    backstory=(
        insight_synthesizer_backstory_prompt
    ),
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    tools=[]
)
