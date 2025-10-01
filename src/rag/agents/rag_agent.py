# src/agents/rag_agents.py - OPTIMIZED VERSION
import os
from crewai import Agent
from llama_index.llms.ollama import Ollama as BaseOllama
from src.config import OLLAMA_BASE_URL, LLM
from src.rag.tools.rag_tools import pg_retriever_tool
from src.prompts.prompt_manager import get_prompt


# Store model name mapping globally to avoid Pydantic attribute issues
_MODEL_NAME_MAP = {}

# Custom Ollama wrapper to handle LiteLLM model name format
class Ollama(BaseOllama):
    """
    Custom Ollama LLM that handles both LiteLLM format (ollama/model) 
    and native Ollama format (model) for API calls.
    
    This wrapper keeps the full "ollama/model" name for CrewAI/LiteLLM,
    but uses just "model" for actual Ollama API calls.
    """
    def __init__(self, *args, **kwargs):
        # Store the original model name before calling super().__init__
        model_name = kwargs.get('model', args[0] if args else None)
        
        # If model has "ollama/" prefix, strip it for Ollama API initialization
        if model_name and model_name.startswith('ollama/'):
            stripped_model = model_name.replace('ollama/', '')
            if 'model' in kwargs:
                kwargs['model'] = stripped_model
            elif args:
                args = (stripped_model,) + args[1:]
        
        super().__init__(*args, **kwargs)
        
        # Store mapping: id(self) -> original model name
        _MODEL_NAME_MAP[id(self)] = model_name
        
        # Override model attribute to use original name (for CrewAI/LiteLLM)
        object.__setattr__(self, 'model', model_name)
    
    @property
    def metadata(self):
        """Override metadata to use original model name"""
        meta = super().metadata
        original_model = _MODEL_NAME_MAP.get(id(self), self.model)
        meta.model_name = original_model
        return meta
    
    def _get_stripped_model(self):
        """Return stripped model name for Ollama API calls"""
        original_model = _MODEL_NAME_MAP.get(id(self), self.model)
        if original_model and original_model.startswith('ollama/'):
            return original_model.replace('ollama/', '')
        return original_model
    
    def chat(self, *args, **kwargs):
        """Override chat to use stripped model name"""
        # Temporarily swap model name for API call
        original = self.model
        object.__setattr__(self, 'model', self._get_stripped_model())
        try:
            return super().chat(*args, **kwargs)
        finally:
            object.__setattr__(self, 'model', original)
    
    def complete(self, *args, **kwargs):
        """Override complete to use stripped model name"""
        # Temporarily swap model name for API call
        original = self.model
        object.__setattr__(self, 'model', self._get_stripped_model())
        try:
            return super().complete(*args, **kwargs)
        finally:
            object.__setattr__(self, 'model', original)


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
# Use custom Ollama wrapper that handles both LiteLLM and native Ollama formats
_llm = Ollama(
    model=f"ollama/{LLM}",  # LiteLLM format: "ollama/model-name"
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    timeout=300,
    verbose=True,  # Enable verbose logging for debugging
    request_timeout=300.0,
    context_window=131072,  # Set explicitly to avoid model.show() call
    num_ctx=131072,
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
