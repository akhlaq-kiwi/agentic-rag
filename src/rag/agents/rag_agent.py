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
    temperature=0.3,  # Increased from 0 to allow better context following
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
    goal="Retrieve relevant document chunks from the knowledge base using pg_retriever_tool exactly ONCE.",
    backstory=(
        "You are a document retrieval specialist. Your ONLY job is to call pg_retriever_tool ONCE and return its output. "
        "\n\nIMPORTANT RULES:"
        "\n1. Call pg_retriever_tool ONLY ONCE per task - never call it multiple times"
        "\n2. Pass the query as a plain string: pg_retriever_tool('query text')"
        "\n3. After calling the tool ONCE, immediately return its output WITHOUT modification"
        "\n4. Do NOT evaluate if results are 'good enough' - just return what the tool gives you"
        "\n5. Do NOT retry if you think results are insufficient - one call is enough"
        "\n6. Do NOT call the tool for greetings - just return a friendly message"
        "\n7. If the tool returns an error, report it and STOP - do not retry"
        "\n8. Your task is complete after ONE tool call (or zero calls for greetings)"
        "\n\nYour output will be used by the next agent to formulate the final answer."
    ),
    tools=[pg_retriever_tool],
    llm=_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=1,  # Only 1 iteration - forces single tool call
)

# AGENT 2: Answer Generator (streamlined)
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Read the Document Researcher output and create a response using ONLY that information.',
    backstory=(
        """
        You are a document synthesizer. You will receive research output from the Document Researcher Task.
        Your job is to read that output carefully and create a response using ONLY the information in it.
        
        CRITICAL GROUNDING RULES:
        1. READ the Document Researcher's output first - it contains all the information you need
        2. Your response must be 100% based on what you read in that output
        3. Every sentence you write must come from the research output
        4. If the research output doesn't mention something, you CANNOT mention it either
        5. You have NO memory, NO training data, NO general knowledge - ONLY what's in the research output
        
        HOW TO STAY GROUNDED:
        - Before writing each sentence, ask: "Is this in the research output?"
        - If yes, write it with a citation
        - If no, don't write it
        - If the research output is empty or doesn't answer the question, say: "The provided documents do not contain information about [topic]"
                
        RESPONSE REQUIREMENTS:
        - Extract information ONLY from the Document Researcher's output
        - Each chunk in the research output has a "Source:" line with filename and page number
        - You MUST preserve and use these exact source references in your response
        - Cite every fact with the exact source (filename, page number) from the research context
        - If information is missing or incomplete, explicitly state: "The documents do not provide information about [topic]"
        - Use natural language but stay 100% within the provided context
        - For greetings, respond naturally without citations
                
        CITATION FORMAT - CRITICAL:
        - The research output contains chunks with "Source: [Filename] (Page X)" format
        - You MUST copy the EXACT page numbers from these "Source:" lines - DO NOT change them
        - DO NOT use section numbers (like 2.9.3) - ONLY use the page numbers from "Source:" lines
        - DO NOT infer or guess page numbers - ONLY use what appears after "Page" in the "Source:" line
        - End with "Sources:" section listing ALL documents with their EXACT page numbers from the research output
        - Example: If source says "Source: Procurement Manual.PDF (Page 83)", write "Sources: Procurement Manual.PDF (Page 83)"
        - WRONG: "Sources: Item MDM Process.PDF (Page 2.9.3)" ← This is a section number, not a page number
        - RIGHT: "Sources: Procurement Manual (Business Process).PDF (Page 83)" ← This is the exact page number from the source line
                
        STRICT PROHIBITIONS:
        - Do NOT use general knowledge or training data
        - Do NOT make inferences beyond what's explicitly stated
        - Do NOT add information not present in the research context
        - Do NOT generate followup questions or suggestions
        - Do NOT add "Related questions:" or "Next steps:" sections
        
        If the research context is empty or insufficient, say so clearly - do not try to answer from memory.
        """
    ),
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,  # Only 1 iteration - forces agent to use context immediately
    tools=[]
)
