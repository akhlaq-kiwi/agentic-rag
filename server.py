# main.py
import logging
import os
import json
import time
import asyncio
from datetime import datetime

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config early to get Phoenix settings
from src.config import OLLAMA_BASE_URL, PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

# Phoenix Tracing Setup - MUST BE BEFORE OTHER IMPORTS
phoenix_base_url = PHOENIX_COLLECTOR_ENDPOINT or "http://phoenix:6006"

# Phoenix expects the base URL (UI port), it will automatically use port 4318 for OTLP
# Set the environment variable for Phoenix to discover
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_base_url

try:
    from phoenix.otel import register
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    import requests
    
    # Import instrumentation for LlamaIndex, OpenAI, and LiteLLM
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    
    # Try to import LiteLLM instrumentor (CrewAI uses LiteLLM internally)
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        has_litellm_instrumentor = True
    except ImportError:
        has_litellm_instrumentor = False
        logger.warning("LiteLLM instrumentor not found - CrewAI LLM calls may not be fully traced")
    
    # Test Phoenix UI connectivity
    try:
        response = requests.get(f"{phoenix_base_url}/healthz", timeout=5)
        logger.info(f"✅ Phoenix UI connectivity: {response.status_code}")
    except Exception as conn_e:
        logger.warning(f"⚠️ Phoenix UI connectivity test failed: {conn_e}")
    
    # Determine OTLP endpoint - use gRPC (port 4317) for better stability
    # Phoenix HTTP endpoint (4318) has connection reset issues
    otlp_endpoint = phoenix_base_url.replace(":6006", ":4317").replace(":4318", ":4317")
    
    # Register with Phoenix using gRPC endpoint
    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME or "agentic-rag",
        endpoint=otlp_endpoint,  # Use gRPC endpoint (port 4317)
        resource=Resource.create({SERVICE_NAME: "agentic-rag-api"}),
        auto_instrument=True  # This automatically instruments CrewAI and other libraries
    )
    
    # Manually instrument LlamaIndex for detailed LLM traces
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("   ✅ LlamaIndex instrumented")
    
    # Instrument OpenAI-compatible clients (Ollama uses OpenAI API format)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("   ✅ OpenAI instrumented")
    
    # Instrument LiteLLM if available (CrewAI uses LiteLLM)
    if has_litellm_instrumentor:
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("   ✅ LiteLLM instrumented (CrewAI support)")
    
    logger.info(f"✅ Arize Phoenix tracing initialized")
    logger.info(f"   Phoenix UI: {phoenix_base_url}")
    logger.info(f"   OTLP Endpoint: {otlp_endpoint} (gRPC)")
    logger.info(f"   Project: {PHOENIX_PROJECT_NAME or 'agentic-rag'}")
    
    # Create a test span to verify tracing is working
    from opentelemetry import trace as otel_trace
    test_tracer = otel_trace.get_tracer(__name__)
    with test_tracer.start_as_current_span("phoenix_startup_verification") as span:
        span.set_attribute("service.name", "agentic-rag-api")
        span.set_attribute("test.type", "startup")
        span.add_event("Phoenix tracing initialized")
        logger.info(f"✅ Test span created - check Phoenix UI at {phoenix_base_url.replace(':4317', ':6006').replace(':4318', ':6006')}")
    
except ImportError as e:
    logger.warning(f"Phoenix/OpenInference module not found: {e}")
    logger.warning("Install with: pip install arize-phoenix-otel openinference-instrumentation-llama-index openinference-instrumentation-openai openinference-instrumentation-litellm")
    phoenix_base_url = PHOENIX_COLLECTOR_ENDPOINT or "http://phoenix:6006"
    otlp_endpoint = None
    has_litellm_instrumentor = False
except Exception as e:
    logger.warning(f"⚠️ Could not initialize Arize Phoenix tracing: {e}")
    logger.exception(e)  # Log full traceback for debugging
    phoenix_base_url = PHOENIX_COLLECTOR_ENDPOINT or "http://phoenix:6006"
    otlp_endpoint = None

# Now import other modules after tracing is set up
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.rag.crew import run_rag_query, run_rag_query_structured
from src.rag.memory.redis_memory import conversation_memory

# Import OpenTelemetry tracing tools for manual spans
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

app = FastAPI(title="Agentic RAG API", description="OpenAI-compatible RAG API for OpenWebUI")

# OpenAI-compatible models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "agentic-rag"

# Initialize agents and crew on startup
rag_crew = None

@app.on_event("startup")
async def startup_event():
    global rag_crew
    try:
        logger.info("Initializing RAG system...")
        # Set rag_crew to True to indicate system is ready
        rag_crew = True
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

@app.get("/")
def root():
    return {"message": "RAG API is running"}

# OpenWebUI-compatible endpoints
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available models for OpenWebUI"""
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id="agentic-rag",
                created=int(time.time()),
                owned_by="agentic-rag"
            ).dict(),
            ModelInfo(
                id="rag-gemma",
                created=int(time.time()),
                owned_by="agentic-rag"
            ).dict()
        ]
    }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with grounding metadata"""
    print(request.messages)
    # Extract the last user message as the query
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    
    # Use OpenWebUI's chat_id as session_id if present in request
    print(f"Received query for API: {user_message}")

    # Use structured response with grounding
    rag_response = run_rag_query_structured(user_message)
    
    # Format response with sources appended
    formatted_response = rag_response.format_with_sources()
    
    # Format the response to be compatible with the OpenAI API standard
    response = {
        "id": "chatcmpl-123", # Dummy ID
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": formatted_response,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0, # You can implement token counting if needed
            "completion_tokens": 0,
            "total_tokens": 0
        },
        # Add grounding metadata as custom field
        "grounding": {
            "sources": [
                {
                    "file_name": s.file_name,
                    "page_number": s.page_number,
                    "relevance_score": s.relevance_score
                }
                for s in rag_response.grounding.sources
            ],
            "total_chunks": rag_response.grounding.total_chunks
        }
    }
    return response

async def generate_stream_response(content: str, model: str):
    """Generate streaming response for chat completions"""
    # Send content in chunks for better streaming
    chunk_size = 10  # words per chunk
    words = content.split()
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_content += " "
            
        chunk = {
            "id": f"chatcmpl-{int(time.time())}-{i}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # Small delay for streaming effect
    
    # Final chunk
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}-final",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# Conversation memory endpoints
@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = conversation_memory.get_history(session_id)
        return {
            "session_id": session_id,
            "conversation_count": len(history),
            "history": history
        }
    except Exception as e:
        return {"error": str(e), "history": []}

@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        # Simple way to clear - we'll just let Redis expire handle it
        return {"message": f"Conversation history for session {session_id} will expire automatically"}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    phoenix_status = "unknown"
    try:
        import requests
        response = requests.get(f"{phoenix_base_url}/healthz", timeout=3)
        phoenix_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        phoenix_status = "disconnected"
    
    return {
        "status": "healthy",
        "rag_initialized": rag_crew is not None,
        "phoenix_tracing": {
            "ui_endpoint": phoenix_base_url,
            "status": phoenix_status,
            "project": PHOENIX_PROJECT_NAME or "agentic-rag"
        },
        "timestamp": datetime.now().isoformat()
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
