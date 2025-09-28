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
phoenix_endpoint = PHOENIX_COLLECTOR_ENDPOINT or "http://phoenix:6006"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

try:
    from phoenix.otel import register
    import requests
    
    # Test Phoenix connectivity first
    try:
        response = requests.get(f"{phoenix_endpoint}/healthz", timeout=5)
        logger.info(f"Phoenix connectivity test: {response.status_code}")
    except Exception as conn_e:
        logger.warning(f"Phoenix connectivity test failed: {conn_e}")
    
    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME or "default",
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=True  # This automatically instruments CrewAI and other libraries
    )
    logger.info(f"✅ Arize Phoenix tracing successfully initialized at {phoenix_endpoint}")
except ImportError as e:
    logger.warning(f"Phoenix module not found: {e}. Install with: pip install arize-phoenix")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize Arize Phoenix tracing: {e}")

# Now import other modules after tracing is set up
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.rag.crew import run_rag_query
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
    """OpenAI-compatible chat completions endpoint"""
    print(request.messages)
    # Extract the last user message as the query
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    
    # Use OpenWebUI's chat_id as session_id if present in request
    print(f"Received query for API: {user_message}")

    result = run_rag_query(user_message)
    
    # Format the response to be compatible with the OpenAI API standard
    response = {
        "id": "chatcmpl-123", # Dummy ID
        "object": "chat.completion",
        "created": 1677652288, # Dummy timestamp
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": str(result), # Ensure the result is a string
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0, # You can implement token counting if needed
            "completion_tokens": 0,
            "total_tokens": 0
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
        response = requests.get(f"{phoenix_endpoint}/healthz", timeout=3)
        phoenix_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        phoenix_status = "disconnected"
    
    return {
        "status": "healthy",
        "rag_initialized": rag_crew is not None,
        "phoenix_tracing": {
            "endpoint": phoenix_endpoint,
            "status": phoenix_status
        },
        "timestamp": datetime.now().isoformat()
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
