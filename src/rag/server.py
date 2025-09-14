# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from src.rag.agents.rag_agent import create_rag_agents
from src.rag.tasks.rag_tasks import create_rag_crew
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
import logging
import os
import json
import time
from datetime import datetime

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
retriever_agent = None
rag_agent = None
llm_agent = None
rag_crew = None

@app.on_event("startup")
async def startup_event():
    global retriever_agent, rag_agent, llm_agent, rag_crew
    try:
        logger.info("Initializing RAG agents and crew...")
        retriever_agent, rag_agent, llm_agent = create_rag_agents()
        rag_crew = create_rag_crew(retriever_agent, rag_agent, llm_agent)
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
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if not rag_crew:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    question = user_messages[-1].content
    
    try:
        # Process with RAG crew
        result = rag_crew.kickoff(inputs={"query": question})
        answer = str(result)
        
        if request.stream:
            return StreamingResponse(
                generate_stream_response(answer, request.model),
                media_type="text/plain"
            )
        else:
            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": len(question.split()),
                    "completion_tokens": len(answer.split()),
                    "total_tokens": len(question.split()) + len(answer.split())
                }
            ).dict()
    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def generate_stream_response(content: str, model: str):
    """Generate streaming response for chat completions"""
    words = content.split()
    
    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.05)  # Small delay for streaming effect
    
    # Final chunk
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
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

# Legacy RAG endpoint
@app.get("/query")
async def query_rag(question: str = Query(..., description="User query")):
    if not rag_crew:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_crew.kickoff(inputs={"query": question})
        return {"question": question, "answer": str(result)}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_crew is not None,
        "timestamp": datetime.now().isoformat()
    }
