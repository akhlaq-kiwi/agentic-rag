# main.py
from fastapi import FastAPI, Query, HTTPException
from src.rag.agents.rag_agent import create_rag_agents
from src.rag.tasks.rag_tasks import create_rag_crew
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM, OLLAMA_BASE_URL
import logging
import os

# Set environment variable for LiteLLM to use Ollama
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG API")

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
