# main.py
from fastapi import FastAPI, Query
from src.rag.agents.rag_agent import create_rag_agents
from src.rag.tasks.rag_tasks import create_rag_crew
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DIM

app = FastAPI(title="Agentic RAG API")

# --- DB Config ---
db_config = {
    "db": DATABASE_NAME,
    "user": DATABASE_USER,
    "password": DATABASE_PASSWORD,
    "host": DATABASE_HOST,
    "port": DATABASE_PORT,
    "table": "documents",
}

retriever_agent, rag_agent, llm_agent = create_rag_agents()
rag_crew = create_rag_crew(retriever_agent, rag_agent, llm_agent)

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/query")
async def query_rag(question: str = Query(..., description="User query")):
    result = rag_crew.kickoff(inputs={"query": question})
    return {"question": question, "answer": str(result)}
