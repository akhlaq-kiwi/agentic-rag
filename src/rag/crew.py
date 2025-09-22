# src/rag/crew.py - OPTIMIZED RAG CREW SETUP
from src.rag.agents.rag_agent import create_rag_agents
from src.rag.tasks.rag_tasks import create_rag_crew

def create_optimized_rag_crew(query: str):
    """Create and configure optimized RAG crew for fast processing."""
    
    # Create optimized agents (2 instead of 4)
    smart_retriever, answer_generator = create_rag_agents()
    
    # Create optimized crew (2 tasks instead of 4)
    crew = create_rag_crew(smart_retriever, answer_generator)
    
    return crew

def run_rag_query(query: str) -> str:
    """Run a RAG query with optimized crew setup."""
    try:
        # Create optimized crew
        crew = create_optimized_rag_crew(query)
        
        # Execute with query
        result = crew.kickoff(inputs={"query": query})
        
        return str(result)
        
    except Exception as e:
        return f"Error processing query: {str(e)}"
