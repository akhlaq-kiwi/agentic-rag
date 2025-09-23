from src.rag.tasks.rag_tasks import create_rag_crew

def create_optimized_rag_crew(query: str):
    """Create and configure optimized RAG crew for fast processing."""
    
    # Create optimized crew (2 tasks instead of 4)
    crew = create_rag_crew(query=query)
    
    return crew

def run_rag_query(query: str) -> str:
    """Run a RAG query with optimized crew setup."""
    try:
        # Create optimized crew
        crew = create_optimized_rag_crew(query)
        
        # Execute with query
        result = crew.kickoff(inputs={"query": query})
        
        return result
        
    except Exception as e:
        return f"Error processing query: {str(e)}"
