from src.rag.tasks.rag_tasks import create_rag_crew
from src.rag.tools.rag_tools import retrieve_with_grounding
from src.rag.models import RAGResponse, GroundingMetadata
import logging

logger = logging.getLogger(__name__)

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


def run_rag_query_structured(query: str) -> RAGResponse:
    """
    Run a RAG query and return structured response with grounding metadata.
    
    This function:
    1. Retrieves relevant chunks with metadata
    2. Generates answer using the crew
    3. Returns structured response with separate answer and grounding
    
    Args:
        query: The user's query
    
    Returns:
        RAGResponse with answer and grounding metadata
    """
    try:
        logger.info(f"Running structured RAG query: '{query[:100]}...'")
        
        # Step 1: Retrieve with grounding
        retrieval_result = retrieve_with_grounding(query)
        
        if not retrieval_result.sources:
            # No sources found
            return RAGResponse(
                answer="I couldn't find relevant information in the documents to answer your question.",
                grounding=GroundingMetadata(
                    sources=[],
                    total_chunks=0,
                    query=query
                )
            )
        
        # Step 2: Generate answer using crew with the formatted context
        crew = create_optimized_rag_crew(query)
        
        # Pass the formatted context to the crew
        # The crew will use this context to generate the answer
        answer = crew.kickoff(inputs={"query": query})
        answer_str = str(answer).strip()
        
        # Remove any "Sources:" section that the LLM might have added
        # since we'll add it programmatically
        if "Sources:" in answer_str:
            answer_str = answer_str.split("Sources:")[0].strip()
        
        # Step 3: Create structured response
        response = RAGResponse(
            answer=answer_str,
            grounding=GroundingMetadata(
                sources=retrieval_result.sources,
                total_chunks=len(retrieval_result.chunks),
                query=query
            )
        )
        
        logger.info(f"Structured response generated with {len(retrieval_result.sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error in run_rag_query_structured: {str(e)}", exc_info=True)
        return RAGResponse(
            answer=f"Error processing query: {str(e)}",
            grounding=GroundingMetadata(
                sources=[],
                total_chunks=0,
                query=query
            )
        )
