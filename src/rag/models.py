"""
Pydantic models for structured RAG responses with grounding metadata.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SourceReference(BaseModel):
    """A single source reference with file and page information."""
    file_name: str = Field(description="Name of the source document")
    page_number: int = Field(description="Page number in the document")
    chunk_text: Optional[str] = Field(default=None, description="The actual text chunk from the document")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score of this chunk")


class GroundingMetadata(BaseModel):
    """Metadata about the sources used to ground the response."""
    sources: List[SourceReference] = Field(description="List of source references used")
    total_chunks: int = Field(description="Total number of chunks retrieved")
    query: str = Field(description="The original query")
    
    def format_sources(self) -> str:
        """Format sources as a citation string."""
        # Group by file name
        file_pages: Dict[str, List[int]] = {}
        for source in self.sources:
            if source.file_name not in file_pages:
                file_pages[source.file_name] = []
            if source.page_number not in file_pages[source.file_name]:
                file_pages[source.file_name].append(source.page_number)
        
        # Sort pages for each file
        for file_name in file_pages:
            file_pages[file_name].sort()
        
        # Format as citation string
        citations = []
        for file_name, pages in sorted(file_pages.items()):
            pages_str = ", ".join(str(p) for p in pages)
            citations.append(f"{file_name} (Page {pages_str})")
        
        return "; ".join(citations)


class RAGResponse(BaseModel):
    """Complete RAG response with answer and grounding metadata."""
    answer: str = Field(description="The natural language answer to the query")
    grounding: GroundingMetadata = Field(description="Metadata about sources used for grounding")
    
    def format_with_sources(self) -> str:
        """Format the complete response with sources section."""
        sources_str = self.grounding.format_sources()
        return f"{self.answer}\n\nSources: {sources_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "grounding": {
                "sources": [
                    {
                        "file_name": s.file_name,
                        "page_number": s.page_number,
                        "relevance_score": s.relevance_score
                    }
                    for s in self.grounding.sources
                ],
                "total_chunks": self.grounding.total_chunks,
                "query": self.grounding.query
            },
            "formatted_response": self.format_with_sources()
        }


class RetrievalResult(BaseModel):
    """Structured result from the retrieval tool."""
    chunks: List[Dict[str, Any]] = Field(description="Retrieved document chunks with metadata")
    sources: List[SourceReference] = Field(description="Source references extracted from chunks")
    formatted_context: str = Field(description="Formatted context for the LLM")
    
    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM consumption."""
        return self.formatted_context
