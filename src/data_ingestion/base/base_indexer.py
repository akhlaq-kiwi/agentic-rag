from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from pathlib import Path


class BaseIndexer(ABC):
    """
    Abstract base class for document indexing systems.
    
    This class defines the interface for indexing processed documents
    into vector stores or search indices for retrieval in RAG systems.
    """
    
    @abstractmethod
    def create_index(self, sources: List[Union[str, Path]], **kwargs) -> Any:
        """
        Create an index from processed document sources.
        
        Args:
            sources: List of paths to processed documents (markdown, text, json, etc.)
            **kwargs: Additional configuration options
            
        Returns:
            Index object or identifier
        """
        pass
    
    def validate_sources(self, sources: List[Union[str, Path]]) -> List[Path]:
        """
        Validate that source files exist and are readable.
        
        Args:
            sources: List of source file paths
            
        Returns:
            List of validated Path objects
            
        Raises:
            FileNotFoundError: If any source file doesn't exist
            ValueError: If sources list is empty
        """
        if not sources:
            raise ValueError("Sources list cannot be empty")
        
        validated_sources = []
        for source in sources:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            if not source_path.is_file():
                raise ValueError(f"Source is not a file: {source_path}")
            validated_sources.append(source_path)
        
        return validated_sources
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats for this indexer.
        
        Returns:
            List of supported file extensions (e.g., ['.md', '.txt', '.json'])
        """
        return ['.md', '.txt', '.json']  # Default formats, can be overridden
    
    def filter_supported_sources(self, sources: List[Union[str, Path]]) -> List[Path]:
        """
        Filter sources to only include supported formats.
        
        Args:
            sources: List of source file paths
            
        Returns:
            List of Path objects with supported formats
        """
        supported_formats = self.get_supported_formats()
        validated_sources = self.validate_sources(sources)
        
        filtered_sources = []
        for source in validated_sources:
            if source.suffix.lower() in supported_formats:
                filtered_sources.append(source)
        
        return filtered_sources