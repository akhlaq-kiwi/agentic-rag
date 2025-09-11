from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from pathlib import Path

"""
    Abstract base class for all extractors.
    An extractor is a class that extracts content from a source and returns it in a specified format.
    Attributes:
        None

    Methods:
        extract(source, export_format): Extracts content from the source and returns it in the specified format.

"""
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, source: Union[str, Path], export_format: str = "text") -> Union[str, Dict, Any]:
        """
        Extracts content from the source and returns it in the specified format.

        Args:
            source (Union[str, Path]): The source to extract content from.
            export_format (str): The format to export the content in.

        Returns:
            Union[str, Dict, Any]: The extracted content in the specified format.
        """
        pass