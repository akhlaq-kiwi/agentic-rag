from docling.document_converter import DocumentConverter
from src.logger import get_logger
from ..base.base_extractor import BaseExtractor
from typing import Union, Dict, Any
from pathlib import Path

logger = get_logger("Docling Extractor", "logs/ingestion.log")

class DoclingExtractor(BaseExtractor):
    def __init__(self):
        self.converter = DocumentConverter()

    def extract(self, source: Union[str, Path], export_format: str = "text") -> Union[str, Dict]:
        logger.info(f"[Docling] Extracting {source} as {export_format}")
        result = self.converter.convert(str(source))
        doc = result.document

        if export_format == "markdown":
            return doc.export_to_markdown()
        elif export_format == "json":
            return doc.export_to_dict()
        elif export_format == "text":
            return doc.export_to_text()
        else:
            logger.error(f"Unsupported format: {export_format}")
            raise ValueError(f"Unsupported export format: {export_format}")