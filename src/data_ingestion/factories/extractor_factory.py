from src.logger import get_logger
from ..base.base_extractor import BaseExtractor
from ..adaptors.docling_adaptor import DoclingExtractor

logger = get_logger("Extractor Factory", "logs/ingestion.log")

class ExtractorFactory:
    @staticmethod
    def get_extractor(name: str) -> BaseExtractor:
        name = name.lower()
        logger.debug(f"Initializing extractor: {name}")
        if name == "docling":
            return DoclingExtractor()
        else:
            logger.error(f"Unknown extractor: {name}")
            raise ValueError(f"Unknown extractor: {name}")