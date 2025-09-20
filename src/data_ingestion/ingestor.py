import os, json
from src.logger import get_logger
from .base.base_extractor import BaseExtractor
from typing import Union, Dict, List
from pathlib import Path

logger = get_logger("Document Ingestor", "logs/ingestion.log")

class DocumentIngestor:
    def __init__(self, extractor: BaseExtractor):
        self.extractor = extractor

    def ingest(self, source: Union[str, Path], export_format: str = "text") -> Union[str, Dict]:
        logger.info(f"Ingesting {source} with {self.extractor.__class__.__name__}")
        try:
            # save files in processed directory with the name of provided foemat name directory
            processed_dir = source.parent.parent / "processed" / export_format
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_file = processed_dir / Path(source).name

            if export_format == "markdown":
                export_file_ext = ".md"
            elif export_format == "json":
                export_file_ext = ".json"
            elif export_format == "text":
                export_file_ext = ".txt"
            else:
                logger.error(f"Unsupported format: {export_format}")
                raise ValueError(f"Unsupported export format: {export_format}")
            processed_file = Path(processed_file).with_suffix(export_file_ext)

            # if processed_file.exists():
            #     logger.info(f"{processed_file} already exists")
            #     return processed_file

            content = self.extractor.extract(source, export_format)
            with open(processed_file, "w", encoding="utf-8") as f:
                if export_format == "json":
                    json.dump(content, f, ensure_ascii=False, indent=4)
                else:
                    f.write(content)
            
            logger.info(f"Saved {processed_file}")
            return processed_file
        except Exception as e:
            logger.exception(f"Failed to ingest {source}")
            raise

    def batch_ingest(self, sources: List[Union[str, Path]], export_format: str = "text") -> List[Union[str, Dict]]:
        logger.info(f"Batch ingesting {len(sources)} documents")
        try:
            processed_files = []
            for filename in sources:
                processed_files.append(self.ingest(filename, export_format))
            return processed_files
        except Exception as e:
            logger.exception(f"Failed to batch ingest {sources}")
            raise

    def ingest_dir(self, source_dir: Union[str, Path], export_format: str = "text") -> List[Union[str, Dict]]:
        logger.info(f"Ingesting directory {source_dir}")
        sources = [
            Path(source_dir) / f
            for f in os.listdir(source_dir)
            if not f.startswith(".") and not f.endswith(".DS_Store")
        ]
        return self.batch_ingest(sources, export_format)