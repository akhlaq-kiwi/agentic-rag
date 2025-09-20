from src.logger import get_logger
from ..base.base_extractor import BaseExtractor
from typing import Union, Dict, Any
from pathlib import Path
from docling.document_converter import DocumentConverter
import inspect

logger = get_logger("Docling Extractor", "logs/ingestion.log")


class DoclingExtractor(BaseExtractor):
    def __init__(self):
        try:
            self.converter = DocumentConverter()
            logger.info("Docling DocumentConverter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentConverter: {e}")
            raise

    def extract(self, source: Union[str, Path], export_format: str = "text", **kwargs: Any) -> Union[str, Dict]:
        """
        For export_format="markdown", returns a single markdown string with explicit page breaks:
            <!-- PAGE_BREAK: 1 -->
            ...page 1 md...
            <!-- PAGE_BREAK: 2 -->
            ...page 2 md...
        Optional kwargs:
            - page_marker (str): custom marker template, default "<!-- PAGE_BREAK: {page_no} -->"
        """
        file_name = Path(source).name
        logger.info(f"[Docling] Extracting {source} as {export_format}")
        result = self.converter.convert(str(source))
        doc = result.document

        if export_format == "json":
            return doc.export_to_dict()
        elif export_format == "text":
            return doc.export_to_text()
        elif export_format == "markdown":
            marker_tpl: str = kwargs.get("page_marker", "<!-- Metadata(Page No.: {page_no}, Filename: {file_name}) -->")

            # Detect support for page-wise export
            supports_page_no = False
            try:
                sig = inspect.signature(doc.export_to_markdown)
                supports_page_no = "page_no" in sig.parameters
            except Exception:
                pass

            # Helper to get number of pages (best-effort across versions)
            def _num_pages(d) -> int:
                try:
                    return d.num_pages()
                except Exception:
                    pages = getattr(d, "pages", None)
                    return len(pages) if pages is not None else 1

            # Preferred path: export each page individually and join with markers
            if supports_page_no:
                n = _num_pages(doc)
                parts = []
                for p in range(1, n + 1):
                    try:
                        page_md = doc.export_to_markdown(page_no=p)
                    except Exception as e:
                        logger.error(f"[Docling] export_to_markdown(page_no={p}) failed: {e}")
                        raise
                    parts.append(f"{marker_tpl.format(page_no=p, file_name=file_name)}\n\n{page_md}".strip())
                joined = "\n\n".join(parts) + "\n"
                return joined

            # Fallback: ask Docling to insert page markers during a single export (if supported)
            try:
                return doc.export_to_markdown(page_break_placeholder=marker_tpl)
            except TypeError:
                # Very old builds may not support page_break_placeholder—just return plain markdown
                logger.warning("[Docling] page_break_placeholder not supported; returning full markdown without markers")
                return doc.export_to_markdown()
        else:
            logger.error(f"Unsupported format: {export_format}")
            raise ValueError(f"Unsupported export format: {export_format}")
