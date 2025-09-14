import re
from typing import List, Dict, Any
from transformers import AutoTokenizer
from ..base.base_chunker import BaseChunker


class MarkdownChunker(BaseChunker):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500, overlap: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text_block(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            token_ids = tokens[start:end]
            chunk_text = self.tokenizer.decode(token_ids)
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "block_type": "text"}
            })
            start += self.chunk_size - self.overlap
        return chunks

    def split(self, md_text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        chunks = []

        # Regex for markdown blocks
        table_pattern = re.compile(r"(\|.*\|(?:\n\|.*\|)+)", re.MULTILINE)
        image_pattern = re.compile(r"!\[.*?\]\(.*?\)")
        code_pattern = re.compile(r"```.*?```", re.DOTALL)

        protected_blocks = []
        for pattern, block_type in [(table_pattern, "table"), (image_pattern, "image"), (code_pattern, "code")]:
            for match in pattern.finditer(md_text):
                protected_blocks.append((match.span(), match.group(), block_type))

        protected_blocks.sort(key=lambda x: x[0][0])
        last_idx = 0

        for (start, end), block_text, block_type in protected_blocks:
            if last_idx < start:
                text_block = md_text[last_idx:start].strip()
                if text_block:
                    chunks.extend(self._split_text_block(text_block, metadata))

            chunks.append({
                "text": block_text.strip(),
                "metadata": {**metadata, "block_type": block_type}
            })
            last_idx = end

        if last_idx < len(md_text):
            text_block = md_text[last_idx:].strip()
            if text_block:
                chunks.extend(self._split_text_block(text_block, metadata))

        return chunks
