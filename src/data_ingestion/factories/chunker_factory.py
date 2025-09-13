from ..adaptors.markdown_chunk_adapter import MarkdownChunker
from ..base.base_chunker import BaseChunker


class ChunkerFactory:
    @staticmethod
    def get_chunker(strategy: str, **kwargs) -> BaseChunker:
        strategy = strategy.lower()
        if strategy == "markdown":
            return MarkdownChunker(**kwargs)
        elif strategy == "char":
            return CharChunker(**kwargs)
        elif strategy == "token":
            return TokenChunker(**kwargs)
        elif strategy == "sentence":
            return SentenceChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunker strategy: {strategy}")
