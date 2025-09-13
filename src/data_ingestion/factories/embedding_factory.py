from ..adaptors.sentence_transformer_embedding_adapter import SentenceTransformerEmbeddingAdapter
from ..base.base_embedding import BaseEmbedding


class EmbedderFactory:
    @staticmethod
    def get_embedder(name: str, **kwargs) -> BaseEmbedding:
        name = name.lower()
        if name in ["sbert", "sentence-transformer", "mini-lm", "nomic-embed-text:v1.5", "nomic-embed-text"]:
            return SentenceTransformerEmbeddingAdapter()
        else:
            raise ValueError(f"Unknown embedding model: {name}")
