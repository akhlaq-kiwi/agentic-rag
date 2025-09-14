from ..adaptors.sentence_transformer_embedding_adapter import SentenceTransformerEmbeddingAdapter
from ..base.base_embedding import BaseEmbedding
from ..adaptors.ollama_embedding_adapter import OllamaEmbeddingAdapter

class EmbedderFactory:
    @staticmethod
    def get_embedder(name: str, **kwargs) -> BaseEmbedding:
        name = name.lower()
        if name in ["intfloat/e5-base-v2", "sbert", "sentence-transformer", "mini-lm", "nomic-embed-text:v1.5", "nomic-embed-text"]:
            return SentenceTransformerEmbeddingAdapter(**kwargs)
        if name in ["ollama", "ollama-embed-text"]:
            return OllamaEmbeddingAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown embedding model: {name}")
