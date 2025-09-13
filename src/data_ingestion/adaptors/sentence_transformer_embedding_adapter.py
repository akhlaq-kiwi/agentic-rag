from typing import List
from sentence_transformers import SentenceTransformer
from ..base.base_embedding import BaseEmbedding


class SentenceTransformerEmbeddingAdapter(BaseEmbedding):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Precompute dimension once
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=False)

    def dimension(self) -> int:
        return self._dim
