import copy
import logging
from typing import List, Dict
from ..base.base_embedding import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)


class OllamaEmbeddingAdapter(BaseEmbedding):
    """
    Embedding adapter that enriches chunks with context using an LLM,
    then embeds them with Ollama's embedding model.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        context_model: str = "gemma:2b",
        base_url: str = "http://ollama:11434",
        context_prompt_template: str = None,
    ):
        self.embed_model = model_name
        self.context_model = context_model
        self.base_url = base_url

        self.embedder = OllamaEmbedding(model_name=model_name, base_url=base_url)
        self.context_llm = Ollama(model=context_model, base_url=base_url, request_timeout=60)

        self.context_prompt_template = context_prompt_template or (
            "You are enriching a document chunk with context.\n\n"
            "<chunk>\n{CHUNK_CONTENT}\n</chunk>\n\n"
            "Provide a 1-2 sentence summary describing:\n"
            "1. Which section/topic this chunk relates to\n"
            "2. How it connects to the overall document\n"
            "3. Its relationship to other sections\n\n"
            "Respond with only the context, nothing else."
        )

        self._dim = None

    def _generate_context(self, text: str) -> str:
        """Generate context string using the context LLM."""
        try:
            prompt = self.context_prompt_template.format(CHUNK_CONTENT=text)
            response = self.context_llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"[ContextualOllamaEmbedding] Context generation failed: {e}")
            return "No additional context available"

    def embed(self, chunks: List[Dict]) -> List[List[float]]:
        """
        Expects a list of dicts:
        [
          {"text": "...", "metadata": {...}},
          {"text": "...", "metadata": {...}}
        ]
        """
        embeddings = []
        for chunk in chunks:
            text = chunk["text"]
            metadata = copy.deepcopy(chunk.get("metadata", {}))

            # Generate context
            context = self._generate_context(text)
            metadata["context"] = context

            # Create contextualized text
            contextual_text = f"[Context: ] {text}"

            # Get embedding from Ollama
            emb = self.embedder.get_text_embedding(contextual_text)
            embeddings.append(emb)

            if self._dim is None:
                self._dim = len(emb)

        return embeddings

    def dimension(self) -> int:
        if self._dim is None:
            test_emb = self.embed([{"text": "test", "metadata": {}}])
            self._dim = len(test_emb[0])
        return self._dim
