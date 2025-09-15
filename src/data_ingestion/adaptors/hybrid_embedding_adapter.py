import copy
import logging
from typing import List, Dict, Any
from ..base.base_embedding import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class HybridEmbeddingAdapter(BaseEmbedding):
    """
    Hybrid embedding adapter that generates both dense embeddings (via Ollama)
    and sparse embeddings (via TF-IDF/BM25) for hybrid search compatibility.
    """

    def __init__(
        self,
        dense_model_name: str = "nomic-embed-text",
        context_model: str = "gemma:2b",
        base_url: str = "http://ollama:11434",
        context_prompt_template: str = None,
        max_features: int = 10000,
        use_context_enrichment: bool = True,
    ):
        self.dense_model_name = dense_model_name
        self.context_model = context_model
        self.base_url = base_url
        self.use_context_enrichment = use_context_enrichment
        
        # Dense embedding components
        self.dense_embedder = OllamaEmbedding(model_name=dense_model_name, base_url=base_url)
        
        if use_context_enrichment:
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
        
        # Sparse embedding components (TF-IDF for BM25-like scoring)
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Only alphanumeric tokens
        )
        self.is_fitted = False
        self._dense_dim = None
        self._sparse_dim = None

    def _generate_context(self, text: str) -> str:
        """Generate context string using the context LLM."""
        if not self.use_context_enrichment:
            return ""
            
        try:
            prompt = self.context_prompt_template.format(CHUNK_CONTENT=text)
            response = self.context_llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"[HybridEmbeddingAdapter] Context generation failed: {e}")
            return ""

    def _fit_sparse_vectorizer(self, texts: List[str]):
        """Fit the sparse vectorizer on the corpus."""
        if not self.is_fitted:
            logger.info("Fitting sparse vectorizer on corpus...")
            self.sparse_vectorizer.fit(texts)
            self.is_fitted = True
            self._sparse_dim = len(self.sparse_vectorizer.vocabulary_)
            logger.info(f"Sparse vectorizer fitted with {self._sparse_dim} features")

    def embed(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate both dense and sparse embeddings for chunks.
        
        Returns:
        List of dicts with both 'dense_embedding' and 'sparse_embedding' keys
        """
        if not chunks:
            return []

        # Extract texts for batch processing
        texts = []
        enriched_chunks = []
        
        for chunk in chunks:
            text = chunk["text"]
            metadata = copy.deepcopy(chunk.get("metadata", {}))
            
            # Generate context if enabled
            if self.use_context_enrichment:
                context = self._generate_context(text)
                metadata["context"] = context
                # Create contextualized text for dense embedding
                contextualized_text = f"[Context: {context}] {text}" if context else text
            else:
                contextualized_text = text
            
            texts.append(text)  # Original text for sparse embedding
            enriched_chunks.append({
                "original_text": text,
                "contextualized_text": contextualized_text,
                "metadata": metadata
            })

        # Fit sparse vectorizer if not already fitted
        self._fit_sparse_vectorizer(texts)

        # Generate sparse embeddings (TF-IDF vectors)
        sparse_matrix = self.sparse_vectorizer.transform(texts)
        
        # Generate dense embeddings
        contextualized_texts = [chunk["contextualized_text"] for chunk in enriched_chunks]
        dense_embeddings = []
        
        for ctx_text in contextualized_texts:
            dense_emb = self.dense_embedder.get_text_embedding(ctx_text)
            dense_embeddings.append(dense_emb)
            
            if self._dense_dim is None:
                self._dense_dim = len(dense_emb)

        # Combine embeddings
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # Convert sparse vector to dense array for storage
            sparse_vector = sparse_matrix[i].toarray().flatten()
            
            # Separate context from stable metadata for deduplication
            original_metadata = copy.deepcopy(chunk.get("metadata", {}))
            context = enriched_chunks[i]["metadata"].get("context", "")
            
            result_chunk = {
                "text": chunk["text"],
                "metadata": original_metadata,  # Use original metadata for dedup_key
                "context": context,  # Store context separately
                "embedding": dense_embeddings[i],  # Keep backward compatibility
                "dense_embedding": dense_embeddings[i],
                "sparse_embedding": sparse_vector.tolist(),
                "embedding_type": "hybrid"
            }
            result_chunks.append(result_chunk)

        return result_chunks

    def dimension(self) -> Dict[str, int]:
        """Return dimensions for both dense and sparse embeddings."""
        if self._dense_dim is None or self._sparse_dim is None:
            # Generate test embeddings to get dimensions
            test_chunks = [{"text": "test document", "metadata": {}}]
            self.embed(test_chunks)
        
        return {
            "dense": self._dense_dim,
            "sparse": self._sparse_dim
        }

    def get_sparse_vocabulary(self) -> Dict[str, int]:
        """Get the sparse vectorizer vocabulary mapping."""
        if not self.is_fitted:
            raise ValueError("Sparse vectorizer not fitted yet. Call embed() first.")
        return self.sparse_vectorizer.vocabulary_

    def transform_query(self, query: str) -> Dict[str, Any]:
        """Transform a query into both dense and sparse representations."""
        if not self.is_fitted:
            raise ValueError("Adapter not fitted yet. Process some documents first.")
        
        # Generate dense embedding for query
        if self.use_context_enrichment:
            # Optionally add context to query as well
            context = self._generate_context(query)
            contextualized_query = f"[Context: {context}] {query}" if context else query
        else:
            contextualized_query = query
            
        dense_embedding = self.dense_embedder.get_text_embedding(contextualized_query)
        
        # Generate sparse embedding for query
        sparse_vector = self.sparse_vectorizer.transform([query]).toarray().flatten()
        
        return {
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_vector.tolist(),
            "original_query": query
        }
