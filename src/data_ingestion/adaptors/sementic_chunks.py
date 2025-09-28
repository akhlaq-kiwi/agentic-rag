from typing import List, Tuple
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2")  # embeddings model

class SemanticChunker:
    def __init__(self):
        """
        Initialize SemanticChunker with spaCy for sentence splitting
        and SentenceTransformer for embeddings.
        """
        self.nlp = nlp
        self.model = model

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def adaptive_threshold(similarities: List[float], alpha: float = 0.5) -> float:
        """Compute adaptive threshold based on similarities"""
        mean = np.mean(similarities)
        std = np.std(similarities)
        return mean - alpha * std

    def chunk(self, text: str, alpha: float = 0.5) -> Tuple[List[str], float]:
        """
        Perform semantic chunking with adaptive threshold.
        Args:
            text (str): Input text
            alpha (float): Aggressiveness of splitting (higher → more splits)
        Returns:
            chunks (List[str]): Semantic chunks
            threshold (float): Adaptive similarity threshold used
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [], 0.0

        embeddings = self.model.encode(sentences)

        sims = [
            self.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        threshold = self.adaptive_threshold(sims, alpha)

        chunks, current_chunk = [], [sentences[0]]

        for i in range(1, len(sentences)):
            if sims[i - 1] >= threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks, threshold
