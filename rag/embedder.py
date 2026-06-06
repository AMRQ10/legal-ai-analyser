from sentence_transformers import SentenceTransformer
from typing import List
import os

class DocumentEmbedder:
    def __init__(self):
        """
        Downloads and loads the embedding model on the first run.
        Subsequent runs load from local cache - no internet needed.

        all-MiniLM-L6-v2 is small (80MB), fast, and performs well
        for semantic similarity tasks. Good enough for legal text.
        """
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded successfully.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of text strings into a list of embedding vectors.
        Returns a list of lists of floats.
        """
        if not texts:
            raise ValueError("No texts provided for embedding.")

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        Used at search time to find relevant chunks.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(query)
        return embedding.tolist()
        

