from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingsStore:
    """Wrapper for sentence-transformers to generate embeddings."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        # Verify the dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of strings."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
