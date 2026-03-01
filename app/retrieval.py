import os
from typing import List, Dict, Any, Tuple
from app.embeddings import EmbeddingsStore
from app.vector_store import EndeeClient
from app.ingestion import DocumentIngestor
from pydantic import BaseModel

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class RetrievalEngine:
    """Coordinates embedding generation and vector search."""
    
    def __init__(self, index_name: str = "second_brain"):
        self.embeddings = EmbeddingsStore()
        self.endee = EndeeClient()
        self.ingestor = DocumentIngestor()
        self.index_name = index_name
        
        # Ensure index exists
        self._init_index()

    def _init_index(self):
        """Create the Endee index if it doesn't exist."""
        # We can just attempt to create; Endee returns gracefully if it exists
        success = self.endee.create_index(self.index_name, self.embeddings.dimension)
        if success:
            print(f"Index '{self.index_name}' is ready.")
        else:
            print(f"Warning: Failed to initialize index '{self.index_name}'. Endee may not be running.")

    def ingest_document(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """Parse, embed, and insert a document into Endee."""
        chunks = self.ingestor.process_file(file_path, metadata)
        if not chunks:
            return 0
            
        print(f"Generated {len(chunks)} chunks for {os.path.basename(file_path)}")
        
        # Batch generation
        texts = [c["text"] for c in chunks]
        embeddings_list = self.embeddings.embed_batch(texts)
        
        vectors_to_insert = []
        for chunk, emb in zip(chunks, embeddings_list):
            vectors_to_insert.append({
                "id": chunk["id"],
                "vector": emb,
                "meta": chunk["meta"]
            })
            
        success = self.endee.insert_vectors(self.index_name, vectors_to_insert)
        if success:
            print(f"Successfully inserted {len(vectors_to_insert)} chunks into Endee.")
            return len(vectors_to_insert)
        else:
            print("Failed to insert chunks into Endee.")
            return 0

    def semantic_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search Endee for relevant chunks based on semantic similarity."""
        query_emb = self.embeddings.embed_text(query)
        raw_results = self.endee.search(self.index_name, query_emb, k=k)
        
        # Parse into SearchResult objects
        results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            results.append(SearchResult(
                id=str(r["id"]),
                text=text,
                score=r["distance"],
                metadata=meta
            ))
            
        return results
