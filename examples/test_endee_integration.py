import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import EndeeClient
from app.embeddings import EmbeddingsStore

def run_demo():
    print("--- AI Second Brain: Endee Integration Demo ---\n")
    
    # 1. Initialize Clients
    endee = EndeeClient()
    embedder = EmbeddingsStore()
    
    # Check Health
    if not endee.check_health():
         print("❌ Endee is not running. Please start it on port 8080.")
         return
    print("✅ Endee is online.")
    
    index_name = "demo_brain"
    dim = embedder.dimension
    
    # 2. Create Index
    print(f"\n[1] Creating Index '{index_name}' with dimension {dim}...")
    success = endee.create_index(index_name, dim)
    if success:
         print("✅ Index created or already exists.")
    else:
         print("❌ Failed to create index.")
         return
         
    # 3. Create sample knowledge
    documents = [
        {"id": "doc_1", "text": "Endee is a high-performance open source vector database built for speed and efficiency.", "source": "Endee Docs"},
        {"id": "doc_2", "text": "Retrieval-Augmented Generation (RAG) improves LLM responses by fetching relevant facts from an external knowledge base.", "source": "AI Notes"},
        {"id": "doc_3", "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.", "source": "FastAPI Docs"},
        {"id": "doc_4", "text": "Sentence Transformers is a Python framework for state-of-the-art sentence, text and image embeddings.", "source": "SentenceTransformers Docs"}
    ]
    
    # 4. Generate Embeddings and Insert
    print(f"\n[2] Embedding and Inserting {len(documents)} documents...")
    vectors_to_insert = []
    for doc in documents:
        embedding = embedder.embed_text(doc["text"])
        vectors_to_insert.append({
             "id": doc["id"],
             "vector": embedding,
             "meta": {
                 "text": doc["text"],
                 "source": doc["source"]
             }
        })
        
    insert_success = endee.insert_vectors(index_name, vectors_to_insert)
    if insert_success:
        print("✅ Documents inserted successfully.")
    else:
        print("❌ Failed to insert documents.")
        return
        
    # 5. Semantic Search
    query = "What is Endee?"
    print(f"\n[3] Running Semantic Search")
    print(f"Query: '{query}'")
    
    query_emb = embedder.embed_text(query)
    results = endee.search(index_name, query_emb, k=2)
    
    print("\n🔍 Search Results:")
    for i, res in enumerate(results):
         meta = res.get("metadata", {})
         score = res.get("distance", 0.0)
         text = meta.get("text", "No text found")
         source = meta.get("source", "Unknown")
         print(f"  {i+1}. [Score: {score:.4f} | Source: {source}] -> {text}")
         
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    run_demo()
