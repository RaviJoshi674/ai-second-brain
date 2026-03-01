# AI Second Brain — Personal Knowledge Assistant

An intelligent, production-style knowledge system that acts as a user's personal memory. It allows users to upload documents, securely store them locally, and interact with the knowledge using semantic search and Retrieval-Augmented Generation (RAG).

**Core Engine:** This project leverages **[Endee](https://github.com/endee-io/endee)** as the hyper-fast vector database powering the core semantic search layer.

---

## 1. Project Overview

The AI Second Brain is designed to solve knowledge fragmentation. We consume dozens of PDFs, articles, and notes daily, but retrieving specific facts weeks later is tedious. The Second Brain ingests text documents, breaks them into contextual chunks, converts those chunks into mathematical embeddings, and stores them in Endee. When a question is asked, the system performs a vector similarity search to instantly retrieve relevant context, which an AI Agent then uses to formulate a grounded, factual answer.

## 2. Problem Statement

Standard keyword search fails at understanding intent. If a user asks "What did I write about transformer architecture?", traditional systems look for the exact word "transformer architecture". They fail if the document says "The attention mechanism in neural networks...". 

Semantic retrieval solves this by mapping meaning onto highly-dimensional continuous vector space. However, vector searches require specialized databases (Vector DBs) capable of indexing and querying these spaces in milliseconds.

## 3. System Design and Architecture

The application is structured into modular micro-components:

1. **Ingestion (`app/ingestion.py`):** Utilizes `pypdf` and `langchain-text-splitters` to recursively chunk large texts into 500-token overlapping context windows.
2. **Embedding Service (`app/embeddings.py`):** Uses SentenceTransformers (`all-MiniLM-L6-v2`) running locally to convert text chunks into 384-dimensional dense vectors.
3. **Vector Storage (`app/vector_store.py`):** A robust REST client tailored for **Endee**, managing index creation, JSON insertions, and unpacking `msgpack` search results.
4. **Retrieval & RAG logic (`app/retrieval.py`, `app/rag_pipeline.py`):** Responsible for the embedding-search-prompt generation lifecycle. Optionally integrates with OpenAI to generate natural language answers from the retrieved context.
5. **Agent Workflow (`app/agent.py`):** A supervisor node that decides whether the user's intent requires querying the vector database or if it can be handled conversationally. It also generates relevant topic recommendations.
6. **API & UI (`app/api.py`, `ui/app.py`):** A FastAPI backend feeding a dynamic, dual-tabbed Streamlit frontend.

```text
User Query --> Streamlit UI --> FastAPI Backend --> Agent
                                                     |
                                                     v (If Search Intent)
                                              Embed Query (SentenceTransformers)
                                                     |
                                                     v
                                             Search Endee Vector DB 
                                                (REST / msgpack)
                                                     |
                                                     v (Retrieve Chunks)
                                              Prompt Generator
                                                     |
                                                     v
                                        LLM (OpenRouter / OpenAI)
                                                     |
                                                     v
                                           Grounded Final Answer
```

## 4. How Endee Is Used

To achieve milliseconds-fast semantic recall over thousands of chunks, a purpose-built Vector DB is strictly required.

**Why Endee was chosen:**
- **Performance:** Written in C++ targeting modern SIMD instructions (AVX2/AVX512/Neon).
- **Efficiency:** Low memory footprint, enabling the entire Second Brain to run comfortably on a single local machine without bloat.
- **Protocol Flexibility:** Supports REST APIs and returns optimized MessagePack binaries, minimizing deserialization overhead in Python.

**Integration Details:**
The Python `EndeeClient` talks directly to Endee's HTTP endpoints (`/api/v1/...`). 
- When an index is created, we use `cosine` similarity with `HNSW` (`M=16`, `ef_con=100`).
- Insert vectors are batched as `application/json` payloads containing the vector arrays and the raw stringified metadata.
- Searches request the top-k nearest neighbors. The result payload is binary MessagePack, which the client parses natively to extract the `distance` score and document metadata without heavy parsing overhead.

## 5. Setup Instructions

### Prerequisites
1. Python 3.9+
2. A running local instance of **[Endee](https://github.com/endee-io/endee)** on port `8080`.
3. (Optional but recommended) `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for the generative step.

### Installation

1. Clone or download this project (`ai-second-brain`).
2. Make the run script executable:
   ```bash
   cd ai-second-brain
   chmod +x run.sh
   ```
3. Set your OpenRouter key (optional, falls back to OpenAI if only `OPENAI_API_KEY` is set):
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
   ```

## 6. How to Run

A unified startup script is provided which initializes a virtual environment, installs the requirements, starts the FastAPI backend, and boots the Streamlit UI.

Ensure Endee is running on `127.0.0.1:8080`, then run:

```bash
./run.sh
```

- The API is served at `http://localhost:8000`
- The User Interface is available at `http://localhost:8501`

*(To stop the services, press `Ctrl+C` in the terminal).*

## 7. Example Test Script

If you want to view a headless demonstration of the core pipeline (chunking -> embedding -> storing -> fetching) directly against Endee, run:

```bash
# Assuming the virtual environment is activated
python examples/test_endee_integration.py
```

## 8. Example Usage Flow

1. **Upload Phase:** Open the UI `Knowledge Base` tab and upload a document (e.g., `Attention_Is_All_You_Need.pdf`).
2. **Ingestion:** The system splits the paper into sections and sends ~384-dim embeddings to Endee.
3. **Query Phase:** Go to the `Chat` tab and ask: "What is the role of the Multi-Head Attention mechanism?".
4. **Retrieval & RAG:** The backend queries Endee. Endee returns the top 5 chunks containing the exact paragraphs about Multi-Head Attention. 
5. **Answer:** The system outputs a factual summary grounded explicitly in the retrieved chunks, alongside citations to the original document.

## 9. Future Improvements
- **Hybrid Search:** Combine Dense vectors (SentenceTransformers) with Sparse search (BM25) utilizing Endee's hybrid vector support.
- **Document Memory:** Implementing index deletion rules for specific documents.
- **Asynchronous Ingestion:** Moving the ingestion pipeline into a background Celery worker for processing massive PDFs without blocking the API.
