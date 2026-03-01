import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.retrieval import RetrievalEngine, SearchResult

class RAGPipeline:
    """Coordinates retrieval and generative language modeling."""

    def __init__(self, retrieval_engine: RetrievalEngine = None):
        self.retrieval = retrieval_engine or RetrievalEngine()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if self.openrouter_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
            )
            self.model = "z-ai/glm-4.5-air:free"
            self.is_openrouter = True
        elif self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
            self.model = "gpt-3.5-turbo"
            self.is_openrouter = False
        else:
            self.client = None
            self.is_openrouter = False
            
        if not self.client:
             print("WARNING: OPENROUTER_API_KEY / OPENAI_API_KEY not found. RAG will fallback to Context-only responses.")

    def build_prompt(self, query: str, context_chunks: List[SearchResult]) -> str:
        """Constructs a prompt bounded by retrieved context."""
        context_str = "\n\n---\n\n".join([
            f"Source: {c.metadata.get('source', 'Unknown')}\nText: {c.text}" 
            for c in context_chunks
        ])
        
        return f"""You are a helpful Personal Knowledge Assistant, an "AI Second Brain".
Your goal is to answer the user's question accurately, based ONLY on the provided context retrieved from the user's notes and documents.

If the answer cannot be found in the context, politely inform the user that you don't have that information in their stored knowledge. Do not hallucinate or rely on outside internet knowledge.

Context from User's Second Brain:
==================================
{context_str}
==================================

User Question: {query}
Answer:"""

    def generate_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Runs the full RAG workflow to answer a query."""
        # 1. Retrieve Context
        results = self.retrieval.semantic_search(query, k=k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant documents in your knowledge base.",
                "context": []
            }
            
        # Extract unique sources for citations
        sources = list(set([r.metadata.get("source", "Unknown") for r in results]))
            
        # 2. Check if we have an LLM enabled
        if not self.client:
             fallback_answer = "API Key not configured (Set OPENROUTER_API_KEY or OPENAI_API_KEY).\n\n"
             fallback_answer += "**Relevant Information Found:**\n\n"
             for i, r in enumerate(results):
                 source = r.metadata.get('source', 'Unknown')
                 fallback_answer += f"**Chunk {i+1}** (from `{source}`):\n> {r.text}...\n\n"
                 
             return {
                 "answer": fallback_answer,
                 "sources": sources,
                 "context": [r.model_dump() for r in results]
             }

        # 3. Generate Answer using LLM
        prompt = self.build_prompt(query, results)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise and helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2, # Keep hallucination low
                "max_tokens": 600
            }
            if self.is_openrouter:
                kwargs["extra_headers"] = {
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-OpenRouter-Title": "AI Second Brain", 
                }
            
            response = self.client.chat.completions.create(**kwargs)
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error communicating with LLM Provider: {e}"
            
        return {
            "answer": answer,
            "sources": sources,
            "context": [r.model_dump() for r in results]
        }

    def generate_answer_stream(self, query: str, k: int = 5):
        """Runs the RAG workflow and yields the answer as a stream of text chunks."""
        # 1. Retrieve Context
        results = self.retrieval.semantic_search(query, k=k)
        
        if not results:
            yield "I couldn't find any relevant documents in your knowledge base."
            return
            
        sources = list(set([r.metadata.get("source", "Unknown") for r in results]))
        
        # 2. Check LLM
        if not self.client:
             yield "API Key not configured (Set OPENROUTER_API_KEY or OPENAI_API_KEY).\n\n"
             yield "**Relevant Information Found:**\n\n"
             for i, r in enumerate(results):
                 source = r.metadata.get('source', 'Unknown')
                 yield f"**Chunk {i+1}** (from `{source}`):\n> {r.text}...\n\n"
             return

        # 3. Stream Answer
        prompt = self.build_prompt(query, results)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise and helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 600,
                "stream": True # Enable streaming
            }
            if self.is_openrouter:
                kwargs["extra_headers"] = {
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-OpenRouter-Title": "AI Second Brain", 
                }
            
            response_stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error communicating with LLM Provider: {e}"
