from pydantic import BaseModel
from typing import Dict, Any, List
from app.rag_pipeline import RAGPipeline

class AgentResponse(BaseModel):
    action: str
    response: str
    recommendations: List[str]

class KnowledgeAgent:
    """An agent that decides to search knowledge or answer directly."""

    def __init__(self, rag_pipeline: RAGPipeline = None):
        self.rag = rag_pipeline or RAGPipeline()
        self.client = self.rag.client

    def decide_action(self, query: str) -> str:
        """Determines if the query needs memory retrieval or is conversational."""
        conversational_intents = ["hi", "hello", "who are you", "what can you do"]
        if any(intent in query.lower() for intent in conversational_intents):
            return "chat"
        return "search_memory"

    def get_recommendations(self, query: str) -> List[str]:
        """Fetch similar topics based on semantic search."""
        results = self.rag.retrieval.semantic_search(query, k=3)
        topics = set()
        for r in results:
             source = r.metadata.get('source', 'Unknown Document')
             topics.add(f"Topics from '{source}'")
        return list(topics)

    def process(self, query: str) -> AgentResponse:
        """The main agent loop."""
        action = self.decide_action(query)
        recommendations = []
        
        if action == "chat":
             response = "Hello! I am your AI Second Brain. I can help you search through your uploaded documents and notes."
        else:
             rag_result = self.rag.generate_answer(query)
             response = rag_result["answer"]
             recommendations = self.get_recommendations(query)
             
        return AgentResponse(
            action=action,
            response=response,
            recommendations=recommendations
        )

    def process_stream(self, query: str):
        """Streaming agent loop."""
        action = self.decide_action(query)
        
        if action == "chat":
             yield "Hello! I am your AI Second Brain. I can help you search through your uploaded documents and notes."
        else:
             rag_stream = self.rag.generate_answer_stream(query)
             for chunk in rag_stream:
                 yield chunk
             
             # After stream finishes, generate and append recommendations
             recommendations = self.get_recommendations(query)
             if recommendations:
                 yield "\n\n**Related Topics in Your Brain:**\n"
                 for r in set(recommendations):
                     yield f"- {r}\n"
