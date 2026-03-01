import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.retrieval import RetrievalEngine
from app.agent import KnowledgeAgent, AgentResponse

app = FastAPI(title="AI Second Brain API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    retrieval_engine = RetrievalEngine(index_name="second_brain")
    agent = KnowledgeAgent(rag_pipeline=None) # Uses default RAG pipeline internally
except Exception as e:
    print(f"Error initializing services: {e}")
    # Continue anyway, let endpoints fail explicitly
    retrieval_engine = None
    agent = None

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    if not retrieval_engine:
        return {"status": "error", "message": "Services not initialized. Is Endee running?"}
    endee_health = retrieval_engine.endee.check_health()
    return {"status": "ok", "endee_connected": endee_health}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file, parse it, and index it into the Second Brain."""
    if not retrieval_engine:
         raise HTTPException(status_code=500, detail="Backend services are offline.")
         
    # Save file temporarily
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Ingest and Index
        num_chunks = retrieval_engine.ingest_document(file_path)
        return {"filename": file.filename, "chunks_indexed": num_chunks, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    finally:
        # Optional: clean up file after ingestion
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query", response_model=AgentResponse)
async def query_brain(request: QueryRequest):
    """Query the second brain."""
    if not agent:
         raise HTTPException(status_code=500, detail="Agent service is offline.")
         
    try:
        response = agent.process(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_stream")
async def query_brain_stream(request: QueryRequest):
    """Query the second brain and stream the response."""
    if not agent:
         raise HTTPException(status_code=500, detail="Agent service is offline.")
         
    return StreamingResponse(agent.process_stream(request.query), media_type="text/plain")
