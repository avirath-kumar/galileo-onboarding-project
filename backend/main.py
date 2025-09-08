from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import uvicorn
import os

# Import agent
from agent_graph import process_query
from rag_pipeline import get_rag_pipeline

# Load env vars
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize openai llm
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create classes for chat requests and responses
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    response: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Agent API is running"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Chat endpoint - uses the agent, will automatically classify / route to rag if needed
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await process_query(
            user_query=request.message,
            conversation_history=request.conversation_history
        )

        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Initialize RAG on startup
@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing RAG pipeline...")
        rag = get_rag_pipeline()
        stats = rag.get_stats()
        print(f"RAG pipeline ready with {stats['total_chunks']} chunks")
    except Exception as e:
        print(f"Warning: Could not initialize RAG pipeline: {str(e)}")

# Main function
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )