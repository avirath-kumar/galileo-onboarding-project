from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import uvicorn
import os
import uuid
from galileo import galileo_context
from datetime import datetime

# Import agent & database
from agent_graph import process_query
from rag_pipeline import get_rag_pipeline
from database import get_db

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

# store galileo session ID globally for backend
GALILEO_SESSION_ID = None

# Create classes for requests and responses
class SessionRequest(BaseModel):
    pass

class SessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Agent API is running"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Session endpoint
@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session"""
    db = get_db()
    session_id = db.create_session()
    return SessionResponse(session_id=session_id)

# Chat endpoint - uses the agent, will automatically classify / route to rag if needed
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        db = get_db()

        # create new session if not provided
        if not request.session_id:
            session_id = db.create_session()
        else:
            session_id = request.session_id
            # verify session exists
            if not db.session_exists(session_id):
                raise HTTPException(status_code=404, detail="Session not found")

        # get conversation history
        conversation_history = db.get_session_messages(session_id)

        # save user message
        db.add_message(session_id, "user", request.message)

        # process with agent, pass session_id and galileo_session_id to the agent
        response = await process_query(
            user_query=request.message,
            conversation_history=conversation_history,
            session_id=session_id,
            galileo_session_id=GALILEO_SESSION_ID
        )

        # save assistant response
        db.add_message(session_id, "assistant", response)

        return ChatResponse(response=response, session_id=session_id)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# get session history endpoint
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    db = get_db()

    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}

# Initialize RAG on startup (non-blocking)
@app.on_event("startup")
async def startup_event():
    try:
        print("Starting RAG pipeline initialization (non-blocking)...")
        # Don't block startup - RAG will initialize on first use
        print("RAG pipeline will initialize on first query")

        # initialize galileo session
        if os.getenv("GALILEO_API_KEY"):
            global GALILEO_SESSION_ID
            GALILEO_SESSION_ID = str(uuid.uuid4())
            session_name = f"Backend Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            galileo_context.start_session(name=session_name, external_id=GALILEO_SESSION_ID)
            print(f"Galileo session started: {session_name} (ID: {GALILEO_SESSION_ID[:8]}...)")
        else:
            print("Galileo API key not found - running without Galileo monitoring")
    
    except Exception as e:
        print(f"Warning: Could not initialize RAG pipeline: {str(e)}")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    try:
        if GALILEO_SESSION_ID:
            galileo_context.clear_session()
            print("Galileo session ended")
    except Exception as e:
        print(f"Error ending Galileo session: {str(e)}")

# Main function
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )