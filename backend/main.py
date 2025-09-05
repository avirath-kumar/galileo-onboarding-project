from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

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

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = llm.invoke(request.message)
        return ChatResponse(response=response.content)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

# Main function
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)