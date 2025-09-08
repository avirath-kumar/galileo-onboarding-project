from typing import TypedDict, Annotated, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import operator
from dotenv import load_dotenv
import os

# import local rag pipeline as library
from rag_pipeline import get_rag_pipeline

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the agent state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # reducer function so multiple nodes can run in parallel
    classification: str  # classify the input type - "general" or "product_question"
    rag_context: str
    final_response: str

# Tool definitions - functions the agent can call
@tool
def placeholder_tool_one(input_param: str) -> Dict:
    """Placeholder tool for future functionality."""
    return {
        "status": "success",
        "data": f"Placeholder result for: {input_param}"
    }

@tool
def placeholder_tool_two(data: List[dict], operation_type: str) -> Dict:
    """Placeholder tool for future functionality."""
    return {
        "status": "success",
        "operation": operation_type,
        "items_processed": len(data) if data else 0
    }   

# Node functions - these get called when the graph executes

# Node 1: Classify the query
def classify_query(state: AgentState) -> AgentState:
    """Classify whether the query is general or needs RAG"""
    messages = state["messages"]
    last_message = messages[-1].content

    classification_prompt = f"""
    Classify this user query into one of these categories:
    
    1. "product_question" - Questions about products, documentation, technical details, 
       specifications, features, how-to questions, troubleshooting, or anything that 
       would benefit from searching product documentation.
    
    2. "general" - General conversation, greetings, questions about yourself, 
       math problems, coding help unrelated to specific products, or other topics 
       that don't require product documentation.
    
    User query: "{last_message}"
    
    Respond with ONLY the category name (either "product_question" or "general").
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # Validate the classification, default to general
    if classification not in ["product_question", "general"]:
        classification = "general"
    
    state["classification"] = classification
    return state

# Node 2a: Handle general chat
def handle_general_chat(state: AgentState) -> AgentState:
    """Handle general conversation without RAG"""
    messages = state["messages"]

    system_prompt = """You are a helpful AI assistant. Provide friendly, informative responses to general queries. Be conversational but concise."""

    response = llm.invoke([HumanMessage(content=system_prompt), *messages])

    state["final_response"] = response.content
    return state

# Node 2b: Handle product questions with RAG
def handle_product_question(state: AgentState) -> AgentState:
    """Use RAG to answer product related questions"""
    messages = state["messages"]
    query = state["messages"][-1].content

    # Get rag pipeline instance
    rag = get_rag_pipeline()

    # retrieve relevant context
    context = rag.get_context_for_query(query, k=3)
    state["rag_context"] = context

    # Generate response using context
    rag_prompt = f"""You are a helpful product support assistant.
    Use the following context from product documentation to answer the user's question.
    If the context doesn't contain relevant information, say so politely and offer general help.
    
    Context from documentation:
    {context}
    
    User question: {query}
    
    Provide a clear, accurate answer based on the documentation provided."""

    response = llm.invoke([HumanMessage(content=rag_prompt)])
    state["final_response"] = response.content

    return state

# Define routing logic
def route_after_classification(state: AgentState) -> Literal["handle_general_chat", "handle_product_question"]:
    """Routing function: Decides which node to go to after classification."""
    
    if state["classification"] == "product_question":
        return "handle_product_question"
    else:
        return "handle_general_chat"

# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent."""
    
    # Initialize the graph with our state type
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("classify_query", classify_query)
    graph.add_node("handle_general_chat", handle_general_chat)
    graph.add_node("handle_product_question", handle_product_question)

    # Define the edges between nodes
    graph.set_entry_point("classify_query")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "handle_general_chat": "handle_general_chat",
            "handle_product_question": "handle_product_question"
        }
    )

    # Define graph end
    graph.add_edge("handle_general_chat", END)
    graph.add_edge("handle_product_question", END)

    # Compile the graph
    return graph.compile()

# Create a single instance of the agent
agent = create_agent_graph()

# Helper function for easy invocation
async def process_query(user_query: str, conversation_history: List[Dict] = None):
    """
    Process a user query through the agent.
    
    Args:
        user_query: The user's natural language query
        conversation_history: Optional previous messages
    
    Returns:
        The agent's response as a string
    """
    # Build message history
    messages = []
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    
    # Add current query
    messages.append(HumanMessage(content=user_query))

    # Create initial state
    initial_state = {
        "messages": messages,
        "classification": "",
        "rag_context": "",
        "final_response": ""
    }

    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    return result["final_response"]