from typing import TypedDict, Annotated, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the agent state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # reducer function so multiple nodes can run in parallel
    classification: str  # classify the input type
    tool_results: List[Dict]
    analysis: str
    final_response: str

# Tool definitions - functions the agent can call
@tool
def example_tool_one(input_param: str) -> Dict:
    """Example tool that processes some input and returns results."""
    
    try:
        # Your tool logic here
        result = {
            "status": "success",
            "data": f"Processed: {input_param}",
            "metadata": {"timestamp": "2024-01-01"}
        }
        return result
    except Exception as e:
        return {"error": f"Tool Error: {str(e)}"}

@tool
def example_tool_two(data: List[dict], operation_type: str) -> Dict:
    """Example tool that performs operations on data."""
    
    if not data:
        return {"error": "No data provided"}
    
    try:
        # Your processing logic here
        if operation_type == "summarize":
            return {"summary": f"Processed {len(data)} items"}
        elif operation_type == "analyze":
            return {"analysis": "Analysis complete"}
        else:
            return {"result": "Operation completed"}
    except Exception as e:
        return {"error": f"Processing Error: {str(e)}"}

@tool
def get_context_info() -> str:
    """Get contextual information for the agent."""
    
    try:
        context_info = """
        System Context Information:
        - Context item 1
        - Context item 2
        - Context item 3
        
        Additional Guidelines:
        - Guideline 1
        - Guideline 2
        """
        return context_info
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

# Node functions - these get called when the graph executes
def classify_input(state: AgentState) -> AgentState:
    """Node 1: Classify the user's input to determine how to handle it."""
    messages = state["messages"]
    last_message = messages[-1].content

    classification_prompt = f"""
    [CLASSIFICATION PROMPT PLACEHOLDER]
    Classify this user input into one of these categories:
    - "type_a": Description of type A inputs
    - "type_b": Description of type B inputs  
    - "general": General inputs
    
    User input: "{last_message}"
    
    Respond with just the category name.
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # Validate the classification
    if classification not in ["type_a", "type_b", "general"]:
        classification = "general"
    
    state["classification"] = classification
    return state

def process_type_a(state: AgentState) -> AgentState:
    """Node 2a: Process Type A inputs using tools."""
    messages = state["messages"]
    last_message = messages[-1].content

    # Get context information
    context = get_context_info.invoke({})

    processing_prompt = f"""
    [TYPE A PROCESSING PROMPT PLACEHOLDER]
    
    Context: {context}
    
    User input: "{last_message}"
    
    Generate appropriate parameters for tool execution.
    """

    response = llm.invoke([HumanMessage(content=processing_prompt)])
    tool_input = response.content.strip()

    # Execute tool and store results
    results = example_tool_one.invoke({"input_param": tool_input})
    state["tool_results"] = [results]

    return state

def analyze_results(state: AgentState) -> AgentState:
    """Node 3: Analyze results and generate insights."""
    results = state["tool_results"]
    query = state["messages"][-1].content

    if not results or "error" in results[0]:
        state["analysis"] = "Unable to process the request. Please try again."
        return state
    
    analysis_prompt = f"""
    [ANALYSIS PROMPT PLACEHOLDER]
    
    User asked: "{query}"
    
    Results: {results}
    
    Provide analysis focusing on:
    - Key point 1
    - Key point 2
    - Key point 3
    """

    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    state["analysis"] = analysis_response.content

    return state

def handle_general(state: AgentState) -> AgentState:
    """Node 2b: Handle general inputs without tool usage."""
    messages = state["messages"]

    system_prompt = """[GENERAL HANDLER PROMPT PLACEHOLDER]
    You are a helpful assistant. 
    Provide a helpful response to the user's input.
    Keep responses concise and friendly."""

    response = llm.invoke([
        HumanMessage(content=system_prompt),
        *messages
    ])

    state["final_response"] = response.content
    return state

def format_response(state: AgentState) -> AgentState:
    """Node 4: Format the final response for the user."""
    
    # Skip if we already have a final response
    if state.get("final_response"):
        return state
    
    # Format responses based on classification
    if state["classification"] in ["type_a", "type_b"]:
        analysis = state.get("analysis", "")
        results = state.get("tool_results", [])

        if results and "error" not in results[0]:
            response = f"{analysis}\n\n"
            # Add any additional formatting logic here
            response += f"(Processed {len(results)} result(s))"
        else:
            response = analysis

        state["final_response"] = response
    
    return state

# Define routing logic
def route_after_classification(state: AgentState) -> Literal["process_type_a", "handle_general"]:
    """Routing function: Decides which node to go to after classification."""
    
    classification = state["classification"]
    
    if classification in ["type_a", "type_b"]:
        return "process_type_a"
    else:
        return "handle_general"

# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent."""
    
    # Initialize the graph with our state type
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("classify_input", classify_input)
    graph.add_node("process_type_a", process_type_a)
    graph.add_node("analyze_results", analyze_results)
    graph.add_node("handle_general", handle_general)
    graph.add_node("format_response", format_response)

    # Define the edges between nodes
    graph.set_entry_point("classify_input")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_input",
        route_after_classification,
        {
            "process_type_a": "process_type_a",
            "handle_general": "handle_general"
        }
    )

    # Linear flow for type_a path
    graph.add_edge("process_type_a", "analyze_results")
    graph.add_edge("analyze_results", "format_response")

    # General path goes straight to response formatting
    graph.add_edge("handle_general", "format_response")

    # All paths end at format_response
    graph.add_edge("format_response", END)

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
        conversation_history: Previous messages in the conversation
    
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
        "tool_results": [],
        "analysis": "",
        "final_response": ""
    }

    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    return result["final_response"]