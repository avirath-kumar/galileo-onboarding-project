from typing import TypedDict, Annotated, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import operator
from dotenv import load_dotenv
import os
import requests
import json

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
def check_inventory_api(product_name: str) -> Dict:
    """Check inventory level for a specific product - atlas 108, nova 75, zephyr 87"""
    try:
        # call inventory api endpoint
        response = requests.get(f"http://localhost:8001/inventory/{product_name}", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Failed to check inventory: {response.text}",
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "error": f"Error connecting to inventory service: {str(e)}"
        }

@tool
def place_order_api(product_name: str, quantity: int, customer_email: str = "customer@example.com") -> Dict:
    """Place order for a specific product. Atlas 108, Nova 75, Zephyr 87"""
    try:
        # call order api endpoint
        order_data = {
            "product_id": product_name,
            "quantity": quantity,
            "customer_email": customer_email
        }

        response = requests.post("http://localhost:8001/order", json=order_data, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Failed to place order: {response.text}",
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "error": f"Error connecting to order service: {str(e)}"
        }

# Bind tools
tools = [check_inventory_api, place_order_api]
llm_with_tools = llm.bind_tools(tools)

# Node functions - these get called when the graph executes

# Node 1: Classify the query
def classify_query(state: AgentState) -> AgentState:
    """Classify whether the query is general or needs RAG"""
    messages = state["messages"]
    last_message = messages[-1].content

    # Check conversation context to see if we're continuing an action
    conversation_context = "\n".join([m.content for m in messages[-3:]]) # last 3 messages for context

    classification_prompt = f"""
    Based on the conversation context, classify the user's intent:
    
    1. "product_question" - Questions about products, documentation, specs, features, troubleshooting
    2. "general" - General conversation, greetings, other topics
    3. "check_inventory" - User wants to check stock/inventory (or responding with product name for inventory check)
    4. "place_order" - User wants to buy/order (or providing order details like product/quantity)
    
    Conversation context:
    {conversation_context}
    
    Respond with ONLY the category name.
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # Validate the classification, default to general
    valid_classifications = ["product_question", "general", "check_inventory", "place_order"]
    if classification not in valid_classifications:
        classification = "general"
    
    state["classification"] = classification
    return state

# Node 2a: Handle general chat
def handle_general_chat(state: AgentState) -> AgentState:
    """Handle general conversation without RAG"""
    messages = state["messages"]

    system_prompt = """You are a helpful AI assistant for Aurora Works.
    We sell three mechanical keyboards:
    - Aurora Works Atlas 108 - Premium full-size mechanical keyboard ($899.99)
    - Aurora Works Nova 75 - Compact 75% mechanical keyboard ($649.99)
    - Aurora Works Zephyr 87 - Tenkeyless mechanical keyboard ($749.99)
    
    Be friendly and conversational. Ask if there's anything specific they'd like help with."""

    response = llm.invoke([HumanMessage(content=system_prompt), *messages])
    state["final_response"] = response.content
    return state

# Node 2b: Handle product questions with RAG
def handle_product_question(state: AgentState) -> AgentState:
    """Use RAG to answer product related questions"""
    messages = state["messages"]
    query = messages[-1].content

    # Get rag pipeline instance
    rag = get_rag_pipeline()

    # retrieve relevant context
    context = rag.get_context_for_query(query, k=3)
    state["rag_context"] = context

    # Generate response using context
    rag_prompt = f"""You are a helpful product support assistant.
    Use the following context to answer the user's question.
    
    Context from documentation:
    {context}
    
    User question: {query}
    
    Provide a clear answer based on the documentation."""

    response = llm.invoke([HumanMessage(content=rag_prompt)])
    state["final_response"] = response.content

    return state

# Node 2c: check inventory
def check_inventory(state: AgentState) -> AgentState:
    "Inventory checking node with user interaction to ensure api params are received through chat w. user"
    messages = state["messages"]
    last_message = messages[-1].content

    # Check if this is a follow-up with product info
    extract_prompt = f"""
    Extract the product name from the conversation. Look for:
    - Atlas 108 (or Atlas)
    - Nova 75 (or Nova)  
    - Zephyr 87 (or Zephyr)
    
    Recent messages:
    {' '.join([m.content for m in messages[-2:]])}
    
    If you can identify the product, respond with JUST the product name.
    If not, respond with "ASK_PRODUCT".
    """

    # pass extraction prompt into LLM call
    extraction = llm.invoke([HumanMessage(content=extract_prompt)])
    product_identified = extraction.content.strip()

    if product_identified == "ASK_PRODUCT":
        # Ask user to specify the product
        response = """Which product would you like to check inventory for?
        
• Aurora Works Atlas 108 (Full-size)
• Aurora Works Nova 75 (75% compact)
• Aurora Works Zephyr 87 (Tenkeyless)

Just tell me the product name or number."""

        state["final_response"] = response
    else:
        # call the inventory api
        inventory_result = check_inventory_api.invoke({"product_name": product_identified})

        if "error" in inventory_result:
            response = f"I couldn't check inventory: {inventory_result['error']}\n\nWould you like to try again?"
        else:
            # Format the inventory response nicely
            response = f"""**{inventory_result['product_name']}** Inventory:

• Available: {inventory_result['available_quantity']} units
• Price: ${inventory_result['price_per_unit']}
• Location: {inventory_result['warehouse_location']}
• Status: {inventory_result['status'].replace('_', ' ').title()}

Would you like to place an order for this product?"""

        state["final_response"] = response

    return state

# Node 2d: place order
def place_order(state: AgentState) -> AgentState:
    """Handle order placement with user interaction"""
    messages = state["messages"]

    # Extract order information from the user's message
    extract_prompt = f"""
    Extract order information from the conversation:
    1. Product name (Atlas 108, Nova 75, or Zephyr 87)
    2. Quantity (number)
    3. Email (if provided)
    
    Full conversation:
    {' '.join([m.content for m in messages[-4:]])}
    
    Respond in JSON: {{"product": "name_or_null", "quantity": number_or_null, "email": "email_or_null"}}
    """

    # make llm call with extraction prompt
    extraction = llm.invoke([HumanMessage(content=extract_prompt)])

    try:
        order_info = json.loads(extraction.content.strip())
    except:
        order_info = {"product": None, "quantity": None, "email": None}

    # check what information is missing
    missing_info = []
    if not order_info.get("product"):
        missing_info.append("product")
    if not order_info.get("quantity"):
        missing_info.append("quantity")

    if missing_info:
        # ask for missing info
        response = "I'd be happy to help you place an order! I just need a few details:\n\n"

        if "product" in missing_info:
            response += """**Which product?**
• Atlas 108 - $899.99
• Nova 75 - $649.99
• Zephyr 87 - $749.99\n\n"""

        if "quantity" in missing_info:
            response += "**How many units?**\n\n"

        response += "Please provide these details."
        state["final_response"] = response
    else:
        # all required info gathered, place order
        email = order_info.get("email", "customer@example.com")
        order_result = place_order_api.invoke({
            "product_name": order_info["product"],
            "quantity": order_info["quantity"],
            "customer_email": email
        })

        if "error" in order_result:
            response = f"Order failed: {order_result['error']}\n\nWould you like to try again?"
        else:
            # format the order confirmation nicely
            response = f"""**Order Confirmed!**

Here are your order details:
- **Order ID**: {order_result['order_id']}
- **Product**: {order_result['product_name']}
- **Quantity**: {order_result['quantity']} units
- **Total Price**: ${order_result['total_price']}
- **Estimated Delivery**: {order_result['estimated_delivery']}

Your order has been successfully placed! You should receive a confirmation email shortly.

Is there anything else I can help you with today?"""

        state["final_response"] = response

    return state

# Define routing logic
def route_after_classification(state: AgentState) -> Literal["handle_general_chat", "handle_product_question", "check_inventory", "place_order"]:
    """Routing function: Decides which node to go to after classification."""
    
    classification_map = {
            "general": "handle_general_chat",
            "product_question": "handle_product_question", 
            "check_inventory": "check_inventory",
            "place_order": "place_order"
        }

    return classification_map.get(state["classification"], "handle_general_chat")

# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent."""
    
    # Initialize the graph with our state type
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("classify_query", classify_query)
    graph.add_node("handle_general_chat", handle_general_chat)
    graph.add_node("handle_product_question", handle_product_question)
    graph.add_node("check_inventory", check_inventory)
    graph.add_node("place_order", place_order)

    # Define the edges between nodes
    graph.set_entry_point("classify_query")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "handle_general_chat": "handle_general_chat",
            "handle_product_question": "handle_product_question",
            "check_inventory": "check_inventory",
            "place_order": "place_order"
        }
    )

    # Handle each chat message as discrete event - end after each node
    graph.add_edge("handle_general_chat", END)
    graph.add_edge("handle_product_question", END)
    graph.add_edge("check_inventory", END)
    graph.add_edge("place_order", END)

    # Compile the graph
    return graph.compile()

# Create a single instance of the agent
agent = create_agent_graph()

# Helper function for easy invocation
async def process_query(user_query: str, conversation_history: List[Dict] = None):
    """Process a user query through the agent."""
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