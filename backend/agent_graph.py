from itertools import product
from typing import Optional, TypedDict, Annotated, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import operator
from dotenv import load_dotenv
import os
import requests
import json
import re
import uuid
from galileo.handlers.langchain import GalileoAsyncCallback
from galileo import galileo_context

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
    # additional state parameters to help track multi turn interactions
    current_action: Optional[str]
    collected_info: Optional[Dict] # store collected order / inventory info
    awaiting_info: Optional[List[str]] # track which info we're waiting for

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

# Helper function to normalize product names
def normalize_product_name(text: str) -> Optional[str]:
    """Extract and normalize product name from text"""
    text_lower = text.lower()

    # map common variations to standard names
    if "atlas" in text_lower or "108" in text_lower:
        return "atlas-108"
    elif "nova" in text_lower or "75" in text_lower:
        return "nova-75"
    elif "zephyr" in text_lower or "87" in text_lower:
        return "zephyr-87"
    
    return None

# Helper function to extract quantity
def extract_quantity(text: str) -> Optional[int]:
    """Extract quantity from text"""
    # Look for patterns like "2", "two", "2 units", etc
    patterns = [
        r'\b(\d+)\s*(?:units?|items?|pieces?)?\b',
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b'
    ]

    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            value = match.group(1)
            if value.isdigit():
                return int(value)
            elif value in number_words:
                return number_words[value]

    return None

# Node functions - these get called when the graph executes

# Node 1: Classify the query
def classify_query(state: AgentState) -> AgentState:
    """Classify whether the query is general or needs RAG"""
    messages = state["messages"]
    last_message = messages[-1].content

    # check if we have an ongoing action from previous state
    current_action = state.get("current_action")
    collected_info = state.get("collected_info", {})

    # if we're in the middle of an action, check if user is providing requested info
    if current_action in ["place_order", "check_inventory"]:
        # check if message contains info we're looking for
        product = normalize_product_name(last_message)
        quantity = extract_quantity(last_message)

        if product or quantity:
            # user is providing info for ongoing action
            state["classification"] = current_action

            # update collected info
            if product:
                collected_info["product"] = product
            if quantity:
                collected_info["quantity"] = quantity
            state["collected_info"] = collected_info

            # return early to avoid re-classification
            return state

    # otherwise, do fresh classification
    conversation_context = "\n".join([m.content for m in messages[-3:]]) # last 3 messages for context

    classification_prompt = f"""
    Classify the user's intent based on this message: "{last_message}"

    Categories:
    1. "product_question" - Questions about product specs, features, documentation, troubleshooting
    2. "check_inventory" - User wants to check stock/availability
    3. "place_order" - User wants to buy/purchase/order a product
    4. "general" - Greetings, general chat, other topics

    Look for keywords:
    - Inventory/stock/available/in stock → check_inventory
    - Buy/order/purchase/want to get → place_order
    - Specs/features/documentation/how does/troubleshooting → product_question

    Recent context: {conversation_context}

    Respond with ONLY the category name.
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # Validate the classification, default to general
    valid_classifications = ["product_question", "general", "check_inventory", "place_order"]
    if classification not in valid_classifications:
        classification = "general"

    state["classification"] = classification
    
    # reset action tracking for new intents
    if classification in ["check_inventory", "place_order"]:
        state["current_action"] = classification
        state["collected_info"] = {}
        state["awaiting_info"] = []
    else:
        state["current_action"] = None
        state["collected_info"] = {}
        state["awaiting_info"] = []
    
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
    
    Be friendly and conversational. You can help with:
    - Answering product questions
    - Checking inventory
    - Placing orders
    
    Ask the user what they would like help with today"""
    
    response = llm.invoke([HumanMessage(content=system_prompt), *messages])
    state["final_response"] = response.content
    
    # clear any ongoing action
    state["current_action"] = None
    state["collected_info"] = {}
    
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
    rag_prompt = f"""You are a helpful product support assistant for Aurora Works.
    Use the following context to answer the user's question.
    
    Context from documentation:
    {context}
    
    User question: {query}
    
    Provide a clear, helpful answer based on the documentation.
    If you mention a product, you can also offer to check inventory or help place an order."""
    
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    state["final_response"] = response.content

    # Clear any ongoing action
    state["current_action"] = None
    state["collected_info"] = {}

    return state

# Node 2c: check inventory
def check_inventory(state: AgentState) -> AgentState:
    "Inventory checking node with user interaction to ensure api params are received through chat w. user"
    messages = state["messages"]
    last_message = messages[-1].content
    collected_info = state.get("collected_info", {})

    # try to extract product from current message if not already collected
    if "product" not in collected_info:
        product = normalize_product_name(last_message)
        if product:
            collected_info["product"] = product
    
    # check convo history for product mentions
    if "product" not in collected_info:
        for msg in messages[-3:]:
            product = normalize_product_name(msg.content)
            if product:
                collected_info["product"] = product
                break
    
    state["collected_info"] = collected_info

    if "product" not in collected_info:
        # ask for product specification
        response = """Which product would you like to check inventory for?
        
• **Atlas 108** - Full-size mechanical keyboard
• **Nova 75** - 75% compact keyboard  
• **Zephyr 87** - Tenkeyless keyboard

Just tell me the product name or model number."""

        state["final_response"] = response
        state["current_action"] = "check_inventory"
    else:
        # we have the product, check inventory
        product_name = collected_info["product"]
        inventory_result = check_inventory_api.invoke({"product_name": product_name})

        if "error" in inventory_result:
            response = f"I couldn't check inventory: {inventory_result['error']}\n\nWould you like to try again?"
            state["current_action"] = None
        else:
            # format the inventory response
            response = f"""**{inventory_result['product_name']}** Inventory Status:

• **Available:** {inventory_result['available_quantity']} units
• **Price:** ${inventory_result['price_per_unit']}
• **Warehouse:** {inventory_result['warehouse_location']}
• **Status:** {inventory_result['status'].replace('_', ' ').title()}

Would you like to place an order for this product? Just let me know how many units you'd like."""

            # set up for potential order
            state["current_action"] = "place_order"
            state["collected_info"] = {"product": product_name} # carry forward product info

        state["final_response"] = response
    
    return state

# Node 2d: place order
def place_order(state: AgentState) -> AgentState:
    """Handle order placement with user interaction"""
    messages = state["messages"]
    last_message = messages[-1].content
    collected_info = state.get("collected_info", {})

    # try to extract info from current message
    if "product" not in collected_info:
        product = normalize_product_name(last_message)
        if product:
            collected_info["product"] = product
    
    if "quantity" not in collected_info:
        quantity = extract_quantity(last_message)
        if quantity:
            collected_info["quantity"] = quantity

    # check recent convo for missed info
    if "product" not in collected_info or "quantity" not in collected_info:
        for msg in messages[-3:]:
            if "product" not in collected_info:
                product = normalize_product_name(msg.content)
                if product:
                    collected_info["product"] = product
            
            if "quantity" not in collected_info:
                quantity = extract_quantity(msg.content)
                if quantity:
                    collected_info["quantity"] = quantity

    # extract email if provided
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, last_message)
    if email_match:
        collected_info["email"] = email_match.group(0)
    
    # update state with any new info found
    state["collected_info"] = collected_info
    
    # check what information is missing
    missing_info = []
    if "product" not in collected_info:
        missing_info.append("product")
    if "quantity" not in collected_info:
        missing_info.append("quantity")
    
    if missing_info:
        # build contextual request for missing info
        response = "I'd be happy to help you place your order! "
        
        if "product" in collected_info:
            product_display = collected_info["product"].replace("-", " ").title()
            response += f"You've selected the **{product_display}**. "

        if "quantity" in collected_info:
            response += f"Quantity: **{collected_info['quantity']} units**. "

        response += "\n\nI just need"

        if "product" in missing_info and "quantity" in missing_info:
            response += """ to know:

**1. Which product?**
• Atlas 108 ($899.99)
• Nova 75 ($649.99)
• Zephyr 87 ($749.99)

**2. How many units?**

Please provide these details."""

        elif "product" in missing_info:
            response += """ to know which product you'd like to order:

• Atlas 108 ($899.99)
• Nova 75 ($649.99)
• Zephyr 87 ($749.99)"""

        elif "quantity" in missing_info:
            response += " to know **how many units** you'd like to order."
        state["final_response"] = response
        state["current_action"] = "place_order"
    else:
        # We have all required info, place the order
        email = collected_info.get("email", "customer@example.com")
        
        order_result = place_order_api.invoke({
            "product_name": collected_info["product"],
            "quantity": collected_info["quantity"],
            "customer_email": email
        })
        
        if "error" in order_result:
            response = f"I encountered an issue placing your order: {order_result['error']}\n\nWould you like to try again?"
            state["current_action"] = None
            state["collected_info"] = {}
        else:
            # Format the order confirmation
            response = f"""**Order Confirmed!**

**Order Details:**
━━━━━━━━━━━━━━━━━━━━━
• **Order ID:** `{order_result['order_id']}`
• **Product:** {order_result['product_name']}
• **Quantity:** {order_result['quantity']} units
• **Total Price:** ${order_result['total_price']}
• **Estimated Delivery:** {order_result['estimated_delivery']}
━━━━━━━━━━━━━━━━━━━━━

Your order has been successfully placed! You'll receive a confirmation email shortly.

Is there anything else I can help you with today?"""
            
            # Clear state after successful order
            state["current_action"] = None
            state["collected_info"] = {}
        
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

# Helper function for easy invocation. callbacks: List of callbacks for monitoring
async def process_query(
    user_query: str,
    conversation_history: List[Dict] = None,
    session_id: str = None,
    galileo_session_id: str = None,
    callbacks: List = None
):
    """Process a user query through the agent."""

    # use session_id as thread_id - if none, create one
    thread_id = session_id or str(uuid.uuid4())

    # Create galileo callback, define config, consistent thread_id
    galileo_callback = GalileoAsyncCallback()
    config = {
        "callbacks": [galileo_callback],
        "configurable": {
            "thread_id": thread_id
        }
    }

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

    # Try to recover state from conversation history
    current_action = None
    collected_info = {}

    # check if last assistant message indicates we're waiting for info
    if conversation_history and len(conversation_history) > 0:
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break

        if last_assistant_msg:
            # Check for patterns indicating ongoing actions
            if "which product" in last_assistant_msg.lower() and "inventory" in last_assistant_msg.lower():
                current_action = "check_inventory"
            elif "which product" in last_assistant_msg.lower() and "order" in last_assistant_msg.lower():
                current_action = "place_order"
            elif "how many units" in last_assistant_msg.lower():
                current_action = "place_order"
                # Try to find product from history
                for msg in conversation_history[-4:]:
                    product = normalize_product_name(msg["content"])
                    if product:
                        collected_info["product"] = product
                        break

    # Create initial state
    initial_state = {
        "messages": messages,
        "classification": "",
        "rag_context": "",
        "final_response": "",
        "current_action": current_action,
        "collected_info": collected_info,
        "awaiting_info": []
    }

    # Run the agent
    result = await agent.ainvoke(initial_state, config) # callback passed in through config

    return result["final_response"]