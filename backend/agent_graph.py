from itertools import product
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
    continue_conversation: bool # track if agent should continue convo or not

# Tool definitions - functions the agent can call
@tool
def check_inventory_api(product_name: str) -> Dict:
    """Check inventory level for a specific product - atlas 108, nova 75, zephyr 87"""
    try:
        # call inventory api endpoint
        response = requests.get(f"http://localhost:8001/inventory/{product_name}")

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

        response = requests.post("http://localhost:8081/order", json=order_data)

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
    
    3. "check_inventory" - User wants to check stock/inventory/availability of a product.
       Keywords: stock, inventory, available, in stock, how many
    
    4. "place_order" - User wants to buy/purchase/order a product.
       Keywords: buy, order, purchase, want to get
    
    5. "end_conversation" - User indicates they're done or don't need more help.
       Keywords: that's all, thank you, bye, done, no more questions
    
    User query: "{last_message}"
    
    Respond with ONLY the category name (either "product_question" or "general").
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # Validate the classification, default to general
    valid_classifications = ["product_question", "general", "check_inventory", "place_order", "end_conversation"]
    if classification not in valid_classifications:
        classification = "general"
    
    state["classification"] = classification
    
    # set continue flag based on classification - since this is bool
    state["continue_conversation"] = classification != "end_conversation"

    return state

# Node 2a: Handle general chat
def handle_general_chat(state: AgentState) -> AgentState:
    """Handle general conversation without RAG"""
    messages = state["messages"]

    system_prompt = """You are a helpful AI assistant for Aurora Works.
    We sell three mechanical keyboards:
    - Aurora Works Atlas 108 - Premium full-size mechanical keyboard
    - Aurora Works Nova 75 - Compact 75% mechanical keyboard  
    - Aurora Works Zephyr 87 - Tenkeyless mechanical keyboard
    
    Provide friendly, informative responses. Be conversational but concise.
    After responding, ask if there's anything else you can help with."""

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

# Node 2c: check inventory
def check_inventory(state: AgentState) -> AgentState:
    "Inventory checking node with user interaction to ensure api params are received through chat w. user"
    messages = state["messages"]
    last_message = messages[-1].content

    # Extract product information from user's message
    extract_prompt = f"""
    The user wants to check inventory. Extract the product name from their message.
    
    Our products are:
    - Atlas 108 (or just Atlas)
    - Nova 75 (or just Nova)
    - Zephyr 87 (or just Zephyr)
    
    User message: "{last_message}"
    
    If you can identify the product, respond with JUST the product name (e.g., "Atlas 108").
    If you cannot identify the product, respond with "ASK_PRODUCT".
    """

    # pass extraction prompt into LLM call
    extraction = llm.invoke([HumanMessage(content=extract_prompt)])
    product_identified = extraction.content.strip()

    if product_identified == "ASK_PRODUCT":
        # Ask user to specify the product
        response = """I'd be happy to check inventory for you! Which product are you interested in?
        
We have three models available:
- Aurora Works Atlas 108 (Full-size keyboard)
- Aurora Works Nova 75 (Compact 75% keyboard)
- Aurora Works Zephyr 87 (Tenkeyless keyboard)

Please specify which one you'd like to check."""

        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))
    else:
        # call the inventory api
        inventory_result = check_inventory_api.invoke({"product_name": product_identified})

        if "error" in inventory_result:
            response = f"I encountered an issue checking inventory: {inventory_result['error']}\n\nWould you like me to try again or help you with something else?"
        else:
            # Format the inventory response nicely
            response = f"""Here's the current inventory for {inventory_result['product_name']}:

📦 **Available Quantity**: {inventory_result['available_quantity']} units
💰 **Price**: ${inventory_result['price_per_unit']}
📍 **Warehouse**: {inventory_result['warehouse_location']}
🔖 **Status**: {inventory_result['status'].replace('_', ' ').title()}

Would you like to place an order for this product, or can I help you with anything else?"""

        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))

    return state

# Node 2d: place order
def place_order(state: AgentState) -> AgentState:
    """Handle order placement with user interaction"""
    messages = state["messages"]
    last_message = messages[-1].content

    # Extract order information from the user's message
    extract_prompt = f"""
    The user wants to place an order. Extract the following information from their message:
    1. Product name (Atlas 108, Nova 75, or Zephyr 87)
    2. Quantity (number of units)
    3. Email address (if provided)
    
    User message: "{last_message}"
    
    Respond in JSON format like this:
    {{"product": "product_name_or_null", "quantity": number_or_null, "email": "email_or_null"}}
    
    Use null for any information not provided.
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
            response += """**Which product would you like to order?**
- Aurora Works Atlas 108 (Full-size keyboard) - $899.99
- Aurora Works Nova 75 (Compact 75% keyboard) - $649.99
- Aurora Works Zephyr 87 (Tenkeyless keyboard) - $749.99\n\n"""

        if "quantity" in missing_info:
            response += "**How many units would you like to order?**\n\n"

        if not order_info.get("email"):
            response += "**What email address should I use for the order?** (Optional - press enter to skip)\n"
        
        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))
    else:
        # all required info gathered, place order
        email = order_info.get("email", "customer@example.com")

        order_result = place_order_api.invoke({
            "product_name": order_info["product"],
            "quantity": order_info["quantity"],
            "customer_email": email
        })

        if "error" in order_result:
            response = f"I encountered an issue placing your order: {order_result['error']}\n\nWould you like me to try again with different details?"
        else:
            # format the order confirmation nicely
            response = f"""✅ **Order Confirmed!**

Here are your order details:
- **Order ID**: {order_result['order_id']}
- **Product**: {order_result['product_name']}
- **Quantity**: {order_result['quantity']} units
- **Total Price**: ${order_result['total_price']}
- **Status**: {order_result['status'].title()}
- **Estimated Delivery**: {order_result['estimated_delivery']}

Your order has been successfully placed! You should receive a confirmation email shortly.

Is there anything else I can help you with today?"""

        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))

    return state


# Node 2e: Handle end of conversation
def handle_end_conversation(state: AgentState) -> AgentState:
    """handle when user wants to end conversation"""
    response = "Thank you for choosing Aurora Works! Have a great day, and feel free to come back anytime if you need assistance with our mechanical keyboards. 👋"
    state["final_response"] = response
    state["messages"].append(AIMessage(content=response))
    state["continue_conversation"] = False
    return state

# Define routing logic
def route_after_classification(state: AgentState) -> Literal["handle_general_chat", "handle_product_question", "check_inventory", "place_order", "handle_end_conversation"]:
    """Routing function: Decides which node to go to after classification."""
    
    classification_map = {
            "general": "handle_general_chat",
            "product_question": "handle_product_question", 
            "check_inventory": "check_inventory",
            "place_order": "place_order",
            "end_conversation": "handle_end_conversation"
        }

    return classification_map.get(state["classification"], "handle_general_chat")

# Define whether to continue or end
def should_continue(state: AgentState) -> Literal["classify_query", END]:
    """Decide whether to continue conversation or end it"""
    if state.get("continue_conversation", True):
        return "classify_query"
    else:
        return END

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
    graph.add_node("handle_end_conversation", handle_end_conversation)

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
            "place_order": "place_order",
            "handle_end_conversation": "handle_end_conversation"
        }
    )

    # Add looping logic - each handler goes back to continue / end decision
    graph.add_conditional_edges("handle_general_chat", should_continue)
    graph.add_conditional_edges("handle_product_question", should_continue)
    graph.add_conditional_edges("check_inventory", should_continue)
    graph.add_conditional_edges("place_order", should_continue)

    # End conversation node goes straight to END
    graph.add_edge("handle_end_conversation", END)

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
        "final_response": "",
        "continue_conversation": True
    }

    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    return result["final_response"]