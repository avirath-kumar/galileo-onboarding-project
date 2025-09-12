import streamlit as st
import requests
import json
import os
import uuid
from dotenv import load_dotenv
from datetime import datetime

from galileo import galileo_context
from galileo.handlers.langchain import GalileoCallback

# Load env vars
load_dotenv()

st.set_page_config(page_title="Aurora Works Product Agent", page_icon="💻", layout="wide")

# initialize session state for the conversation
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function for backend health check
def check_backend_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Helper function for chat with backend
def send_message(message, session_id=None):
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
            
        response = requests.post("http://localhost:8000/chat", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Main frontend page function
def main():
    # Start Galileo session if not already started
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_name = f"Aurora Works - {current_time}"
    galileo_context.start_session(name=session_name, external_id=str(uuid.uuid4()))
    
    st.title("💻 Aurora Works Product Agent")
    
    # Sidebar for session mgmt
    with st.sidebar:
        st.header("Conversation")

        # wipe session id, message history when new convo button pressed
        if st.button("New Conversation"):
            
            # reset session state
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
        
        # session info
        if st.session_state.session_id:
            st.success(f"Session: {st.session_state.session_id[:8]}...")
        else:
            st.info("Start a conversation to create a session")

        st.divider()

        # backend status - might remove this
        if check_backend_health():
            st.success("Backend connected")
        else:
            st.error("Backend not available")
    
    # display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # chat input
    if prompt := st.chat_input("Ask about our products, check inventory, or place an order..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # invoke send_message function which goes to backend
                response_data = send_message(prompt, st.session_state.session_id)

                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    # Update session id if new
                    if not st.session_state.session_id:
                        st.session_state.session_id = response_data["session_id"]

                    # display and store response
                    response_text = response_data["response"]
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()