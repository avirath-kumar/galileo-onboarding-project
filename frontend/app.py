import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

st.set_page_config(page_title="Aurora Works Product Agent", page_icon="💻", layout="wide")

# Initialize session state for the conversation
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
            # Try to get error details from response
            try:
                error_detail = response.json().get("detail", f"Status code: {response.status_code}")
            except:
                error_detail = f"Status code: {response.status_code}"
            return {"error": error_detail}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Make sure it's running on port 8000."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The backend might be processing a heavy request."}
    except Exception as e:
        return {"error": str(e)}

# Main frontend page function
def main():
    
    st.title("💻 Aurora Works Product Agent")
    
    # Sidebar for session mgmt
    with st.sidebar:
        st.header("Conversation")

        # New conversation button
        if st.button("New Conversation"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
        
        # Session info
        if st.session_state.session_id:
            st.success(f"Session: {st.session_state.session_id[:8]}...")
        else:
            st.info("Start a conversation to create a session")

        st.divider()

        # Backend status
        if check_backend_health():
            st.success("Backend connected")
        else:
            st.error("Backend not available")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about our products, check inventory, or place an order..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_message(prompt, st.session_state.session_id)

                # Check if response is an error (string) or success (dict)
                if isinstance(response_data, dict) and "error" not in response_data:
                    # Update session id if new
                    if not st.session_state.session_id:
                        st.session_state.session_id = response_data["session_id"]

                    # Display and store response
                    response_text = response_data["response"]
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Handle error case
                    error_msg = response_data.get("error", response_data) if isinstance(response_data, dict) else response_data
                    st.error(f"Error: {error_msg}")

if __name__ == "__main__":
    main()