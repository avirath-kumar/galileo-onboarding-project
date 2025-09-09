import streamlit as st
import requests
import json

st.set_page_config(page_title="Aurora Works Product Agent", page_icon="💻", layout="wide")

# Helper function for backend health check
def check_backend_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Helper function for chat with backend
def send_message(message):
    try:
        response = requests.post("http://localhost:8000/chat", json={"message": message}, timeout=60)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Main frontend page function
def main():
    st.title("💻 Aurora Works Product Agent")
    
    if check_backend_health():
        st.success("Backend is connected")
    else:
        st.error("Backend is not available. Make sure the FastAPI server is running.")
    
    st.markdown("---")
    
    st.header("Ask product questions here:")
    
    user_input = st.text_input("Enter your message:")
    
    if st.button("Send") and user_input:
        with st.spinner("Getting response..."):
            response = send_message(user_input)

        st.write(f"**You:** {user_input}")
        st.write(f"**Agent**: {response}")

if __name__ == "__main__":
    main()