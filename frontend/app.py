import streamlit as st
import requests

st.set_page_config(page_title="Agent Frontend", page_icon="🤖", layout="wide")

def check_backend_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    st.title("🤖 Agent Frontend")
    
    if check_backend_health():
        st.success("Backend is connected")
    else:
        st.error("Backend is not available. Make sure the FastAPI server is running.")
    
    st.markdown("---")
    
    st.header("Agent Interface")
    
    user_input = st.text_input("Enter your message:")
    
    if st.button("Send") and user_input:
        st.write(f"You said: {user_input}")

if __name__ == "__main__":
    main()