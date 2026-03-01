import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Second Brain",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Second Brain")
st.markdown("Your personal knowledge assistant powered by Endee and RAG.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Verify Backend Connection
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    if health.get("status") == "ok":
        st.sidebar.success("✅ Connected to Backend")
        if health.get("endee_connected"):
            st.sidebar.success("🚀 Endee is Online")
        else:
            st.sidebar.error("❌ Endee is Offline")
    else:
        st.sidebar.error(f"⚠️ Backend Error: {health.get('message')}")
except Exception as e:
    st.sidebar.error("❌ Backend Offline. Please check your terminal.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

# Theme Toggle
# Streamlit uses the system default or the setting from config.toml
# We will create/modify config.toml to force the theme.
config_path = ".streamlit/config.toml"
current_theme = "dark" # Default fallback
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        content = f.read()
        if 'base="light"' in content or "base='light'" in content:
            current_theme = "light"

import time

is_dark = st.sidebar.toggle("🌙 Dark Mode", value=(current_theme == "dark"))
target_theme = "dark" if is_dark else "light"

if target_theme != current_theme:
    os.makedirs(".streamlit", exist_ok=True)
    with open(config_path, "w") as f:
        f.write(f'[theme]\nbase="{target_theme}"\n')
    
    st.toast("Applying theme... Please wait or refresh the page!", icon="⏳")
    time.sleep(1.5) # Give Streamlit watcher time to detect config.toml change
    st.rerun()
st.sidebar.markdown("---")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "💬 Chat"

tab_chat, tab_knowledge = st.tabs(["💬 Chat", "📚 Knowledge Base"])

with tab_knowledge:
    st.header("Upload Documents")
    st.markdown("Upload PDFs, Markdown, or text files to add to your Second Brain.")
    
    uploaded_file = st.file_uploader("Choose a file...", type=["pdf", "txt", "md"])
    if st.button("Upload to Brain"):
        if uploaded_file is not None:
             with st.spinner('Ingesting and embedding document...'):
                 try:
                     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                     response = requests.post(f"{API_URL}/upload", files=files)
                     
                     if response.status_code == 200:
                         data = response.json()
                         st.success(f"Successfully processed `{data['filename']}` into {data['chunks_indexed']} chunks!")
                     else:
                         st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                 except Exception as e:
                     st.error(f"Failed to connect: {e}")
        else:
            st.warning("Please select a file first.")

with tab_chat:
    st.markdown("Ask questions about your uploaded documents.")
    
    # Display chat messages explicitly within the tab container body
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Global Chat Input (Streamlit Requires this outside of columns/tabs) ---
if prompt := st.chat_input("Ask your Second Brain..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Force a rerun to show the user's message immediately within the tab
    st.rerun()

# If the last message is from the user, generate a response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with tab_chat:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # Calling the streaming endpoint
                payload = {"query": st.session_state.messages[-1]["content"]}
                response = requests.post(f"{API_URL}/query_stream", json=payload, stream=True)
                
                if response.status_code == 200:
                    full_response = ""
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_msg = f"Error: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Failed to connect to API: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
