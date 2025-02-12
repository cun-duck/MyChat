import streamlit as st
from utils.crawl4ai_wrapper import crawl_url
from utils.hf_inference import generate_response
import json
import os
import time

# Judul aplikasi
st.set_page_config(page_title="Customizable Chatbot", page_icon="🤖", layout="wide")
st.title("Customizable Chatbot 🤖")
st.markdown("This chatbot uses AI models from Hugging Face and can be customized with R.A.G data.")

# Sidebar for configuration
st.sidebar.header("Configuration")
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password", placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx")
model_options = ["Qwen/Qwen2.5-Coder-32B-Instruct", "Other Model"]
selected_model = st.sidebar.selectbox("Select AI Model:", model_options)

# Input URL for R.A.G customization
url = st.sidebar.text_input("Enter URL to Crawl Data (Optional):", placeholder="https://example.com")

if st.sidebar.button("Crawl Data"):
    if url:
        with st.spinner("Crawling data..."):
            try:
                # Crawling data using crawl4ai
                crawled_data = crawl_url(url)
                # Save data to JSON file
                os.makedirs("data", exist_ok=True)
                with open("data/crawled_data.json", "w") as f:
                    json.dump(crawled_data, f)
                st.sidebar.success("Data successfully crawled and saved!")
            except Exception as e:
                st.sidebar.error(f"Failed to crawl data: {e}")
    else:
        st.sidebar.warning("Please enter a URL first.")

# Manual prompt optimization (optional)
custom_prompt = st.sidebar.text_area(
    "Optimize Prompt Manually (Optional):",
    value="Answer the question based on the provided context."
)

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Default values
default_context = """
This is a default context for the chatbot. You can ask general questions about technology, programming, or AI.
For example:
- What is machine learning?
- How does a neural network work?
- Explain the concept of natural language processing.
"""
default_prompt = """
You are a helpful assistant. If no specific context is provided, answer general questions based on your training data.
If the question is unclear or cannot be answered, politely inform the user.
"""

# Load R.A.G data or use default context
if os.path.exists("data/crawled_data.json"):
    with open("data/crawled_data.json", "r") as f:
        rag_data = json.load(f)
    context = " ".join(rag_data.get("text", [])) or default_context
else:
    context = default_context

# Use custom prompt or default prompt
prompt_to_use = custom_prompt.strip() or default_prompt

# Chat interface
st.subheader("Chat with the Bot")
user_question = st.chat_input("Ask something to the chatbot:")

if user_question:
    if hf_token:
        with st.spinner("Processing response..."):
            try:
                # Generate response using Hugging Face Inference API
                response = generate_response(user_question, context, prompt_to_use, hf_token, selected_model)

                # Add to conversation history
                st.session_state.conversation.append({"role": "user", "content": user_question})
                st.session_state.conversation.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your Hugging Face token in the sidebar.")

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Delete data after 2 minutes of inactivity
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

if time.time() - st.session_state.last_activity > 120:  # 2 minutes
    if os.path.exists("data/crawled_data.json"):
        os.remove("data/crawled_data.json")
    st.session_state.clear()
else:
    st.session_state.last_activity = time.time()
