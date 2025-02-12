import streamlit as st
from utils.hf_inference import generate_response
from utils.pdf_extractor import extract_text_from_pdf, split_text_into_chunks
from utils.pdf_extractor import get_relevant_chunks
import json
import os
import time

# Judul aplikasi
st.set_page_config(page_title="Customizable Chatbot", page_icon="🤖", layout="wide")
st.title("Customizable Chatbot 👾")
st.markdown("This chatbot with Customizable R.A.G data.")

# Sidebar for configuration
st.sidebar.header("Configuration")
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password", placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx")
model_options = ["Qwen/Qwen2.5-Coder-32B-Instruct", "Other Model"]
selected_model = st.sidebar.selectbox("Select AI Model:", model_options)

# Input file PDF for R.A.G customization
uploaded_file = st.sidebar.file_uploader("Upload a PDF file for R.A.G data:", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        try:
            # Extract text from the uploaded PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Split text into chunks
            chunks = split_text_into_chunks(pdf_text, chunk_size=500, overlap=100)
            
            # Save chunks to JSON file
            os.makedirs("data", exist_ok=True)
            with open("data/pdf_data.json", "w") as f:
                json.dump({"chunks": chunks}, f)
            st.sidebar.success(f"PDF text successfully extracted and split into {len(chunks)} chunks!")
        except Exception as e:
            st.sidebar.error(f"Failed to process PDF: {e}")

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
if os.path.exists("data/pdf_data.json"):
    with open("data/pdf_data.json", "r") as f:
        rag_data = json.load(f)
    chunks = rag_data.get("chunks", [])
    context = " ".join(chunks) if chunks else default_context
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
                # Load chunks from JSON
                if os.path.exists("data/pdf_data.json"):
                    with open("data/pdf_data.json", "r") as f:
                        rag_data = json.load(f)
                    chunks = rag_data.get("chunks", [])
                else:
                    chunks = []

                # Get relevant chunks based on the question
                relevant_chunks = get_relevant_chunks(user_question, chunks, top_n=3)
                context = " ".join(relevant_chunks) if relevant_chunks else default_context

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
    if os.path.exists("data/pdf_data.json"):
        os.remove("data/pdf_data.json")
    st.session_state.clear()
else:
    st.session_state.last_activity = time.time()
