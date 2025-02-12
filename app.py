import streamlit as st
from utils.hf_inference import generate_response
from utils.pdf_extractor import extract_text_from_pdf, split_text_into_chunks, get_relevant_chunks
from utils.model_utils import load_model_options
import json
import os
import time

# Judul aplikasi dan konfigurasi halaman
st.set_page_config(page_title="Customizable Chatbot", page_icon="🤖", layout="wide")

# Load custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

# Load Google Font
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Judul aplikasi (tetap di bagian atas)
st.title("Customizable Chatbot 👾")
st.markdown("This chatbot uses AI models from Hugging Face and can be customized with R.A.G data.")

# Sidebar for configuration
st.sidebar.header("Configuration")
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password", placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx")

# Load model options
model_options = load_model_options()
selected_model = st.sidebar.selectbox("Select AI Model:", model_options)

# Input file PDF for R.A.G customization (optional)
uploaded_file = st.sidebar.file_uploader("Upload a PDF file for R.A.G data (Optional):", type=["pdf"])

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
else:
    # Use default context if no PDF is uploaded
    chunks = []

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
This is a general-purpose chatbot that can answer questions about technology, programming, AI, and other topics.
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

# Ensure context and prompt are always used
context = context or default_context
prompt_to_use = prompt_to_use or default_prompt

# Create two columns: one for chat and one for feedback
chat_col, feedback_col = st.columns([4, 1])

with chat_col:
    # Frame khusus untuk chat
    st.markdown('<div class="chat-frame">', unsafe_allow_html=True)

    # Placeholder for conversation history (dinamis)
    chat_container = st.container()

    # Display conversation history
    with chat_container:
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for message in st.session_state.conversation:
            if message["role"] == "user":
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.write("👤")  # Avatar pengguna
                with col2:
                    st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "assistant":
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.write("🤖")  # Avatar bot
                with col2:
                    st.markdown(f'<div class="bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Text input area (tetap di bagian bawah)
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    user_question = st.chat_input("Ask something to the chatbot:")
    st.markdown('</div>', unsafe_allow_html=True)

    if user_question:
        if hf_token:
            with st.spinner("Processing response..."):
                try:
                    # Get relevant chunks based on the question (fallback to default context if no chunks)
                    if chunks:
                        try:
                            relevant_chunks = get_relevant_chunks(user_question, chunks, top_n=3)
                            if relevant_chunks:
                                context = " ".join(relevant_chunks)
                            else:
                                context = default_context
                        except Exception as e:
                            context = default_context
                    else:
                        context = default_context

                    # Generate response using Hugging Face Inference API
                    response = generate_response(user_question, context, prompt_to_use, hf_token, selected_model)

                    # Add to conversation history
                    st.session_state.conversation.append({"role": "user", "content": user_question})
                    st.session_state.conversation.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter your Hugging Face token in the sidebar.")

    st.markdown('</div>', unsafe_allow_html=True)  # Akhiri frame chat

with feedback_col:
    # Feedback section (kolom kecil di kanan)
    st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
    st.subheader("Feedback")
    if chunks:
        if relevant_chunks:
            st.success("Model is using data from the uploaded PDF.")
            st.markdown("**Relevant Context:**")
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")
        else:
            st.warning("No relevant data found in the uploaded PDF. Using default context.")
    else:
        st.info("No PDF uploaded. Using default context.")
    st.markdown('</div>', unsafe_allow_html=True)

# Delete data after 2 minutes of inactivity
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

if time.time() - st.session_state.last_activity > 120:  # 2 minutes
    if os.path.exists("data/pdf_data.json"):
        os.remove("data/pdf_data.json")
    st.session_state.clear()
else:
    st.session_state.last_activity = time.time()
