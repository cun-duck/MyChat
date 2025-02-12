import streamlit as st
from utils.crawl4ai_wrapper import crawl_url
from utils.hf_inference import generate_response
import json
import os
import time

# Judul aplikasi
st.set_page_config(page_title="Customizable Chatbot", page_icon="🤖", layout="wide")
st.title("Customizable Chatbot 🤖")
st.markdown("Chatbot ini menggunakan model AI dari Hugging Face dan dapat dikostumasi dengan data R.A.G.")

# Sidebar untuk konfigurasi
st.sidebar.header("Konfigurasi")
hf_token = st.sidebar.text_input("Masukkan Hugging Face Token:", type="password", placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx")
model_options = ["Qwen/Qwen2.5-Coder-32B-Instruct", "Model Lainnya"]
selected_model = st.sidebar.selectbox("Pilih Model AI:", model_options)

# Input URL untuk kostumasi data R.A.G
st.subheader("Step 1: Masukkan URL untuk Crawling Data")
url = st.text_input("Masukkan URL:", placeholder="https://example.com")

if st.button("Crawl Data"):
    if url:
        with st.spinner("Mencrawling data..."):
            try:
                # Crawling data menggunakan crawl4ai
                crawled_data = crawl_url(url)
                # Simpan data ke file JSON
                os.makedirs("data", exist_ok=True)
                with open("data/crawled_data.json", "w") as f:
                    json.dump(craw_data, f)
                st.success("Data berhasil dicrawl dan disimpan!")
            except Exception as e:
                st.error(f"Gagal mencrawling data: {e}")
    else:
        st.warning("Silakan masukkan URL terlebih dahulu.")

# Input prompt manual
st.subheader("Step 2: Optimasi Prompt Manual")
custom_prompt = st.text_area("Edit prompt manual (opsional):", value="Jawab pertanyaan berdasarkan konteks yang diberikan.")

# Interaksi dengan chatbot
st.subheader("Step 3: Mulai Berinteraksi dengan Chatbot")
user_question = st.text_input("Tanyakan sesuatu kepada chatbot:")

if st.button("Kirim Pertanyaan"):
    if user_question and hf_token:
        with st.spinner("Memproses respons..."):
            try:
                # Load data R.A.G
                if os.path.exists("data/crawled_data.json"):
                    with open("data/crawled_data.json", "r") as f:
                        rag_data = json.load(f)
                    context = " ".join(rag_data.get("text", []))
                else:
                    context = "Tidak ada data R.A.G yang tersedia."

                # Generate respons menggunakan Hugging Face Inference API
                response = generate_response(user_question, context, custom_prompt, hf_token, selected_model)
                
                # Format output jika mengandung script
                if "```" in response:
                    st.markdown("### Output Script:")
                    st.code(response.split("```")[1], language="python")
                else:
                    st.success("Respons chatbot:")
                    st.write(response)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan masukkan pertanyaan dan token Hugging Face terlebih dahulu.")

# Hapus data setelah 2 menit tidak aktif
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

if time.time() - st.session_state.last_activity > 120:  # 2 menit
    if os.path.exists("data/crawled_data.json"):
        os.remove("data/crawled_data.json")
    st.session_state.clear()
else:
    st.session_state.last_activity = time.time()
