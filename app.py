# app.py

import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- KONFIGURASI AWAL & PEMUATAN DATA ---

# Muat environment variables dari file .env
load_dotenv()

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="SABALING",
    page_icon="🤖",
    layout="centered"
)

# --- ANDA SEBAGAI PENGEMBANG, KONFIGURASI FILE PDF DI SINI ---
PDF_FILE_PATH = "data/Modul-Kesehatan-Mental.pdf"

@st.cache_resource
def load_and_process_data(file_path):
    """
    Fungsi untuk memuat, memproses PDF, dan membuat embedding.
    Di-cache agar hanya berjalan sekali per sesi.
    """
    # 1. Ekstrak Teks dari PDF
    try:
        with fitz.open(file_path) as doc:
            full_text = "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Gagal membaca file PDF: {e}", icon="🚨")
        return None
    
    if not full_text.strip():
        st.error("Dokumen PDF tidak berisi teks yang dapat diekstrak.", icon="🚨")
        return None

    # 2. Pecah Teks menjadi Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)

    # 3. Buat Embeddings
    try:
        embedding_model = 'models/text-embedding-004'
        result = genai.embed_content(
            model=embedding_model,
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # 4. Buat DataFrame
        df = pd.DataFrame({'chunk_text': chunks, 'embedding': result['embedding']})
        return df
    except Exception as e:
        st.error(f"Gagal membuat embedding: {e}", icon="🚨")
        return None

def find_best_passages(query, dataframe, embedding_model_name):
    """Mencari chunk yang paling relevan dengan query."""
    query_embedding = genai.embed_content(
        model=embedding_model_name,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    all_embeddings = np.stack(dataframe['embedding'].to_numpy())
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_indices]['chunk_text'].tolist())

def generate_rag_response(query, dataframe, rag_model, embedding_model_name):
    """Menghasilkan jawaban dari model RAG."""
    context = find_best_passages(query, dataframe, embedding_model_name)
    prompt = f"""
Anda adalah asisten AI yang ramah dan membantu. Jawab pertanyaan pengguna dengan ringkas dan jelas HANYA berdasarkan pada konteks yang diberikan.
Jika informasi tidak ada dalam konteks, katakan "Maaf, saya tidak memiliki informasi mengenai hal itu."

KONTEKS:
{context}

PERTANYAAN:
{query}

JAWABAN ANDA:
"""
    try:
        response = rag_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi."

# --- Konfigurasi API dan Model ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Kunci API tidak ditemukan. Harap atur di file .env Anda.", icon="🚨")
        st.stop()
    genai.configure(api_key=api_key)
    RAG_MODEL = genai.GenerativeModel('gemini-2.0-flash')
    EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
    
    # Memuat dan memproses data di latar belakang
    df_embeddings = load_and_process_data(PDF_FILE_PATH)
    
except Exception as e:
    st.error(f"Gagal menginisialisasi model AI: {e}", icon="🚨")
    st.stop()


# --- ANTARMUKA CHAT UTAMA ---
st.title("🤖 SABALING.ai")
st.write("Sahabat Konseling anda untuk meningkatkan pengetahuan kesehatan mental.")

# Inisialisasi riwayat chat jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input dari pengguna
if prompt := st.chat_input("Apa yang ingin Anda ketahui?"):
    if df_embeddings is None:
        st.warning("Maaf, chatbot sedang tidak siap. Ada masalah dengan dokumen sumber.", icon="⚠️")
    else:
        # Tambahkan pesan pengguna ke riwayat dan tampilkan
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Hasilkan dan tampilkan respons dari asisten
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                response = generate_rag_response(
                    prompt,
                    df_embeddings,
                    RAG_MODEL,
                    EMBEDDING_MODEL_NAME
                )
                st.markdown(response)
        
        # Tambahkan respons asisten ke riwayat
        st.session_state.messages.append({"role": "assistant", "content": response})