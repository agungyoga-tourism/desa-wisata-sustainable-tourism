# app.py

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Asisten Riset Desa Wisata", layout="wide")
st.title("🤖 Asisten Riset untuk Scoping Review 'Desa Wisata'")
st.write("Sistem ini menggunakan Retrieval-Augmented Generation (RAG) untuk menjawab pertanyaan hanya berdasarkan korpus riset yang disediakan.")

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI
# ==============================================================================

@st.cache_data
def load_data():
    try:
        corpus_path = 'ScR_TV_Corpus.csv'
        anchors_path = 'ScR_TV_Anchors.csv'
        df_corpus = pd.read_csv(corpus_path)
        df_anchors = pd.read_csv(anchors_path)
        return df_corpus, df_anchors
    except FileNotFoundError:
        st.error("Pastikan file 'ScR_TV_Corpus.csv' dan 'ScR_TV_Anchors.csv' berada di direktori utama repositori GitHub Anda.")
        return None, None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def prepare_and_embed_knowledge_base(_corpus_db, _anchors_db, _model):
    # ... (Fungsi ini tetap sama, tidak perlu diubah) ...
    corpus_cols = ['title', 'abstract', 'key_arguments_findings', 'critical_comments_linkages', 'outcome_notes', 'notes']
    anchor_cols = ['citation_full', 'abstract', 'verbatim_definition', 'typology_summary', 'purpose_quote', 'notes']
    corpus_copy = _corpus_db.copy()
    anchors_copy = _anchors_db.copy()
    corpus_copy['text_for_embedding'] = corpus_copy[corpus_cols].fillna('').agg(' '.join, axis=1)
    anchors_copy['text_for_embedding'] = anchors_copy[anchor_cols].fillna('').agg(' '.join, axis=1)
    combined_db = pd.concat([
        corpus_copy[['doc_id', 'rrn', 'citation_full', 'text_for_embedding']],
        anchors_copy[['anchor_id', 'citation_full', 'text_for_embedding']]
    ], ignore_index=True)
    combined_db['unique_id'] = combined_db.apply(
        lambda row: f"RRN{str(int(row['doc_id'])).zfill(3)}" if pd.notna(row['doc_id']) else row['anchor_id'], axis=1
    )
    knowledge_base = combined_db.dropna(subset=['text_for_embedding'])
    embeddings = _model.encode(knowledge_base['text_for_embedding'].tolist())
    return knowledge_base, embeddings

def retrieve_semantic_context(query, knowledge_base_df, embeddings_matrix, model, top_n=5):
    # ... (Fungsi ini tetap sama, tidak perlu diubah) ...
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]
    retrieved_df = knowledge_base_df.iloc[top_n_indices].copy()
    retrieved_df['similarity_score'] = similarities[top_n_indices]
    return retrieved_df

def generate_narrative_answer(query, context_df):
    # --- PAGAR PENGAMAN #1: Periksa jika konteks kosong ---
    if context_df.empty:
        return "Saya tidak menemukan informasi yang cukup relevan di dalam basis data untuk menjawab pertanyaan ini. Silakan coba ajukan pertanyaan yang berbeda."

    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Harap konfigurasikan GOOGLE_API_KEY Anda di Streamlit Secrets. Error: {e}")
        return None

    model = genai.GenerativeModel('gemini-2.5-flash')
    context_text = "Berikut adalah potongan-potongan informasi relevan dari basis data riset:\n\n"
    for index, row in context_df.iterrows():
        context_text += f"--- Dokumen (ID: {row['unique_id']}) ---\nSitasi: {row['citation_full']}\nKutipan Relevan: {row['text_for_embedding'][:1500]}...\n\n"
    
    # --- PAGAR PENGAMAN #2: Prompt yang lebih ketat ---
    prompt = f"""
    Anda adalah seorang asisten riset ahli yang sangat teliti.
    Tugas Anda adalah menjawab pertanyaan pengguna secara ringkas dan akurat.
    JAWAB PERTANYAAN BERIKUT HANYA DAN EKSKLUSIF BERDASARKAN "KONTEKS" YANG DIBERIKAN.
    JANGAN berspekulasi, menambahkan informasi, atau menggunakan pengetahuan eksternal apa pun.
    Jika informasi untuk menjawab pertanyaan tidak ada di dalam KONTEKS, Anda WAJIB menjawab: "Informasi untuk menjawab pertanyaan ini tidak ditemukan dalam dokumen yang relevan."
    Untuk setiap klaim yang Anda buat, sebutkan sumbernya dengan merujuk pada ID Dokumen yang relevan.

    =========================
    KONTEKS:
    {context_text}
    =========================

    PERTANYAAN PENGGUNA:
    {query}

    JAWABAN AKURAT ANDA:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghubungi API Gemini: {e}")
        return None

# ==============================================================================
# BAGIAN 2: ALUR UTAMA APLIKASI
# ==============================================================================

df_corpus, df_anchors = load_data()

if df_corpus is not None and df_anchors is not None:
    embedding_model = load_embedding_model()
    # Menampilkan pesan status saat proses embedding
    with st.spinner("Mempersiapkan basis pengetahuan dan embeddings... (mungkin butuh beberapa saat saat pertama kali dimuat)"):
        knowledge_base, corpus_embeddings = prepare_and_embed_knowledge_base(df_corpus, df_anchors, embedding_model)
    st.success("Basis pengetahuan siap.")

    st.header("Ajukan Pertanyaan Riset Anda")
    user_query = st.text_input("Masukkan pertanyaan Anda tentang tata kelola, pemerataan manfaat, atau faktor keberhasilan CBT:", "Jelaskan berbagai tipologi desa wisata yang ada dalam korpus.")

    if st.button("Cari Jawaban"):
        if user_query:
            with st.spinner("Langkah 1: Mencari dokumen relevan..."):
                retrieved_df = retrieve_semantic_context(user_query, knowledge_base, corpus_embeddings, embedding_model)

            with st.expander("Lihat Dokumen yang Paling Relevan (Konteks untuk AI)"):
                st.dataframe(retrieved_df[['unique_id', 'citation_full', 'similarity_score']])

            with st.spinner("Langkah 2: Menyusun jawaban dengan Gemini 2.5 Flash..."):
                final_answer = generate_narrative_answer(user_query, retrieved_df)
                if final_answer:
                    st.markdown("---")
                    st.subheader("Jawaban dari Asisten Riset:")
                    st.markdown(final_answer)
        else:
            st.warning("Harap masukkan pertanyaan.")
