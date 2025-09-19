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

# --- Fungsi-fungsi Inti (dari Colab, sedikit dimodifikasi untuk cache) ---

# Menggunakan cache Streamlit agar data dan model tidak dimuat ulang setiap kali
@st.cache_data
def load_data():
    try:
        corpus_path = 'ScR_TV_Corpus.csv'  # Pastikan file ini ada di repo GitHub Anda
        anchors_path = 'ScR_TV_Anchors.csv' # Pastikan file ini ada di repo GitHub Anda
        df_corpus = pd.read_csv(corpus_path)
        df_anchors = pd.read_csv(anchors_path)
        return df_corpus, df_anchors
    except FileNotFoundError:
        st.error("Pastikan file 'ScR_TV_Corpus.csv' dan 'ScR_TV_Anchors.csv' berada di repositori GitHub Anda.")
        return None, None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def prepare_knowledge_base(_corpus_db, _anchors_db):
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
    return combined_db.dropna(subset=['text_for_embedding'])

@st.cache_data
def create_embeddings(_knowledge_base, _model):
    return _model.encode(_knowledge_base['text_for_embedding'].tolist())

# --- Inisialisasi Data & Model ---
df_corpus, df_anchors = load_data()
if df_corpus is not None:
    embedding_model = load_embedding_model()
    knowledge_base = prepare_knowledge_base(df_corpus, df_anchors)
    corpus_embeddings = create_embeddings(knowledge_base, embedding_model)

    # --- Antarmuka Pengguna (UI) ---
    st.header("Ajukan Pertanyaan Riset Anda")
    user_query = st.text_input("Masukkan pertanyaan Anda tentang tata kelola, pemerataan manfaat, atau faktor keberhasilan CBT:", "Apa saja faktor kunci keberhasilan untuk pariwisata berbasis komunitas (CBT) menurut literatur?")

    if st.button("Cari Jawaban"):
        if user_query:
            # Konfigurasi API Key dari Streamlit Secrets
            try:
                GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
                genai.configure(api_key=GOOGLE_API_KEY)
            except:
                st.error("Harap konfigurasikan GOOGLE_API_KEY Anda di Streamlit Secrets.")
                st.stop()

            with st.spinner("Mencari dokumen relevan..."):
                # Langkah 2: Retrieve
                retrieved_df = retrieve_semantic_context(user_query, knowledge_base, corpus_embeddings, embedding_model)

            with st.expander("Lihat Dokumen yang Paling Relevan (Konteks untuk AI)"):
                st.dataframe(retrieved_df[['unique_id', 'citation_full', 'similarity_score']])

            with st.spinner("Menyusun jawaban dengan Gemini 2.5 Flash..."):
                # Langkah 3: Generate
                final_answer = generate_narrative_answer(user_query, retrieved_df)
                st.markdown("---")
                st.subheader("Jawaban dari Asisten Riset:")
                st.markdown(final_answer)
        else:
            st.warning("Harap masukkan pertanyaan.")

# --- Fungsi yang Dipanggil (dari Colab) ---
def retrieve_semantic_context(query, knowledge_base_df, embeddings_matrix, model, top_n=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]
    retrieved_df = knowledge_base_df.iloc[top_n_indices].copy()
    retrieved_df['similarity_score'] = similarities[top_n_indices]
    return retrieved_df

def generate_narrative_answer(query, context_df):
    model = genai.GenerativeModel('gemini-2.5-flash')
    context_text = ""
    for index, row in context_df.iterrows():
        context_text += f"--- Dokumen (ID: {row['unique_id']}) ---\nSitasi: {row['citation_full']}\nKutipan Relevan: {row['text_for_embedding'][:1000]}...\n\n"
    
    prompt = f"""
    Anda adalah seorang asisten riset ahli. JAWAB PERTANYAAN BERIKUT HANYA BERDASARKAN "KONTEKS" YANG DIBERIKAN.
    Jangan gunakan pengetahuan eksternal. Sebutkan sumber dengan merujuk pada ID Dokumen yang relevan.
    KONTEKS:\n{context_text}\nPERTANYAAN PENGGUNA:\n{query}\n\nJAWABAN AKURAT ANDA:"""
    
    response = model.generate_content(prompt)
    return response.text