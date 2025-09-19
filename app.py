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

@st.cache_data(show_spinner=False)
def prepare_and_embed_knowledge_base(_corpus_db, _anchors_db):
    st.info("Mempersiapkan basis pengetahuan...")

    corpus_cols = ['title','abstract','key_arguments_findings',
                   'critical_comments_linkages','outcome_notes','notes']
    anchor_cols = ['citation_full','abstract','verbatim_definition',
                   'typology_summary','purpose_quote','notes']

    # Siapkan kolom agar tidak KeyError bila ada yang hilang
    corpus_copy = _corpus_db.copy()
    for c in corpus_cols + ['doc_id','rrn','citation_full']:
        if c not in corpus_copy.columns:
            corpus_copy[c] = ''

    anchors_copy = _anchors_db.copy()
    for c in anchor_cols + ['anchor_id','citation_full']:
        if c not in anchors_copy.columns:
            anchors_copy[c] = ''

    # Gabungkan teks untuk embedding
    corpus_copy['text_for_embedding'] = corpus_copy[corpus_cols].fillna('').agg(' '.join, axis=1)
    anchors_copy['text_for_embedding'] = anchors_copy[anchor_cols].fillna('').agg(' '.join, axis=1)

    combined_db = pd.concat([
        corpus_copy[['doc_id','rrn','citation_full','text_for_embedding']],
        anchors_copy[['anchor_id','citation_full','text_for_embedding']]
    ], ignore_index=True, sort=False)

    # unique_id yang robust
    def _make_uid(row):
        doc = row.get('doc_id', np.nan)
        anc = row.get('anchor_id', np.nan)
        doc_num = pd.to_numeric(doc, errors='coerce')
        if pd.notna(doc_num):
            return f"RRN{int(doc_num):03d}"
        if pd.notna(anc):
            return str(anc).strip()
        return f"UNK_{row.name}"

    combined_db['unique_id'] = combined_db.apply(_make_uid, axis=1)

    # Bersihkan baris kosong dan duplikat
    knowledge_base = combined_db.dropna(subset=['text_for_embedding']).copy()
    knowledge_base = knowledge_base[knowledge_base['text_for_embedding'].str.strip().ne('')].copy()
    knowledge_base['citation_full'] = knowledge_base['citation_full'].fillna('—')
    knowledge_base.drop_duplicates(subset=['unique_id'], inplace=True)

    st.success(f"Basis pengetahuan dengan {len(knowledge_base)} entri berhasil disiapkan.")

    # Buat embeddings (panggil model dari cache_resource DI DALAM fungsi ini)
    st.info("Membuat embeddings (representasi numerik)...")
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(
        knowledge_base['text_for_embedding'].tolist()
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    st.success(f"Berhasil membuat {len(embeddings)} embeddings.")

    return knowledge_base, embeddings

def retrieve_semantic_context(query, knowledge_base_df, embeddings_matrix, model, top_n=5, threshold=0.5):
    # Encode query
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

    # Urutkan skor
    all_indices = np.argsort(similarities)[::-1]
    all_scores = similarities[all_indices]

    # Filter di atas threshold
    relevant_mask = all_scores > threshold
    relevant_indices = all_indices[relevant_mask]
    final_indices = relevant_indices[:top_n]

    # Fallback: jika tidak ada yang lolos threshold, ambil TOP-N teratas
    if len(final_indices) == 0:
        final_indices = all_indices[:min(top_n, len(all_indices))]

    # Ambil baris yang relevan
    retrieved_df = knowledge_base_df.iloc[final_indices].copy()
    retrieved_df['similarity_score'] = similarities[final_indices]
    return retrieved_df

def generate_narrative_answer(query, context_df):
    if context_df.empty:
        return "**Saya tidak menemukan informasi yang cukup relevan di dalam basis data untuk menjawab pertanyaan ini.**\n\n*Saran: Coba turunkan 'Ambang Batas Relevansi' di bilah sisi atau ajukan pertanyaan yang berbeda.*"

    # API key Gemini
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Harap konfigurasikan GOOGLE_API_KEY Anda di Streamlit Secrets. Error: {e}")
        return None

    model = genai.GenerativeModel('gemini-2.5-flash')

    # Meta-konteks umum (tanpa angka tetap)
    meta_context_text = """
    Seluruh basis data yang digunakan sebagai sumber pengetahuan berasal dari sebuah scoping review sistematis (2015–2025) tentang 'Desa Wisata'.
    Semua dokumen di dalamnya telah melalui proses penyaringan dengan kriteria inklusi yang berfokus pada 'tourism village',
    'community-based tourism' pada skala desa, atau 'village-level rural tourism'.
    Oleh karena itu, Anda harus mengasumsikan bahwa semua informasi yang diberikan sudah berada dalam konteks 'desa wisata'.
    """

    # Susun konteks yang dioper ke LLM
    context_lines = ["Berikut adalah potongan-potongan informasi paling relevan untuk pertanyaan ini:\n"]
    for _, row in context_df.iterrows():
        snippet = (row['text_for_embedding'] or "")[:1500]
        context_lines.append(
            f"--- Dokumen (ID: {row['unique_id']}) ---\n"
            f"Sitasi: {row['citation_full']}\n"
            f"Kutipan Relevan: {snippet}...\n"
        )
    context_text = "\n".join(context_lines)

    prompt = f"""
PERAN: Anda adalah asisten riset ahli yang sangat teliti.

META-KONTEKS KESELURUHAN:
{meta_context_text}

TUGAS ANDA:
1. Baca dan pahami META-KONTEKS di atas (berlaku untuk semua pertanyaan).
2. Baca KONTEKS SPESIFIK berikut (informasi paling relevan untuk pertanyaan saat ini).
3. Jawab PERTANYAAN PENGGUNA secara analitis. Sintesiskan informasi dari KONTEKS SPESIFIK, dan interpretasikan melalui lensa META-KONTEKS.
4. WAJIB menjawab HANYA berdasarkan informasi yang disediakan. JANGAN gunakan pengetahuan eksternal.
5. Jika informasi tidak ada di KONTEKS SPESIFIK, jawab eksplisit: "Informasi untuk menjawab pertanyaan ini tidak ditemukan secara spesifik dalam dokumen yang relevan, namun secara umum, seluruh korpus ini membahas desa wisata."
6. Sebutkan ID Dokumen untuk setiap klaim spesifik yang Anda buat.

=========================
KONTEKS SPESIFIK YANG DITEMUKAN:
{context_text}
=========================

PERTANYAAN PENGGUNA:
{query}

JAWABAN ANALITIS ANDA:
""".strip()

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghubungi API Gemini: {e}")
        return None

# ==============================================================================
# BAGIAN 2: ALUR UTAMA APLIKASI
# ==============================================================================

# Sidebar
st.sidebar.header("Pengaturan Lanjutan")
relevance_threshold = st.sidebar.slider(
    "Ambang Batas Relevansi (Relevance Threshold)",
    min_value=0.0, max_value=1.0, value=0.55, step=0.05,
    help="Hanya dokumen dengan skor kemiripan di atas ambang ini yang akan digunakan sebagai konteks. Tingkatkan jika hasilnya terlalu umum, turunkan jika tidak ada hasil yang ditemukan."
)

# Memuat data & model
df_corpus, df_anchors = load_data()

if df_corpus is not None and df_anchors is not None:
    # Model untuk QUERY (tidak dipassing ke fungsi cache)
    embedding_model = load_embedding_model()

    with st.spinner("Mempersiapkan basis pengetahuan... (hanya saat pertama kali dimuat)"):
        knowledge_base, corpus_embeddings = prepare_and_embed_knowledge_base(df_corpus, df_anchors)
    st.success("Basis pengetahuan siap.")

    st.header("Ajukan Pertanyaan Riset Anda")
    user_query = st.text_input(
        "Masukkan pertanyaan Anda tentang tata kelola, pemerataan manfaat, atau faktor keberhasilan CBT:",
        "Jelaskan berbagai tipologi desa wisata yang ada dalam korpus."
    )

    if st.button("Cari Jawaban"):
        if user_query:
            with st.spinner("Langkah 1: Mencari dokumen relevan..."):
                retrieved_df = retrieve_semantic_context(
                    user_query,
                    knowledge_base,
                    corpus_embeddings,
                    embedding_model,
                    threshold=relevance_threshold
                )

            st.markdown("---")
            st.subheader("Hasil Pencarian Konteks")

            if retrieved_df.empty:
                st.warning(f"Tidak ada dokumen yang ditemukan di atas ambang batas relevansi (skor > {relevance_threshold}). Konteks yang dikirim ke AI kosong.")
            else:
                st.success(f"Ditemukan {len(retrieved_df)} dokumen relevan di atas ambang batas.")
                with st.expander("Klik untuk melihat detail dokumen yang ditemukan"):
                    df_view = retrieved_df[['unique_id', 'citation_full', 'similarity_score']].copy()
                    df_view['similarity_score'] = df_view['similarity_score'].round(3)
                    st.dataframe(df_view, use_container_width=True)

            with st.spinner("Langkah 2: Menyusun jawaban dengan Gemini 2.5 Flash..."):
                final_answer = generate_narrative_answer(user_query, retrieved_df)
                if final_answer:
                    st.markdown("---")
                    st.subheader("Jawaban dari Asisten Riset:")
                    st.markdown(final_answer)
        else:
            st.warning("Harap masukkan pertanyaan.")
