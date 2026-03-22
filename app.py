"""
Streamlit Web UI for Document Q&A RAG Pipeline.

Premium dark-mode interface with analytics dashboard,
conversation memory, confidence scoring, and document management.
"""

import streamlit as st
from pathlib import Path
import time
from datetime import datetime

from config import DOCUMENTS_DIR, VECTORSTORE_DIR, SUPPORTED_EXTENSIONS


# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQ — Intelligent Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --bg-card-hover: #1e1e35;
        --accent-primary: #6C63FF;
        --accent-secondary: #4ECDC4;
        --accent-gradient: linear-gradient(135deg, #6C63FF 0%, #4ECDC4 100%);
        --text-primary: #E8E8F0;
        --text-secondary: #9999B0;
        --text-dim: #666680;
        --border-subtle: rgba(108, 99, 255, 0.15);
        --glass: rgba(26, 26, 46, 0.6);
        --glass-border: rgba(108, 99, 255, 0.2);
        --shadow-glow: 0 0 40px rgba(108, 99, 255, 0.15);
        --confidence-high: #4ECDC4;
        --confidence-medium: #FFD93D;
        --confidence-low: #FF6B6B;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary) !important;
        color: var(--text-primary);
    }

    /* ─── Glassmorphism Header ─── */
    .hero-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.12) 0%, rgba(78, 205, 196, 0.08) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-glow);
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(108, 99, 255, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(78, 205, 196, 0.06) 0%, transparent 50%);
        animation: float 8s ease-in-out infinite alternate;
    }

    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -10px) rotate(2deg); }
    }

    .hero-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .hero-header p {
        margin: 0.5rem 0 0;
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 4px;
        position: relative;
        z-index: 1;
    }

    .badge-semantic { background: rgba(108, 99, 255, 0.2); color: #6C63FF; }
    .badge-bm25 { background: rgba(78, 205, 196, 0.2); color: #4ECDC4; }
    .badge-memory { background: rgba(255, 217, 61, 0.2); color: #FFD93D; }
    .badge-rerank { background: rgba(255, 107, 107, 0.2); color: #FF6B6B; }

    /* ─── Stat Cards ─── */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stat-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-primary);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* ─── Answer Box ─── */
    .answer-box {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.06) 0%, rgba(78, 205, 196, 0.04) 100%);
        border: 1px solid var(--glass-border);
        border-left: 4px solid var(--accent-primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.8;
        color: var(--text-primary);
        backdrop-filter: blur(10px);
    }

    /* ─── Confidence Badges ─── */
    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .confidence-high {
        background: rgba(78, 205, 196, 0.15);
        color: var(--confidence-high);
        border: 1px solid rgba(78, 205, 196, 0.3);
    }

    .confidence-medium {
        background: rgba(255, 217, 61, 0.15);
        color: var(--confidence-medium);
        border: 1px solid rgba(255, 217, 61, 0.3);
    }

    .confidence-low {
        background: rgba(255, 107, 107, 0.15);
        color: var(--confidence-low);
        border: 1px solid rgba(255, 107, 107, 0.3);
    }

    /* ─── Source Cards ─── */
    .source-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .source-card:hover {
        border-color: var(--accent-primary);
        background: var(--bg-card-hover);
    }

    .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-weight: 600;
        color: var(--accent-secondary);
    }

    .source-preview {
        color: var(--text-dim);
        font-size: 0.8rem;
        line-height: 1.6;
    }

    /* ─── Sidebar ─── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #12121a 100%) !important;
        border-right: 1px solid var(--border-subtle);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }

    /* ─── Metadata pill ─── */
    .meta-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(108, 99, 255, 0.1);
        color: var(--accent-primary);
        margin-right: 6px;
    }

    /* ─── Chat bubbles override ─── */
    .stChatMessage {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
    }

    /* ─── Dividers ─── */
    hr {
        border-color: var(--border-subtle) !important;
    }

</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()) if VECTORSTORE_DIR.exists() else False
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0.0
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{int(time.time())}"


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.divider()

    # ── Upload Documents ──
    st.markdown("### 📁 Upload Documents")
    st.caption(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=[ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="Drag and drop or browse files to build your knowledge base",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")

        if st.button("📥 Ingest Uploaded Files", type="primary", use_container_width=True):
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = DOCUMENTS_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Ingesting documents..."):
                from ingest import ingest_pipeline
                stats = ingest_pipeline()

            st.session_state.vectorstore_ready = True
            st.success(
                f"✅ Done!\n\n"
                f"📊 Documents: {stats['documents']}\n\n"
                f"🧩 Chunks: {stats['chunks']}\n\n"
                f"⏱ Time: {stats.get('elapsed_seconds', '?')}s"
            )

    st.divider()

    # ── Local Documents ──
    st.markdown("### 📂 Local Documents")
    st.caption(f"`{DOCUMENTS_DIR}`")

    if DOCUMENTS_DIR.exists():
        existing_docs = []
        for ext in SUPPORTED_EXTENSIONS:
            existing_docs.extend(DOCUMENTS_DIR.glob(f"*{ext}"))

        if existing_docs:
            for doc in existing_docs:
                size_kb = round(doc.stat().st_size / 1024, 1)
                st.markdown(f"• `{doc.name}` ({size_kb} KB)")
        else:
            st.caption("No documents found")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Ingest", use_container_width=True):
            with st.spinner("Processing..."):
                from ingest import ingest_pipeline
                stats = ingest_pipeline()
            st.session_state.vectorstore_ready = True
            st.success(f"✅ {stats['chunks']} chunks")

    with col2:
        force_ingest = st.button("🔄 Force", use_container_width=True, help="Re-ingest all files")
        if force_ingest:
            with st.spinner("Re-processing all..."):
                from ingest import ingest_pipeline
                stats = ingest_pipeline(force=True)
            st.session_state.vectorstore_ready = True
            st.success(f"✅ {stats['chunks']} chunks")

    st.divider()

    # ── Status ──
    st.markdown("### 📊 Status")
    if st.session_state.vectorstore_ready:
        st.success("✅ Vector store ready")

        try:
            from ingest import get_vectorstore_stats
            vs_stats = get_vectorstore_stats()
            st.metric("Chunks indexed", vs_stats["chunk_count"])
            st.metric("Documents", len(vs_stats["documents"]))
            if vs_stats["has_bm25"]:
                st.success("✅ BM25 keyword index")
        except Exception:
            pass
    else:
        st.warning("⚠️ Ingest documents first")

    st.divider()

    # ── Actions ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.total_latency = 0.0
            from rag_chain import clear_memory
            clear_memory(st.session_state.session_id)
            st.rerun()

    with col2:
        if st.button("🧹 Reset All", use_container_width=True):
            import shutil
            if VECTORSTORE_DIR.exists():
                shutil.rmtree(VECTORSTORE_DIR)
            from config import BM25_INDEX_DIR
            if BM25_INDEX_DIR.exists():
                shutil.rmtree(BM25_INDEX_DIR)
            st.session_state.vectorstore_ready = False
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption(
        "**DocQ** — Enterprise RAG Pipeline\n\n"
        "Google GenAI • LangChain • ChromaDB\n\n"
        "Hybrid Search • Re-Ranking • Memory"
    )


# ─── Main Content ───────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1>🧠 DocQ — Intelligent Document Q&A</h1>
    <p>Ask natural language questions — powered by hybrid search, re-ranking, and conversation memory</p>
    <div style="margin-top: 12px;">
        <span class="hero-badge badge-semantic">🔮 Semantic Search</span>
        <span class="hero-badge badge-bm25">🔑 BM25 Keywords</span>
        <span class="hero-badge badge-memory">💬 Memory</span>
        <span class="hero-badge badge-rerank">📊 Re-Ranking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    doc_count = 0
    if DOCUMENTS_DIR.exists():
        for ext in SUPPORTED_EXTENSIONS:
            doc_count += len(list(DOCUMENTS_DIR.glob(f"*{ext}")))
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{doc_count}</div>
        <div class="stat-label">📄 Documents</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{st.session_state.total_queries}</div>
        <div class="stat-label">💬 Queries</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_latency = (
        round(st.session_state.total_latency / st.session_state.total_queries, 1)
        if st.session_state.total_queries > 0
        else 0
    )
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{avg_latency}s</div>
        <div class="stat-label">⏱ Avg Latency</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    status_text = "Ready" if st.session_state.vectorstore_ready else "Setup"
    status_color = "var(--confidence-high)" if st.session_state.vectorstore_ready else "var(--confidence-low)"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value" style="-webkit-text-fill-color: {status_color}; color: {status_color};">{status_text}</div>
        <div class="stat-label">🗄️ Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Chat History ────────────────────────────────────────────────────────────

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(entry["answer"])

        # Confidence badge
        confidence = entry.get("confidence", "MEDIUM")
        conf_class = f"confidence-{confidence.lower()}"
        conf_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
        st.markdown(
            f'<span class="confidence-badge {conf_class}">'
            f'{conf_icons.get(confidence, "🟡")} {confidence} Confidence</span>'
            f'&nbsp;&nbsp;<span class="meta-pill">⏱ {entry.get("latency", "?")}s</span>'
            f'<span class="meta-pill">🔍 {entry.get("retrieval_method", "hybrid")}</span>',
            unsafe_allow_html=True,
        )

        if entry.get("sources"):
            with st.expander("📎 View Source Documents", expanded=False):
                for i, src in enumerate(entry["sources"], 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">📄 Source {i}: {src["source"]}'
                        f'&nbsp;<span class="meta-pill">{src.get("file_type", "?").upper()}</span>'
                        f'<span class="meta-pill">Chunk #{src.get("chunk_index", "?")}</span>'
                        f'</div>'
                        f'<div class="source-preview">{src["content_preview"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ─── Chat Input ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.vectorstore_ready:
        st.error("⚠️ Please ingest documents first using the sidebar.")
    else:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Searching documents & reasoning..."):
                try:
                    from rag_chain import ask
                    result = ask(prompt, session_id=st.session_state.session_id)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    result = None

            if result:
                st.markdown(result["answer"])

                # Confidence badge
                confidence = result.get("confidence", "MEDIUM")
                conf_class = f"confidence-{confidence.lower()}"
                conf_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
                latency = result.get("latency_seconds", "?")

                st.markdown(
                    f'<span class="confidence-badge {conf_class}">'
                    f'{conf_icons.get(confidence, "🟡")} {confidence} Confidence</span>'
                    f'&nbsp;&nbsp;<span class="meta-pill">⏱ {latency}s</span>'
                    f'<span class="meta-pill">🔍 {result.get("retrieval_method", "hybrid")}</span>',
                    unsafe_allow_html=True,
                )

                if result.get("sources"):
                    with st.expander("📎 View Source Documents", expanded=False):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="source-header">📄 Source {i}: {src["source"]}'
                                f'&nbsp;<span class="meta-pill">{src.get("file_type", "?").upper()}</span>'
                                f'<span class="meta-pill">Chunk #{src.get("chunk_index", "?")}</span>'
                                f'</div>'
                                f'<div class="source-preview">{src["content_preview"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Update session state
                st.session_state.total_queries += 1
                st.session_state.total_latency += result.get("latency_seconds", 0)

                st.session_state.chat_history.append({
                    "question": prompt,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", "MEDIUM"),
                    "latency": result.get("latency_seconds", "?"),
                    "retrieval_method": result.get("retrieval_method", "hybrid"),
                })
