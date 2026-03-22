"""
Streamlit Web UI for Document Q&A RAG Pipeline.

Provides an interactive chat-like interface for querying
documents with real-time ingestion and semantic search.
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil

from config import DOCUMENTS_DIR, VECTORSTORE_DIR


# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A — RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
    }

    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .stat-badge {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }

    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.divider()

    # Document Upload Section
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload enterprise documents to build your knowledge base",
    )

    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")

        if st.button("📥 Save & Ingest", type="primary", use_container_width=True):
            # Save uploaded files to documents directory
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = DOCUMENTS_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.success(f"✅ Saved {len(uploaded_files)} file(s)")

            # Run ingestion
            with st.spinner("🔄 Ingesting documents... This may take a moment."):
                from ingest import ingest_pipeline
                stats = ingest_pipeline()

            st.session_state.vectorstore_ready = True

            st.success(
                f"✅ Ingestion complete!\n\n"
                f"📊 Documents: {stats['documents']}\n\n"
                f"🧩 Chunks: {stats['chunks']}"
            )

    st.divider()

    # Ingest existing documents
    st.markdown("### 📂 Ingest Local Documents")
    st.caption(f"Documents directory: `{DOCUMENTS_DIR}`")

    existing_docs = list(DOCUMENTS_DIR.glob("*.txt")) + list(DOCUMENTS_DIR.glob("*.pdf"))
    if existing_docs:
        st.info(f"Found {len(existing_docs)} document(s) in directory")

    if st.button("🚀 Ingest Documents", use_container_width=True):
        with st.spinner("🔄 Processing documents..."):
            from ingest import ingest_pipeline
            stats = ingest_pipeline()

        st.session_state.vectorstore_ready = True
        st.success(
            f"✅ Done! {stats['documents']} doc(s), {stats['chunks']} chunk(s)"
        )

    st.divider()

    # Vector store status
    st.markdown("### 📊 Status")
    if st.session_state.vectorstore_ready:
        st.success("✅ Vector store is ready")
    else:
        st.warning("⚠️ No vector store found. Please ingest documents first.")

    # Clear history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Powered by **Vertex AI** • **LangChain** • **ChromaDB**\n\n"
        "Built with ❤️ using Streamlit"
    )


# ─── Main Content ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 Document Q&A</h1>
    <p>Ask natural language questions about your enterprise documents</p>
</div>
""", unsafe_allow_html=True)

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    doc_count = len(list(DOCUMENTS_DIR.glob("*.txt")) + list(DOCUMENTS_DIR.glob("*.pdf")))
    st.metric("📄 Documents", doc_count)
with col2:
    st.metric("💬 Questions Asked", len(st.session_state.chat_history))
with col3:
    status = "🟢 Ready" if st.session_state.vectorstore_ready else "🔴 Not Ready"
    st.metric("🗄️ Vector Store", status)

st.divider()

# Chat History Display
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant", avatar="📚"):
        st.markdown(entry["answer"])

        if entry.get("sources"):
            with st.expander("📎 View Sources", expanded=False):
                for i, src in enumerate(entry["sources"], 1):
                    st.markdown(
                        f"**Source {i}:** `{src['source']}`\n\n"
                        f">{src['content_preview']}"
                    )

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.vectorstore_ready:
        st.error("⚠️ Please ingest documents first using the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("🔍 Searching documents & generating answer..."):
                from rag_chain import ask
                result = ask(prompt)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📎 View Sources", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f"**Source {i}:** `{src['source']}`\n\n"
                            f">{src['content_preview']}"
                        )

        # Save to history
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        })
