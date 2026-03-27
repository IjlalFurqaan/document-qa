# 🧠 DocQ — Intelligent Document Q&A

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=flat-square)](https://langchain.com)
[![Google GenAI](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://cloud.google.com/vertex-ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=flat-square)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)

## 📖 Introduction

**DocQ** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** pipeline that transforms the way you interact with your documents. Instead of manually searching through pages of PDFs, Word files, Markdown notes, or CSV datasets, you can simply ask a natural language question and receive a precise, source-cited answer in seconds.

Modern organizations generate and store vast amounts of unstructured knowledge — policy documents, technical manuals, research papers, meeting notes, and more. Finding specific information buried inside these documents is often a time-consuming, frustrating process that involves keyword searches, manual skimming, and guesswork. **DocQ eliminates this friction** by indexing your documents into a searchable knowledge base and using Google's **Gemini 2.0 Flash** LLM to generate grounded, contextual answers.

What sets DocQ apart from a naive "embed-and-query" setup is its **hybrid search architecture**. Rather than relying on a single retrieval strategy, DocQ combines **semantic vector search** (via ChromaDB and Google GenAI Embeddings) with **keyword-based BM25 search**, merging results through **Reciprocal Rank Fusion**. The fused results are then **re-ranked by Gemini itself**, ensuring that only the most relevant chunks are used for answer generation. This multi-stage retrieval pipeline dramatically improves answer quality, especially for queries that mix technical jargon with plain-language phrasing.

DocQ also supports **conversation memory**, allowing you to ask follow-up questions that build on previous context — just like chatting with a knowledgeable colleague. Every answer is accompanied by a **confidence score** (HIGH / MEDIUM / LOW) and **source citations**, so you always know how reliable the response is and where to verify it. Whether you prefer a sleek **dark-mode web UI** built with Streamlit or a **rich terminal CLI** with tables and spinners, DocQ delivers a premium experience across both interfaces.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Hybrid Search** | Combines semantic (ChromaDB) + keyword (BM25) retrieval via Reciprocal Rank Fusion |
| 📊 **LLM Re-Ranking** | Gemini scores and re-ranks retrieved chunks for maximum relevance |
| 💬 **Conversation Memory** | Follow-up questions with full context from previous turns |
| 🎯 **Confidence Scoring** | Each answer rated HIGH / MEDIUM / LOW with visual badges |
| 📁 **Multi-Format Ingestion** | Supports PDF, TXT, DOCX, Markdown, and CSV files |
| 🔄 **Duplicate Detection** | SHA-256 hashing skips unchanged files on re-ingestion |
| 🎨 **Premium Web UI** | Dark-mode Streamlit interface with glassmorphism design |
| 📈 **Analytics Dashboard** | Track query count, average latency, document stats |
| 🖥️ **Rich CLI** | Beautiful terminal interface with tables, spinners, and color |
| 📝 **Structured Logging** | File-based logging with configurable levels |

---

## 🏗️ Architecture

```mermaid
graph TD
    A[📄 Documents<br/>PDF · TXT · DOCX · MD · CSV] --> B[Document Loader<br/>LangChain Multi-Format]
    B --> C[Text Splitter<br/>Recursive, 1000 chars]
    C --> D1[Semantic Index<br/>ChromaDB + GenAI Embeddings]
    C --> D2[Keyword Index<br/>BM25Okapi]

    E[❓ User Query] --> F1[Semantic Search<br/>Top-K Similarity]
    E --> F2[BM25 Search<br/>Keyword Matching]

    F1 --> G[Reciprocal Rank Fusion<br/>Weighted Score Merging]
    F2 --> G

    G --> H[LLM Re-Ranker<br/>Gemini Relevance Scoring]
    H --> I[Context Assembly<br/>+ Conversation Memory]
    I --> J[Gemini 2.0 Flash<br/>Answer Generation]
    J --> K[✅ Grounded Answer<br/>+ Confidence Score<br/>+ Source Citations]

    style A fill:#6C63FF20,stroke:#6C63FF
    style K fill:#4ECDC420,stroke:#4ECDC4
    style G fill:#FFD93D20,stroke:#FFD93D
    style H fill:#FF6B6B20,stroke:#FF6B6B
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Google Cloud SDK** with Vertex AI API enabled, OR a **Gemini API key**

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/document-qa.git
cd document-qa

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

### Authentication (choose one)

**Option A — Google Cloud ADC (recommended):**
```bash
gcloud auth application-default login
# Then edit .env → set GOOGLE_CLOUD_PROJECT
```

**Option B — Gemini API Key:**
```bash
# Edit .env → set GOOGLE_API_KEY=your-key-here
```

---

## 💻 Usage

### CLI — Ingest Documents

```bash
# Ingest documents from ./documents (with duplicate detection)
python main.py ingest

# Force re-ingest all files
python main.py ingest --force

# Ingest from a custom directory
python main.py ingest --dir /path/to/docs
```

### CLI — Query Documents

```bash
# Interactive mode (with conversation memory)
python main.py query

# Single question
python main.py query -q "What is the remote work policy?"
```

### CLI — Manage Vector Store

```bash
# View vector store statistics
python main.py stats

# Clear all indexes
python main.py clear
```

### Web UI

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
document-qa/
├── config.py           # Configuration, constants, and logging setup
├── ingest.py           # Multi-format ingestion with BM25 + ChromaDB
├── rag_chain.py        # Hybrid retrieval, re-ranking, memory, confidence
├── main.py             # CLI entry point with ingest/query/stats/clear
├── app.py              # Premium Streamlit web UI
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── documents/          # Place your documents here
├── vectorstore/        # ChromaDB persisted data (auto-generated)
├── bm25_index/         # BM25 keyword index (auto-generated)
└── logs/               # Application logs (auto-generated)
```

---

## ⚙️ Configuration

All settings are configurable via `.env` or `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | — | Your GCP project ID |
| `GOOGLE_API_KEY` | — | Alternative: Gemini API key |
| `HYBRID_SEARCH_ALPHA` | `0.7` | Semantic vs keyword weight (0.0–1.0) |
| `MEMORY_WINDOW_SIZE` | `5` | Conversation turns to retain |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING) |

---


## 🛠️ Tech Stack

- **[LangChain](https://langchain.com)** — Orchestration framework
- **[Google Gemini 2.0 Flash](https://ai.google.dev)** — LLM for answer generation & re-ranking
- **[Google GenAI Embeddings](https://ai.google.dev)** — `text-embedding-004` for semantic search
- **[ChromaDB](https://www.trychroma.com)** — Persisted vector store
- **[BM25Okapi](https://github.com/dorianbrown/rank_bm25)** — Keyword search index
- **[Streamlit](https://streamlit.io)** — Interactive web interface
- **[Rich](https://rich.readthedocs.io)** — Beautiful terminal output

---

## 📄 License

MIT
