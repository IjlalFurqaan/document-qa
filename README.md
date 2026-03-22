# Document Q&A — RAG Pipeline

LLM-powered document Q&A system using **Vertex AI**, **LangChain**, and **ChromaDB** for semantic search over enterprise documents.

## Architecture

```
Documents (PDF/TXT)
        ↓
   Document Loader (LangChain)
        ↓
   Text Splitter (Recursive, 1000 chars)
        ↓
   Embeddings (Vertex AI text-embedding-004)
        ↓
   Vector Store (ChromaDB, persisted)
        ↓
   Retriever (Similarity Search, top-k=4)
        ↓
   LLM (Gemini 1.5 Flash via Vertex AI)
        ↓
   Grounded Answer + Source Citations
```

## Prerequisites

1. **Python 3.10+**
2. **Google Cloud SDK** with Vertex AI API enabled
3. Authenticate:
   ```bash
   gcloud auth application-default login
   ```

## Setup

```bash
# Navigate to project
cd document-qa

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your GCP project ID
```

## Usage

### CLI — Ingest Documents

```bash
# Ingest documents from the default ./documents folder
python main.py ingest

# Ingest from a custom directory
python main.py ingest --dir /path/to/your/docs
```

### CLI — Query Documents

```bash
# Interactive mode
python main.py query

# Single question
python main.py query -q "What is the remote work policy?"
```

### Web UI

```bash
streamlit run app.py
```

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Configuration and constants |
| `ingest.py` | Document loading, chunking, and vectorization |
| `rag_chain.py` | RAG chain with LangChain LCEL + Vertex AI |
| `main.py` | CLI entry point |
| `app.py` | Streamlit web UI |

## Tech Stack

- **LangChain** — Orchestration framework
- **Vertex AI** — Gemini 1.5 Flash (LLM) + text-embedding-004 (Embeddings)
- **ChromaDB** — Local vector store with persistence
- **Streamlit** — Interactive web interface
