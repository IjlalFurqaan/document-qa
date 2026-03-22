"""
Configuration module for the Document Q&A RAG Pipeline.
Loads environment variables and defines application constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ─── Google Cloud ────────────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ─── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-004"

# ─── LLM Model ──────────────────────────────────────────────────────────────
LLM_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 2048

# ─── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 4
COLLECTION_NAME = "document_qa"

# ─── Prompt Template ────────────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based strictly on the provided context documents.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above.
- If the context does not contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
- Be concise, accurate, and professional.
- Cite relevant sections when possible.

ANSWER:"""
