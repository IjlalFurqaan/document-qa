"""
Configuration module for the Document Q&A RAG Pipeline.
Loads environment variables and defines application constants.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
BM25_INDEX_DIR = BASE_DIR / "bm25_index"
LOG_DIR = BASE_DIR / "logs"

# ─── Google Cloud ────────────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "models/text-embedding-004"

# ─── LLM Model ──────────────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 4096

# ─── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 6
RETRIEVAL_RERANK_TOP_K = 4
COLLECTION_NAME = "document_qa"

# ─── Hybrid Search ──────────────────────────────────────────────────────────
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.7"))
# Alpha controls the weight between semantic (1.0) and keyword search (0.0).
# 0.7 = 70% semantic, 30% keyword.

# ─── Conversation Memory ────────────────────────────────────────────────────
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))

# ─── Confidence ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.5

# ─── Supported File Types ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md", ".csv"]

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"


def setup_logging(verbose: bool = False):
    """Configure structured logging for the application."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ─── Prompt Template ────────────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """You are an expert document analyst providing grounded answers based strictly on the retrieved context.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the retrieved context above.
2. If the context does not contain sufficient information, clearly state: "I don't have enough information in the provided documents to answer this question."
3. Consider the conversation history for follow-up questions — resolve pronouns and references to previous answers.
4. Be concise, accurate, and professional.
5. Cite the specific document sources using [Source: filename] notation.
6. At the end of your answer, on a new line, provide a confidence rating in this exact format:
   CONFIDENCE: <HIGH|MEDIUM|LOW>

ANSWER:"""
