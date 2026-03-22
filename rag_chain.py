"""
RAG Chain Module.

Builds the Retrieval-Augmented Generation chain with hybrid search
(semantic + BM25 keyword), re-ranking, conversation memory, and
confidence scoring via Google GenAI (Gemini).
"""

import logging
import re
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import (
    GCP_PROJECT,
    GCP_LOCATION,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    QA_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K,
    RETRIEVAL_RERANK_TOP_K,
    HYBRID_SEARCH_ALPHA,
    MEMORY_WINDOW_SIZE,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_MEDIUM,
)
from ingest import get_vectorstore, load_bm25_index

logger = logging.getLogger(__name__)

# ─── In-Memory Conversation Store ────────────────────────────────────────────
# Maps session_id → list of (question, answer) tuples
_conversation_store: dict[str, list[tuple[str, str]]] = {}


def _get_memory(session_id: str) -> list[tuple[str, str]]:
    """Retrieve conversation history for a session."""
    return _conversation_store.get(session_id, [])


def _save_to_memory(session_id: str, question: str, answer: str):
    """Save a Q&A turn to conversation memory."""
    if session_id not in _conversation_store:
        _conversation_store[session_id] = []

    _conversation_store[session_id].append((question, answer))

    # Trim to window size
    if len(_conversation_store[session_id]) > MEMORY_WINDOW_SIZE:
        _conversation_store[session_id] = _conversation_store[session_id][-MEMORY_WINDOW_SIZE:]


def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    _conversation_store.pop(session_id, None)


def _format_chat_history(session_id: str) -> str:
    """Format conversation history into a string for the prompt."""
    history = _get_memory(session_id)
    if not history:
        return "No previous conversation."

    formatted = []
    for q, a in history:
        formatted.append(f"User: {q}\nAssistant: {a}")

    return "\n\n".join(formatted)


# ─── Document Formatting ────────────────────────────────────────────────────

def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", doc.metadata.get("source", "Unknown"))
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted.append(
            f"[Document {i} — Source: {source} | Chunk {chunk_idx}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# ─── Hybrid Retrieval ────────────────────────────────────────────────────────

def _hybrid_retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
    """
    Perform hybrid retrieval combining semantic search (ChromaDB) and
    keyword search (BM25) using Reciprocal Rank Fusion (RRF).

    Args:
        query: The user's query.
        top_k: Number of final results to return.

    Returns:
        List of Document objects, ranked by fused score.
    """
    # 1. Semantic search via ChromaDB
    vectorstore = get_vectorstore()
    semantic_results = vectorstore.similarity_search(query, k=top_k)
    logger.info(f"Semantic search returned {len(semantic_results)} results")

    # 2. BM25 keyword search
    bm25_data = load_bm25_index()
    if bm25_data is None:
        logger.warning("No BM25 index found — using semantic-only retrieval")
        return semantic_results

    bm25, corpus, metadata_list = bm25_data
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top-k BM25 results
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [
        Document(page_content=corpus[i], metadata=metadata_list[i])
        for i in top_bm25_indices
        if bm25_scores[i] > 0
    ]
    logger.info(f"BM25 search returned {len(bm25_results)} results")

    # 3. Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    rrf_docs: dict[str, Document] = {}
    k_constant = 60  # Standard RRF constant

    alpha = HYBRID_SEARCH_ALPHA

    for rank, doc in enumerate(semantic_results):
        doc_key = doc.page_content[:200]
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + alpha * (1.0 / (k_constant + rank + 1))
        rrf_docs[doc_key] = doc

    for rank, doc in enumerate(bm25_results):
        doc_key = doc.page_content[:200]
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + (1 - alpha) * (1.0 / (k_constant + rank + 1))
        if doc_key not in rrf_docs:
            rrf_docs[doc_key] = doc

    # Sort by fused score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    fused_results = [rrf_docs[k] for k in sorted_keys[:top_k]]

    logger.info(f"Hybrid retrieval returned {len(fused_results)} fused results")
    return fused_results


# ─── Re-Ranking ──────────────────────────────────────────────────────────────

def _rerank_documents(query: str, documents: list[Document], top_k: int = RETRIEVAL_RERANK_TOP_K) -> list[Document]:
    """
    Re-rank retrieved documents using the LLM to score relevance.

    Args:
        query: The user's query.
        documents: List of candidate documents.
        top_k: Number of documents to keep after re-ranking.

    Returns:
        List of re-ranked documents (most relevant first).
    """
    if len(documents) <= top_k:
        return documents

    llm = _get_llm()

    rerank_prompt = PromptTemplate(
        template=(
            "Rate the relevance of the following text passage to the query on a scale of 0-10.\n\n"
            "Query: {query}\n\n"
            "Passage: {passage}\n\n"
            "Respond with ONLY a number from 0-10, nothing else."
        ),
        input_variables=["query", "passage"],
    )

    scored_docs = []
    for doc in documents:
        try:
            chain = rerank_prompt | llm | StrOutputParser()
            score_str = chain.invoke({"query": query, "passage": doc.page_content[:500]})
            score = float(re.search(r"(\d+(?:\.\d+)?)", score_str).group(1))
            scored_docs.append((score, doc))
        except Exception as e:
            logger.warning(f"Re-ranking failed for a chunk: {e}")
            scored_docs.append((5.0, doc))  # Default mid-score on failure

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored_docs[:top_k]]

    logger.info(f"Re-ranked {len(documents)} → {len(reranked)} documents")
    return reranked


# ─── Confidence Parsing ─────────────────────────────────────────────────────

def _parse_confidence(answer: str) -> tuple[str, str, float]:
    """
    Parse the confidence level from the LLM response.

    Returns:
        Tuple of (clean_answer, confidence_label, confidence_score).
    """
    confidence_label = "MEDIUM"
    confidence_score = 0.6

    # Try to extract CONFIDENCE line
    match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", answer, re.IGNORECASE)
    if match:
        confidence_label = match.group(1).upper()
        # Remove the confidence line from the answer
        answer = re.sub(r"\n?\s*CONFIDENCE:\s*(HIGH|MEDIUM|LOW)\s*$", "", answer, flags=re.IGNORECASE).strip()

    score_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    confidence_score = score_map.get(confidence_label, 0.6)

    return answer, confidence_label, confidence_score


# ─── LLM ─────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    """Create and return a Google GenAI LLM instance."""
    kwargs = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
    }

    if GOOGLE_API_KEY:
        kwargs["google_api_key"] = GOOGLE_API_KEY
    else:
        kwargs["project"] = GCP_PROJECT
        kwargs["location"] = GCP_LOCATION

    return ChatGoogleGenerativeAI(**kwargs)


# ─── Main Ask Function ──────────────────────────────────────────────────────

def ask(query: str, session_id: str = "default") -> dict:
    """
    Ask a question with hybrid retrieval, re-ranking, memory, and confidence.

    Args:
        query: The natural language question.
        session_id: Session ID for conversation memory isolation.

    Returns:
        Dictionary with answer, sources, confidence, and latency.
    """
    start_time = time.time()

    # 1. Hybrid retrieval (semantic + BM25)
    retrieved_docs = _hybrid_retrieve(query)

    # 2. Re-rank
    reranked_docs = _rerank_documents(query, retrieved_docs)

    # 3. Format context and chat history
    context = _format_docs(reranked_docs)
    chat_history = _format_chat_history(session_id)

    # 4. Build and invoke the chain
    llm = _get_llm()
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
    )

    chain = prompt | llm | StrOutputParser()
    raw_answer = chain.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history,
    })

    # 5. Parse confidence
    answer, confidence_label, confidence_score = _parse_confidence(raw_answer)

    # 6. Save to memory
    _save_to_memory(session_id, query, answer)

    # 7. Gather sources
    sources = []
    for doc in reranked_docs:
        sources.append({
            "source": doc.metadata.get("file_name", doc.metadata.get("source", "Unknown")),
            "file_type": doc.metadata.get("file_type", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "content_preview": doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content,
        })

    latency = round(time.time() - start_time, 2)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence_label,
        "confidence_score": confidence_score,
        "latency_seconds": latency,
        "retrieval_method": "hybrid",
    }
