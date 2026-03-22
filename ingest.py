"""
Document Ingestion Module.

Handles multi-format document loading (PDF, TXT, DOCX, Markdown, CSV),
intelligent chunking, embedding via Google GenAI, BM25 index generation,
and persistence in ChromaDB with duplicate detection.
"""

import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from rank_bm25 import BM25Okapi

from config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    BM25_INDEX_DIR,
    GCP_PROJECT,
    GCP_LOCATION,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)
console = Console()


# ─── File Hashing for Duplicate Detection ────────────────────────────────────

HASH_REGISTRY_PATH = VECTORSTORE_DIR / ".file_hashes.json"


def _compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()


def _load_hash_registry() -> dict:
    """Load the persisted file hash registry."""
    if HASH_REGISTRY_PATH.exists():
        with open(HASH_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_hash_registry(registry: dict):
    """Persist the file hash registry."""
    HASH_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ─── Document Loading ────────────────────────────────────────────────────────

def load_documents(directory: Optional[Path] = None, force: bool = False) -> tuple[list, dict]:
    """
    Load PDF, TXT, DOCX, Markdown, and CSV documents with duplicate detection.

    Args:
        directory: Path to the documents folder. Defaults to DOCUMENTS_DIR.
        force: If True, skip duplicate detection and reload all files.

    Returns:
        Tuple of (loaded documents, per-file status dict).
    """
    doc_dir = directory or DOCUMENTS_DIR

    if not doc_dir.exists():
        console.print(f"[red]✗ Directory not found:[/red] {doc_dir}")
        sys.exit(1)

    hash_registry = _load_hash_registry()
    all_docs = []
    file_status = {}  # filename -> "loaded" | "skipped" | "failed"

    loader_map = {
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".csv": (CSVLoader, {"encoding": "utf-8"}),
        ".md":  (UnstructuredMarkdownLoader, {}),
    }

    if DOCX_AVAILABLE:
        loader_map[".docx"] = (Docx2txtLoader, {})

    for ext in SUPPORTED_EXTENSIONS:
        files = list(doc_dir.glob(f"*{ext}"))
        if not files:
            continue

        loader_cls_info = loader_map.get(ext)
        if not loader_cls_info:
            continue

        loader_cls, loader_kwargs = loader_cls_info
        console.print(f"  📄 Found [cyan]{len(files)}[/cyan] {ext.upper().lstrip('.')} file(s)")

        for filepath in files:
            try:
                # Duplicate detection
                file_hash = _compute_file_hash(filepath)
                if not force and filepath.name in hash_registry and hash_registry[filepath.name] == file_hash:
                    file_status[filepath.name] = "skipped"
                    console.print(f"    [dim]⏭ Skipped (unchanged):[/dim] {filepath.name}")
                    continue

                # Load document
                loader = loader_cls(str(filepath), **loader_kwargs)
                docs = loader.load()

                # Enrich metadata
                for doc in docs:
                    doc.metadata.update({
                        "file_name": filepath.name,
                        "file_type": ext.lstrip("."),
                        "file_size_kb": round(filepath.stat().st_size / 1024, 1),
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    })

                all_docs.extend(docs)
                hash_registry[filepath.name] = file_hash
                file_status[filepath.name] = "loaded"
                logger.info(f"Loaded {len(docs)} page(s) from {filepath.name}")

            except Exception as e:
                file_status[filepath.name] = "failed"
                console.print(f"    [red]✗ Failed to load:[/red] {filepath.name} — {e}")
                logger.error(f"Failed to load {filepath.name}: {e}", exc_info=True)

    _save_hash_registry(hash_registry)

    if not all_docs:
        console.print("[yellow]⚠ No new documents to ingest.[/yellow]")

    return all_docs, file_status


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks with metadata enrichment.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects with chunk_index metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


# ─── Embeddings ──────────────────────────────────────────────────────────────

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and return a Google GenAI embeddings instance."""
    kwargs = {"model": EMBEDDING_MODEL}

    if GOOGLE_API_KEY:
        kwargs["google_api_key"] = GOOGLE_API_KEY
    else:
        kwargs["project"] = GCP_PROJECT
        kwargs["location"] = GCP_LOCATION

    return GoogleGenerativeAIEmbeddings(**kwargs)


# ─── Vector Store ────────────────────────────────────────────────────────────

def create_vectorstore(chunks: list) -> Chroma:
    """
    Embed document chunks and store them in ChromaDB.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        ChromaDB vector store instance.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vectorstore


def get_vectorstore() -> Chroma:
    """Load and return the existing persisted vector store."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vectorstore


# ─── BM25 Index ──────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list):
    """
    Build and persist a BM25 keyword index for hybrid search.

    Args:
        chunks: List of chunked Document objects.
    """
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    corpus = [chunk.page_content for chunk in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    # Persist the corpus and metadata for later retrieval
    index_data = {
        "corpus": corpus,
        "tokenized_corpus": tokenized_corpus,
        "metadata": [chunk.metadata for chunk in chunks],
    }

    with open(BM25_INDEX_DIR / "bm25_data.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    logger.info(f"BM25 index built with {len(corpus)} chunks")


def load_bm25_index() -> tuple[BM25Okapi, list, list] | None:
    """
    Load the persisted BM25 index.

    Returns:
        Tuple of (BM25Okapi instance, corpus, metadata) or None.
    """
    index_path = BM25_INDEX_DIR / "bm25_data.json"
    if not index_path.exists():
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    tokenized = index_data["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized)

    return bm25, index_data["corpus"], index_data["metadata"]


# ─── Vector Store Stats ─────────────────────────────────────────────────────

def get_vectorstore_stats() -> dict:
    """
    Get statistics about the current vector store.

    Returns:
        Dictionary with doc_count, chunk_count, file_types, last_ingested info.
    """
    stats = {
        "chunk_count": 0,
        "file_types": {},
        "documents": [],
        "last_ingested": None,
        "has_bm25": (BM25_INDEX_DIR / "bm25_data.json").exists(),
    }

    if not VECTORSTORE_DIR.exists():
        return stats

    try:
        vs = get_vectorstore()
        collection = vs._collection
        count = collection.count()
        stats["chunk_count"] = count

        if count > 0:
            # Sample metadata to gather stats
            results = collection.get(limit=min(count, 1000), include=["metadatas"])
            metadatas = results.get("metadatas", [])

            seen_files = set()
            latest_time = None

            for meta in metadatas:
                fname = meta.get("file_name", "unknown")
                ftype = meta.get("file_type", "unknown")
                ingested = meta.get("ingested_at")

                if fname not in seen_files:
                    seen_files.add(fname)
                    stats["documents"].append(fname)

                stats["file_types"][ftype] = stats["file_types"].get(ftype, 0) + 1

                if ingested and (latest_time is None or ingested > latest_time):
                    latest_time = ingested

            stats["last_ingested"] = latest_time

    except Exception as e:
        logger.error(f"Failed to get vectorstore stats: {e}")

    return stats


# ─── Ingestion Pipeline ─────────────────────────────────────────────────────

def ingest_pipeline(directory: Optional[Path] = None, force: bool = False) -> dict:
    """
    Run the full ingestion pipeline: load → chunk → embed → store → BM25 index.

    Args:
        directory: Path to the documents folder.
        force: If True, re-ingest all files regardless of duplicates.

    Returns:
        Dictionary with ingestion statistics.
    """
    console.print()
    console.print(
        Panel(
            "[bold cyan]📥 Document Ingestion Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )

    stats = {
        "documents": 0,
        "chunks": 0,
        "status": "success",
        "file_status": {},
        "elapsed_seconds": 0,
    }

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Step 1: Load documents
        task = progress.add_task("Loading documents...", total=4)
        documents, file_status = load_documents(directory, force=force)
        stats["documents"] = len(documents)
        stats["file_status"] = file_status
        progress.update(task, advance=1, description="✅ Documents loaded")

        if not documents:
            stats["status"] = "no_documents"
            return stats

        # Step 2: Chunk documents
        progress.update(task, description="Splitting into chunks...")
        chunks = chunk_documents(documents)
        stats["chunks"] = len(chunks)
        progress.update(task, advance=1, description="✅ Chunks created")

        # Step 3: Embed and store in ChromaDB
        progress.update(task, description="Embedding & storing in vector DB...")
        create_vectorstore(chunks)
        progress.update(task, advance=1, description="✅ Vector store created")

        # Step 4: Build BM25 keyword index
        progress.update(task, description="Building keyword search index...")
        build_bm25_index(chunks)
        progress.update(task, advance=1, description="✅ BM25 index built")

    elapsed = round(time.time() - start_time, 1)
    stats["elapsed_seconds"] = elapsed

    console.print()

    # Summary table
    summary = Table(title="📊 Ingestion Summary", show_lines=True, border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", style="green")
    summary.add_row("Documents loaded", str(stats["documents"]))
    summary.add_row("Chunks created", str(stats["chunks"]))
    summary.add_row("Vector store", str(VECTORSTORE_DIR))
    summary.add_row("BM25 index", str(BM25_INDEX_DIR))
    summary.add_row("Time elapsed", f"{elapsed}s")

    console.print(summary)

    # Per-file status
    if file_status:
        file_table = Table(title="📄 Per-File Status", show_lines=True, border_style="dim")
        file_table.add_column("File", style="cyan")
        file_table.add_column("Status")

        status_styles = {"loaded": "[green]✅ Loaded[/green]", "skipped": "[yellow]⏭ Skipped[/yellow]", "failed": "[red]✗ Failed[/red]"}

        for fname, fstatus in file_status.items():
            file_table.add_row(fname, status_styles.get(fstatus, fstatus))

        console.print(file_table)

    console.print()
    console.print("[bold green]✅ Ingestion complete![/bold green]")

    return stats
