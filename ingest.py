"""
Document Ingestion Module.

Handles loading documents from a directory, splitting them into chunks,
generating embeddings via Vertex AI, and persisting them in ChromaDB.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    GCP_PROJECT,
    GCP_LOCATION,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)

console = Console()


def load_documents(directory: Optional[Path] = None) -> list:
    """
    Load PDF and TXT documents from the specified directory.

    Args:
        directory: Path to the documents folder. Defaults to DOCUMENTS_DIR.

    Returns:
        List of loaded LangChain Document objects.
    """
    doc_dir = directory or DOCUMENTS_DIR

    if not doc_dir.exists():
        console.print(f"[red]✗ Directory not found:[/red] {doc_dir}")
        sys.exit(1)

    all_docs = []

    # Load TXT files
    txt_files = list(doc_dir.glob("*.txt"))
    if txt_files:
        console.print(f"  📄 Found [cyan]{len(txt_files)}[/cyan] TXT file(s)")
        txt_loader = DirectoryLoader(
            str(doc_dir),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        all_docs.extend(txt_loader.load())

    # Load PDF files
    pdf_files = list(doc_dir.glob("*.pdf"))
    if pdf_files:
        console.print(f"  📄 Found [cyan]{len(pdf_files)}[/cyan] PDF file(s)")
        pdf_loader = DirectoryLoader(
            str(doc_dir),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
        )
        all_docs.extend(pdf_loader.load())

    if not all_docs:
        console.print("[yellow]⚠ No documents found in the directory.[/yellow]")

    return all_docs


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def get_embeddings() -> VertexAIEmbeddings:
    """Create and return a Vertex AI embeddings instance."""
    return VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )


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
    """
    Load and return the existing persisted vector store.

    Returns:
        ChromaDB vector store instance.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vectorstore


def ingest_pipeline(directory: Optional[Path] = None) -> dict:
    """
    Run the full ingestion pipeline: load → chunk → embed → store.

    Args:
        directory: Path to the documents folder.

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

    stats = {"documents": 0, "chunks": 0, "status": "success"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:

        # Step 1: Load documents
        task = progress.add_task("Loading documents...", total=3)
        documents = load_documents(directory)
        stats["documents"] = len(documents)
        progress.update(task, advance=1, description="✅ Documents loaded")

        if not documents:
            stats["status"] = "no_documents"
            return stats

        # Step 2: Chunk documents
        progress.update(task, description="Splitting into chunks...")
        chunks = chunk_documents(documents)
        stats["chunks"] = len(chunks)
        progress.update(task, advance=1, description="✅ Chunks created")

        # Step 3: Embed and store
        progress.update(task, description="Embedding & storing in vector DB...")
        create_vectorstore(chunks)
        progress.update(task, advance=1, description="✅ Vector store created")

    console.print()
    console.print(f"  📊 Documents loaded: [green]{stats['documents']}[/green]")
    console.print(f"  🧩 Chunks created:   [green]{stats['chunks']}[/green]")
    console.print(f"  💾 Vector store:      [green]{VECTORSTORE_DIR}[/green]")
    console.print()
    console.print("[bold green]✅ Ingestion complete![/bold green]")

    return stats
