"""
CLI Entry Point for the Document Q&A RAG Pipeline.

Provides commands for document ingestion, interactive querying,
vector store stats, and store management.
"""

import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from config import VECTORSTORE_DIR, BM25_INDEX_DIR, setup_logging

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         📚  Document Q&A — Enterprise RAG Pipeline  📚         ║
║   Hybrid Search • Re-Ranking • Conversation Memory • GenAI     ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─── Credential Check ────────────────────────────────────────────────────────

def _check_credentials():
    """Verify Google Cloud credentials are available."""
    try:
        import google.auth
        google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return True
    except Exception:
        console.print(
            Panel(
                "[bold red]✗ Google Cloud credentials not found[/bold red]\n\n"
                "Run the following command to authenticate:\n"
                "  [cyan]gcloud auth application-default login[/cyan]\n\n"
                "Or set the [cyan]GOOGLE_API_KEY[/cyan] environment variable in your [cyan].env[/cyan] file.\n\n"
                "For more details, see:\n"
                "  https://cloud.google.com/docs/authentication/external/set-up-adc",
                title="🔑 Authentication Required",
                border_style="red",
            )
        )
        return False


# ─── Ingest Command ─────────────────────────────────────────────────────────

def cmd_ingest(args):
    """Handle the 'ingest' command."""
    if not _check_credentials():
        return

    from ingest import ingest_pipeline

    doc_dir = Path(args.dir) if args.dir else None
    ingest_pipeline(doc_dir, force=args.force)


# ─── Query Command ──────────────────────────────────────────────────────────

def cmd_query(args):
    """Handle the 'query' command."""
    if not _check_credentials():
        return

    from rag_chain import ask

    if args.question:
        _ask_and_display(args.question)
    else:
        _interactive_mode()


def _ask_and_display(question: str, session_id: str = "cli"):
    """Ask a question and display the result with confidence and sources."""
    from rag_chain import ask

    console.print()
    console.print(f"[bold cyan]❓ Question:[/bold cyan] {question}")
    console.print()

    with console.status("[bold green]🔍 Searching & reasoning...", spinner="dots"):
        result = ask(question, session_id=session_id)

    # Confidence badge
    confidence = result.get("confidence", "MEDIUM")
    confidence_colors = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}
    confidence_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    color = confidence_colors.get(confidence, "yellow")
    icon = confidence_icons.get(confidence, "🟡")

    # Answer panel
    console.print(Panel(
        Markdown(result["answer"]),
        title=f"[bold green]💡 Answer[/bold green]  {icon} [{color}]{confidence} confidence[/{color}]",
        subtitle=f"⏱ {result.get('latency_seconds', '?')}s • 🔍 {result.get('retrieval_method', 'hybrid')}",
        border_style="green",
        padding=(1, 2),
    ))

    # Sources table
    if result["sources"]:
        table = Table(
            title="📎 Source Documents",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", style="cyan", width=3)
        table.add_column("Source", style="green")
        table.add_column("Type", style="magenta", width=6)
        table.add_column("Chunk", style="dim", width=6)
        table.add_column("Preview", style="dim", max_width=60)

        for i, src in enumerate(result["sources"], 1):
            table.add_row(
                str(i),
                src["source"],
                src.get("file_type", "?"),
                str(src.get("chunk_index", "?")),
                src["content_preview"][:80] + "...",
            )

        console.print(table)
    console.print()


def _interactive_mode():
    """Run interactive Q&A REPL with conversation memory."""
    console.print(
        Panel(
            "[bold]Interactive Q&A Mode[/bold]\n"
            "Type your questions and press Enter.\n"
            "Follow-up questions are supported via conversation memory.\n"
            "Type [cyan]quit[/cyan] or [cyan]exit[/cyan] to stop.\n"
            "Type [cyan]clear[/cyan] to reset conversation memory.",
            border_style="blue",
        )
    )

    session_id = "cli_interactive"

    while True:
        try:
            question = console.input("\n[bold cyan]You → [/bold cyan]").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye! 👋[/dim]")
                break
            if question.lower() == "clear":
                from rag_chain import clear_memory
                clear_memory(session_id)
                console.print("[yellow]🗑 Conversation memory cleared.[/yellow]")
                continue

            _ask_and_display(question, session_id=session_id)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]")
            break


# ─── Stats Command ──────────────────────────────────────────────────────────

def cmd_stats(args):
    """Handle the 'stats' command — display vector store statistics."""
    from ingest import get_vectorstore_stats

    stats = get_vectorstore_stats()

    console.print()
    console.print(
        Panel(
            "[bold cyan]📊 Vector Store Statistics[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(show_lines=True, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Total chunks", str(stats["chunk_count"]))
    table.add_row("Unique documents", str(len(stats["documents"])))
    table.add_row("BM25 index", "✅ Available" if stats["has_bm25"] else "❌ Not built")
    table.add_row("Last ingested", stats.get("last_ingested") or "N/A")

    console.print(table)

    if stats["file_types"]:
        type_table = Table(title="📁 Chunks by File Type", show_lines=True, border_style="dim")
        type_table.add_column("Type", style="magenta")
        type_table.add_column("Chunks", style="green")
        for ftype, count in stats["file_types"].items():
            type_table.add_row(ftype.upper(), str(count))
        console.print(type_table)

    if stats["documents"]:
        console.print("\n[bold]📄 Indexed Documents:[/bold]")
        for doc in stats["documents"]:
            console.print(f"  • [cyan]{doc}[/cyan]")

    console.print()


# ─── Clear Command ──────────────────────────────────────────────────────────

def cmd_clear(args):
    """Handle the 'clear' command — reset the vector store."""
    if not args.force:
        confirm = console.input(
            "[bold red]⚠ This will delete the vector store and BM25 index. "
            "Are you sure? (y/N): [/bold red]"
        ).strip().lower()
        if confirm != "y":
            console.print("[dim]Cancelled.[/dim]")
            return

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        console.print("[green]✅ Vector store deleted[/green]")

    if BM25_INDEX_DIR.exists():
        shutil.rmtree(BM25_INDEX_DIR)
        console.print("[green]✅ BM25 index deleted[/green]")

    console.print("[bold green]✅ All indexes cleared.[/bold green]")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    """Main entry point with argument parsing."""
    console.print(BANNER, style="bold blue")

    parser = argparse.ArgumentParser(
        description="Document Q&A — Enterprise RAG Pipeline powered by Google GenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the vector store",
    )
    ingest_parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to documents directory (default: ./documents)",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest all files (skip duplicate detection)",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the document knowledge base",
    )
    query_parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question (omit for interactive mode)",
    )

    # Stats command
    subparsers.add_parser(
        "stats",
        help="Display vector store statistics",
    )

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear the vector store and BM25 index",
    )
    clear_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]💡 Quick start:[/yellow]\n"
            "  1. [cyan]python main.py ingest[/cyan]          — Index your documents\n"
            "  2. [cyan]python main.py query[/cyan]           — Start interactive Q&A\n"
            "  3. [cyan]python main.py query -q '...'[/cyan]  — Ask a single question\n"
            "  4. [cyan]python main.py stats[/cyan]           — View vector store stats\n"
            "  5. [cyan]python main.py clear[/cyan]           — Reset all indexes\n"
        )


if __name__ == "__main__":
    main()
