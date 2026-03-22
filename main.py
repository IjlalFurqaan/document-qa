"""
CLI Entry Point for the Document Q&A RAG Pipeline.

Provides commands for document ingestion and interactive querying.
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║       📚  Document Q&A — RAG Pipeline  📚           ║
║   Powered by Vertex AI • LangChain • ChromaDB       ║
╚══════════════════════════════════════════════════════╝
"""


def cmd_ingest(args):
    """Handle the 'ingest' command."""
    from ingest import ingest_pipeline

    doc_dir = Path(args.dir) if args.dir else None
    ingest_pipeline(doc_dir)


def cmd_query(args):
    """Handle the 'query' command."""
    from rag_chain import ask

    if args.question:
        # Single question mode
        _ask_and_display(args.question)
    else:
        # Interactive REPL mode
        _interactive_mode()


def _ask_and_display(question: str):
    """Ask a question and display the result."""
    console.print()
    console.print(f"[bold cyan]❓ Question:[/bold cyan] {question}")
    console.print()

    with console.status("[bold green]Thinking...", spinner="dots"):
        result = ask(question)

    # Display the answer
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]💡 Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Display sources
    if result["sources"]:
        table = Table(title="📎 Source Documents", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Source", style="green")
        table.add_column("Preview", style="dim")

        for i, src in enumerate(result["sources"], 1):
            table.add_row(
                str(i),
                src["source"],
                src["content_preview"][:100] + "...",
            )

        console.print(table)
    console.print()


def _interactive_mode():
    """Run interactive Q&A REPL."""
    console.print(
        Panel(
            "[bold]Interactive Q&A Mode[/bold]\n"
            "Type your questions and press Enter.\n"
            "Type [cyan]quit[/cyan] or [cyan]exit[/cyan] to stop.",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]You → [/bold cyan]").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye! 👋[/dim]")
                break

            _ask_and_display(question)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]")
            break


def main():
    """Main entry point with argument parsing."""
    console.print(BANNER, style="bold blue")

    parser = argparse.ArgumentParser(
        description="Document Q&A — RAG Pipeline powered by Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]💡 Quick start:[/yellow]\n"
            "  1. [cyan]python main.py ingest[/cyan]        — Index your documents\n"
            "  2. [cyan]python main.py query[/cyan]         — Start interactive Q&A\n"
            "  3. [cyan]python main.py query -q '...'[/cyan] — Ask a single question\n"
        )


if __name__ == "__main__":
    main()
