from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="module-rag",
    help="CLI-based RAG system over university module slides and notes.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def hello() -> None:
    """Smoke-test command — confirms the CLI is wired up correctly."""
    from module_rag.config import get_settings

    settings = get_settings()
    console.print("[bold green]module-rag is alive![/bold green]")
    console.print(f"  LLM model  : [cyan]{settings.ollama_model}[/cyan]")
    console.print(f"  Embeddings : [cyan]{settings.embedding_model}[/cyan]")
    console.print(f"  Chroma dir : [cyan]{settings.chroma_dir}[/cyan]")


@app.command()
def ingest(
    raw_dir: Annotated[str, typer.Option(help="Directory containing raw documents.")] = "data/raw",
    processed_dir: Annotated[
        str, typer.Option(help="Directory for ChromaDB and hash store.")
    ] = "data/processed",
) -> None:
    """Ingest documents from raw-dir into ChromaDB."""
    from pathlib import Path

    from rich.table import Table

    from module_rag.ingestion.pipeline import ingest as run_ingest

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    if not raw_path.exists():
        console.print(f"[red]raw-dir not found:[/red] {raw_path}")
        raise typer.Exit(code=1)

    console.rule("[bold]Ingestion")
    stats = run_ingest(raw_path, processed_path)
    console.rule()

    table = Table(title="Ingestion Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Files processed", str(stats.files_processed))
    table.add_row("Files skipped (unchanged)", str(stats.files_skipped))
    table.add_row("Chunks created", str(stats.chunks_created))
    table.add_row("Time", f"{stats.elapsed_seconds:.1f}s")
    if stats.errors:
        table.add_row("[red]Errors[/red]", str(len(stats.errors)))
    console.print(table)

    for err in stats.errors:
        console.print(f"  [red]✗[/red] {err}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    mode: Annotated[str, typer.Option(help="Retrieval mode.")] = "baseline",
    processed_dir: Annotated[
        str, typer.Option(help="Processed data directory.")
    ] = "data/processed",
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print retrieved chunks.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print agent trajectory.")] = False,
) -> None:
    """Ask a question against the ingested documents."""
    from pathlib import Path

    from rich.panel import Panel
    from rich.rule import Rule

    from module_rag.config import get_settings
    from module_rag.generation.chains import build_chain, format_docs

    processed_path = Path(processed_dir)
    settings = get_settings()

    _IMPLEMENTED = {"baseline"}
    if mode not in _IMPLEMENTED:
        console.print(f"[yellow]Mode '{mode}' not yet implemented — using baseline.[/yellow]")
        mode = "baseline"

    if mode == "baseline":
        from module_rag.retrieval.baseline import get_baseline_retriever

        retriever = get_baseline_retriever(processed_path)

    docs = retriever.invoke(question)

    if show_context:
        console.print(Rule("Retrieved context"))
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source_file", "?")
            page = doc.metadata.get("page_or_slide", "?")
            console.print(f"[bold cyan][{i}] {src} p.{page}[/bold cyan]")
            console.print(doc.page_content.strip())
            console.print()
        console.print(Rule())

    chain = build_chain(settings)
    context = format_docs(docs)

    console.print(Rule(f"Answer ({mode})"))
    answer = chain.invoke({"question": question, "context": context})
    console.print(Panel(answer, border_style="green"))

    sources = {
        f"{d.metadata.get('source_file', '?')} p.{d.metadata.get('page_or_slide', '?')}"
        for d in docs
    }
    console.print(f"[dim]Sources: {', '.join(sorted(sources))}[/dim]")


@app.command()
def eval(
    modes: Annotated[
        str, typer.Option(help="Comma-separated retrieval modes to benchmark.")
    ] = "baseline,hybrid,hybrid+rerank,agent",
    sample: Annotated[int | None, typer.Option(help="Limit to N questions.")] = 10,
) -> None:
    """Run Ragas evaluation benchmark. (Phase 5)"""
    console.print("[yellow]Phase 5 not yet implemented.[/yellow]")


@app.command()
def compare() -> None:
    """Print the most recent evaluation results table. (Phase 5)"""
    console.print("[yellow]Phase 5 not yet implemented.[/yellow]")


if __name__ == "__main__":
    app()
