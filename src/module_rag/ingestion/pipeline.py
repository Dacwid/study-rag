"""Ingestion pipeline: load → chunk → embed → store in ChromaDB.

Idempotency is handled via an MD5 hash store (JSON file in persist_dir).
Files whose hash hasn't changed are skipped. Files that have changed have
their old chunks deleted from Chroma before new ones are added.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from module_rag.config import get_settings
from module_rag.ingestion.chunking import chunk_documents
from module_rag.ingestion.loaders import SUPPORTED_EXTENSIONS, load_file

console = Console()

_HASH_STORE = "file_hashes.json"
_COLLECTION = "module_notes"


@dataclass
class IngestStats:
    """Summary of one ingestion run."""

    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_hash_store(persist_dir: Path) -> dict[str, str]:
    store = persist_dir / _HASH_STORE
    return json.loads(store.read_text()) if store.exists() else {}


def _save_hash_store(persist_dir: Path, store: dict[str, str]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / _HASH_STORE).write_text(json.dumps(store, indent=2))


def ingest(raw_dir: Path, persist_dir: Path) -> IngestStats:
    """Walk raw_dir, embed new/changed files, and store chunks in ChromaDB.

    Args:
        raw_dir: Directory containing raw documents (searched recursively).
        persist_dir: Root directory for ChromaDB and the hash store.

    Returns:
        IngestStats with file counts, chunk counts, and elapsed time.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    stats = IngestStats()
    t0 = time.time()

    hash_store = _load_hash_store(persist_dir)

    paths = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not paths:
        console.print("[yellow]No supported files found.[/yellow]")
        stats.elapsed_seconds = time.time() - t0
        return stats

    console.print(f"Found [bold]{len(paths)}[/bold] file(s). Initialising embeddings…")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    chroma_dir = persist_dir / "chroma"

    vector_store = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    for path in paths:
        file_key = str(path)
        current_hash = _file_hash(path)

        if hash_store.get(file_key) == current_hash:
            stats.files_skipped += 1
            console.print(f"  [dim]skip[/dim]  {path.name}")
            continue

        try:
            if file_key in hash_store:
                _delete_by_source(vector_store, file_key)

            docs = load_file(path)
            chunks = chunk_documents(docs)

            if chunks:
                vector_store.add_documents(
                    documents=chunks,
                    ids=[c.metadata["chunk_id"] for c in chunks],
                )

            hash_store[file_key] = current_hash
            stats.files_processed += 1
            stats.chunks_created += len(chunks)
            console.print(f"  [green]load[/green]  {path.name} → {len(chunks)} chunks")

        except Exception as exc:
            msg = f"{path.name}: {exc}"
            stats.errors.append(msg)
            console.print(f"  [red]error[/red] {msg}")

    _save_hash_store(persist_dir, hash_store)
    stats.elapsed_seconds = time.time() - t0
    return stats


def _delete_by_source(vector_store: object, source_file: str) -> None:
    """Delete all Chroma documents whose source_file matches."""
    try:
        collection = vector_store._collection  # type: ignore[attr-defined]
        results = collection.get(where={"source_file": source_file})
        if results and results.get("ids"):
            collection.delete(ids=results["ids"])
    except Exception as exc:
        console.print(f"  [yellow]warn[/yellow] could not delete old chunks: {exc}")
