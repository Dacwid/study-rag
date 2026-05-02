"""Baseline retriever: vanilla cosine-similarity search over ChromaDB.

This is the simplest possible retrieval strategy — embed the query with the
same model used at ingest time and return the top-k most similar chunks.
It serves as the performance floor that every other technique is measured against.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from module_rag.config import get_settings
from module_rag.ingestion.pipeline import _COLLECTION

_K = 4


def get_baseline_retriever(persist_dir: Path) -> VectorStoreRetriever:
    """Return a Chroma similarity-search retriever (k=4).

    Args:
        persist_dir: Root processed directory containing the ``chroma/`` sub-folder.

    Returns:
        A LangChain retriever ready to call ``.invoke(query)``.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir / "chroma"),
    )
    return vector_store.as_retriever(search_kwargs={"k": _K})
