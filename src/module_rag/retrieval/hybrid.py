"""Hybrid retriever: BM25 sparse + dense vector ensemble with RRF fusion.

Dense retrieval excels at semantic similarity but misses exact keyword matches.
BM25 is the opposite — strong on rare keywords, weak on paraphrase. Combining
both with Reciprocal Rank Fusion (RRF) captures the strengths of each, which
typically lifts recall by 5–15% over either alone — especially on technical
content where jargon and exact terms matter.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from module_rag.config import get_settings
from module_rag.ingestion.pipeline import _COLLECTION

_K = 4
_FETCH_K = 20


def _get_chroma(persist_dir: Path):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir / "chroma"),
    )


def _fetch_all_docs(vector_store) -> list[Document]:
    """Pull every document out of Chroma to build the BM25 index."""
    results = vector_store._collection.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(results["documents"] or [], results["metadatas"] or [], strict=False)
        if text
    ]


def get_hybrid_retriever(persist_dir: Path, k: int = _K):
    """Return an EnsembleRetriever combining BM25 and Chroma (weights 0.5/0.5).

    Args:
        persist_dir: Root processed directory.
        k: Number of documents to return.

    Returns:
        LangChain EnsembleRetriever using RRF fusion.
    """
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    vector_store = _get_chroma(persist_dir)
    all_docs = _fetch_all_docs(vector_store)

    bm25 = BM25Retriever.from_documents(all_docs, k=k)
    dense = vector_store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def get_hybrid_retriever_wide(persist_dir: Path) -> object:
    """Return a hybrid retriever with k=20 for use before reranking."""
    return get_hybrid_retriever(persist_dir, k=_FETCH_K)
