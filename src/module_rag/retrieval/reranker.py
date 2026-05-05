"""Cross-encoder reranker retriever.

Dense bi-encoder retrieval scores query and document independently, so their
representations never interact. A cross-encoder (reranker) receives the query
and document *together*, letting the model reason over both at once — this is
far more accurate but too slow to run over the whole corpus.

The two-stage pipeline solves this: bi-encoder retrieves a wide candidate set
(top-20) cheaply, cross-encoder reranks to top-4 precisely. This typically
outperforms either technique alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from module_rag.config import get_settings

_FETCH_K = 20
_TOP_K = 4


class RerankerRetriever(BaseRetriever):
    """Wraps any base retriever with a cross-encoder reranking step."""

    base_retriever: Any
    reranker_model: str
    top_k: int = _TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from sentence_transformers import CrossEncoder

        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        cross_encoder = CrossEncoder(self.reranker_model)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_k]]


def get_reranker_retriever(persist_dir: Path, base_retriever=None) -> RerankerRetriever:
    """Return a RerankerRetriever over the given base retriever.

    If no base retriever is provided, uses the baseline retriever with k=20
    as the candidate source.

    Args:
        persist_dir: Root processed directory.
        base_retriever: Upstream retriever that fetches candidates. Defaults to
            baseline with k=20.

    Returns:
        RerankerRetriever that scores and re-orders top candidates.
    """
    if base_retriever is None:
        from module_rag.retrieval.baseline import get_baseline_retriever

        base_retriever = get_baseline_retriever(persist_dir)
        base_retriever.search_kwargs["k"] = _FETCH_K

    settings = get_settings()
    return RerankerRetriever(
        base_retriever=base_retriever,
        reranker_model=settings.reranker_model,
        top_k=_TOP_K,
    )


def get_hybrid_rerank_retriever(persist_dir: Path) -> RerankerRetriever:
    """Convenience: hybrid top-20 → cross-encoder rerank to top-4.

    This is the default production mode — best recall/precision trade-off
    in benchmarks.
    """
    from module_rag.retrieval.hybrid import get_hybrid_retriever_wide

    wide_hybrid = get_hybrid_retriever_wide(persist_dir)
    settings = get_settings()
    return RerankerRetriever(
        base_retriever=wide_hybrid,
        reranker_model=settings.reranker_model,
        top_k=_TOP_K,
    )
