"""Contextual compression retriever.

Standard retrieval returns whole chunks regardless of whether every sentence
is relevant. Contextual compression uses an LLM to extract only the sentences
from each chunk that actually address the query — reducing noise in the context
window and improving faithfulness.

Trade-off: one extra LLM call per retrieved chunk, so 4× slower than the base
retriever. Best used when answer quality matters more than latency.
"""

from __future__ import annotations

from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def get_compression_retriever(persist_dir: Path) -> ContextualCompressionRetriever:
    """Return a ContextualCompressionRetriever over the hybrid retriever.

    Each retrieved chunk is passed through an LLMChainExtractor that strips
    irrelevant sentences before the context reaches the generation chain.

    Args:
        persist_dir: Root processed directory.

    Returns:
        ContextualCompressionRetriever wrapping a hybrid base retriever.
    """
    from langchain_ollama import ChatOllama

    from module_rag.config import get_settings
    from module_rag.retrieval.hybrid import get_hybrid_retriever

    settings = get_settings()
    llm = ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    compressor = LLMChainExtractor.from_llm(llm)
    base = get_hybrid_retriever(persist_dir)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
