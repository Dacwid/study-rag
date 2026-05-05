"""Metadata-filter retriever: LLM-powered query router.

Many questions implicitly scope to a module or week: "explain the merge sort
from week 3" or "what did the algorithms lecture say about Big-O". Running
dense retrieval without those filters wastes context on irrelevant modules.

This module uses a small structured LLM call to extract ``module`` and ``week``
from the query. When found, they are passed as Chroma ``where`` clauses so only
matching chunks are searched. Falls back to unfiltered retrieval when no scope
is detected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_WEEK_RE = re.compile(r"\bweek\s*(\d+)\b", re.IGNORECASE)
_MODULE_RE = re.compile(r"\b([a-z]{2,}\d+)\b", re.IGNORECASE)


def _extract_filters_regex(query: str) -> dict:
    """Extract week/module filters from a query using regex (no LLM)."""
    filters: dict = {}
    week_m = _WEEK_RE.search(query)
    if week_m:
        filters["week"] = int(week_m.group(1))
    module_m = _MODULE_RE.search(query)
    if module_m:
        filters["module"] = module_m.group(1).lower()
    return filters


def _extract_filters_llm(query: str, llm: Any) -> dict:
    """Use a structured LLM call to extract week/module when regex misses."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract metadata filters from the query. "
                'Reply with JSON only, e.g. {"week": 3, "module": "cs101"}. '
                "Use null for fields not mentioned. Reply {} if nothing found.",
            ),
            ("human", "{query}"),
        ]
    )
    import json

    raw = (prompt | llm | StrOutputParser()).invoke({"query": query})
    try:
        parsed = json.loads(raw.strip())
        return {k: v for k, v in parsed.items() if v is not None}
    except (json.JSONDecodeError, AttributeError):
        return {}


class MetadataFilterRetriever(BaseRetriever):
    """Wraps a Chroma vector store and applies metadata filters when detected."""

    persist_dir: Path
    k: int = 4
    use_llm_router: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        from module_rag.config import get_settings
        from module_rag.ingestion.pipeline import _COLLECTION

        settings = get_settings()
        filters = _extract_filters_regex(query)

        if not filters and self.use_llm_router:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
            filters = _extract_filters_llm(query, llm)

        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vector_store = Chroma(
            collection_name=_COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir / "chroma"),
        )

        search_kwargs: dict = {"k": self.k}
        if filters:
            search_kwargs["filter"] = filters

        return vector_store.as_retriever(search_kwargs=search_kwargs).invoke(query)


def get_metadata_filter_retriever(persist_dir: Path, use_llm_router: bool = False):
    """Return a metadata-aware retriever that scopes search by week/module.

    Args:
        persist_dir: Root processed directory.
        use_llm_router: If True, fall back to an LLM when regex finds nothing.

    Returns:
        MetadataFilterRetriever with automatic filter extraction.
    """
    return MetadataFilterRetriever(persist_dir=persist_dir, use_llm_router=use_llm_router)
