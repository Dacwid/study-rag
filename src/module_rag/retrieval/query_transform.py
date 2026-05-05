"""Query transformation retrievers: HyDE and Multi-Query.

Both techniques address the vocabulary gap between how a user phrases a
question and how the answer is phrased in the source material.

**HyDE** (Hypothetical Document Embeddings): Instead of embedding the raw
question, ask the LLM to write a short hypothetical answer, then embed *that*.
The hypothesis lives in the same semantic space as real answers, so it lands
closer to relevant passages than the question alone would.

**Multi-Query**: Generate N paraphrases of the original question, retrieve
for each, then deduplicate by chunk_id. Diverse queries probe different parts
of the embedding space, recovering documents that a single query would miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert. Write a concise, factual passage (3-5 sentences) "
            "that directly answers the question. Write as if from a textbook.",
        ),
        ("human", "{question}"),
    ]
)

_MULTIQUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate exactly 3 different phrasings of the user's question. "
            "Output only the questions, one per line, no numbering.",
        ),
        ("human", "{question}"),
    ]
)


class HyDERetriever(BaseRetriever):
    """Retrieves using a LLM-generated hypothetical answer as the query."""

    vector_store: Any
    llm: Any
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from langchain_core.output_parsers import StrOutputParser

        chain = _HYDE_PROMPT | self.llm | StrOutputParser()
        hypothesis = chain.invoke({"question": query})
        return self.vector_store.similarity_search(hypothesis, k=self.k)


class MultiQueryRetriever(BaseRetriever):
    """Retrieves with N query variants and deduplicates by chunk_id."""

    base_retriever: Any
    llm: Any
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from langchain_core.output_parsers import StrOutputParser

        chain = _MULTIQUERY_PROMPT | self.llm | StrOutputParser()
        raw = chain.invoke({"question": query})
        variants = [q.strip() for q in raw.strip().splitlines() if q.strip()]
        variants.append(query)

        seen: set[str] = set()
        results: list[Document] = []
        for variant in variants:
            for doc in self.base_retriever.invoke(variant):
                cid = doc.metadata.get("chunk_id", doc.page_content[:64])
                if cid not in seen:
                    seen.add(cid)
                    results.append(doc)

        return results[: self.k]


def get_hyde_retriever(persist_dir: Path) -> HyDERetriever:
    """Return a HyDE retriever backed by the Chroma vector store.

    Args:
        persist_dir: Root processed directory.

    Returns:
        HyDERetriever that embeds a hypothetical answer before searching.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama

    from module_rag.config import get_settings
    from module_rag.ingestion.pipeline import _COLLECTION

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir / "chroma"),
    )
    llm = ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    return HyDERetriever(vector_store=vector_store, llm=llm)


def get_multiquery_retriever(persist_dir: Path) -> MultiQueryRetriever:
    """Return a Multi-Query retriever backed by the baseline retriever.

    Args:
        persist_dir: Root processed directory.

    Returns:
        MultiQueryRetriever that deduplicates across 3 query variants + original.
    """
    from langchain_ollama import ChatOllama

    from module_rag.config import get_settings
    from module_rag.retrieval.baseline import get_baseline_retriever

    settings = get_settings()
    llm = ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    base = get_baseline_retriever(persist_dir)
    return MultiQueryRetriever(base_retriever=base, llm=llm)
