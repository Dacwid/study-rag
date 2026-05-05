"""Parent document retriever.

Embedding very long passages dilutes their meaning — a 1500-token chunk about
ten topics embeds as an average of all of them, so it retrieves poorly for any
one topic. Embedding small child chunks (300 tokens) gives precise matching,
but they lack the surrounding context the LLM needs to answer well.

The parent document retriever combines both: child chunks are embedded for
retrieval precision, but the LLM receives the full parent chunk (1500 tokens)
for richer context.

Note: parents are held in an InMemoryStore rebuilt each call from the existing
Chroma collection. Cold-start is ~2–5s depending on corpus size. A persistent
docstore (LocalFileStore) would eliminate this if latency becomes an issue.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def get_parent_doc_retriever(persist_dir: Path):
    """Return a ParentDocumentRetriever using InMemoryStore for parent chunks.

    Fetches all existing chunks from Chroma as the source corpus, splits them
    into small child chunks for embedding, and holds larger parent chunks in
    memory for the LLM context.

    Args:
        persist_dir: Root processed directory.

    Returns:
        ParentDocumentRetriever ready to call ``.invoke(query)``.
    """
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from module_rag.config import get_settings

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    from module_rag.ingestion.pipeline import _COLLECTION

    source_store = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir / "chroma"),
    )
    raw = source_store._collection.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
    source_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"] or [], raw["metadatas"] or [], strict=False)
        if text
    ]

    child_store = Chroma(
        collection_name="parent_doc_children",
        embedding_function=embeddings,
    )

    docstore = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=300, chunk_overlap=30
    )
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=1500, chunk_overlap=150
    )

    retriever = ParentDocumentRetriever(
        vectorstore=child_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(source_docs)
    return retriever
