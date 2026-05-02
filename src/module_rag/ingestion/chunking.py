"""Chunking strategies for ingested documents.

Slide documents (title / body / notes) are kept as single chunks because they
are already small and semantically atomic — splitting them would break the
structure that makes slide-aware retrieval useful.

All other documents are split with RecursiveCharacterTextSplitter using a
700-token window and 100-token overlap (measured by the cl100k_base tokeniser,
which is a reasonable proxy for sentence-transformer token counts).
"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
_SLIDE_PREFIX = "slide_"


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk a list of Documents, preserving and extending their metadata.

    Slide documents are passed through unsplit. All other documents are split
    by the recursive character splitter. Every output chunk gains a ``chunk_id``
    UUID field in its metadata.

    Args:
        docs: Raw documents from any loader.

    Returns:
        List of chunks ready for embedding.
    """
    splitter = _splitter()
    chunks: list[Document] = []

    for doc in docs:
        content_type = doc.metadata.get("content_type", "")
        if content_type.startswith(_SLIDE_PREFIX):
            chunks.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "chunk_id": str(uuid.uuid4())},
                )
            )
        else:
            for split in splitter.split_documents([doc]):
                split.metadata["chunk_id"] = str(uuid.uuid4())
                chunks.append(split)

    return chunks
