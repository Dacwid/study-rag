"""LCEL generation chain and document formatting utilities.

The chain receives pre-retrieved context (already formatted as a string) plus
the user question, and returns a grounded answer with inline source citations.
Keeping retrieval separate from generation means any retriever can be swapped
in without touching this file.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from module_rag.config import Settings

_SYSTEM = (
    "You are a precise study assistant. "
    "Answer using ONLY the context provided. "
    "After each factual claim, cite its source in the format [file, p.N]. "
    "If the context is insufficient, say so explicitly — do not guess."
)

_HUMAN = """\
Context:
{context}

Question: {question}

Answer (with citations):"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a numbered context block.

    Args:
        docs: Retrieved LangChain Documents.

    Returns:
        A single string with source markers before each chunk.
    """
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page_or_slide", "?")
        parts.append(f"[{source}, p.{page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def build_chain(settings: Settings) -> Runnable:
    """Build the LCEL generation chain for a given settings instance.

    Chain signature: ``{"question": str, "context": str} → str``

    Args:
        settings: Application settings (used for Ollama URL and model name).

    Returns:
        A LangChain Runnable that takes a dict and returns a string answer.
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    return _PROMPT | llm | StrOutputParser()
