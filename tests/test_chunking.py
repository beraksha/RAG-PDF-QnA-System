from langchain_core.documents import Document

from src.rag.ingest import split_documents
from src.rag.query import format_context


def test_split_respects_chunk_size():
    doc = Document(page_content="word " * 1000, metadata={"source": "a.pdf", "page": 0})
    chunks = split_documents([doc], chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)


def test_format_context_has_citation():
    d = Document(page_content="hello", metadata={"source": "a.pdf", "page": 2})
    assert "[a.pdf, p.3]" in format_context([d])
