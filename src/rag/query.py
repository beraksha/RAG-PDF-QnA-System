"""Retrieve relevant chunks and ask the LLM to answer using only that context."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config
from .ingest import get_store

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources as [file, page].",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')}, p.{int(d.metadata.get('page', 0)) + 1}]\n{d.page_content}"
        for d in docs
    )


def answer(question: str, k: int = config.TOP_K):
    docs = get_store().similarity_search(question, k=k)
    if not docs:
        return "No documents indexed yet. Run `python -m src.rag.ingest` first.", []
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    msg = PROMPT.invoke({"context": format_context(docs), "question": question})
    return llm.invoke(msg).content, docs
