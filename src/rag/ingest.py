"""Load PDFs -> split into chunks -> embed -> store in ChromaDB."""
import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_pdf(path: Path):
    """Return one Document per page, tagged with the source file name."""
    pages = PyPDFLoader(str(path)).load()
    for p in pages:
        p.metadata["source"] = path.name
    return pages


def split_documents(docs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def get_store():
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def ingest(paths):
    store = get_store()
    total = 0
    for path in paths:
        chunks = split_documents(load_pdf(path))
        if not chunks:
            print(f"[skip] {path.name}: no extractable text (scanned PDF? needs OCR)")
            continue
        store.add_documents(chunks)
        total += len(chunks)
        print(f"[ok] {path.name}: {len(chunks)} chunks")
    print(f"Done. {total} chunks added.")


def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs into the vector store")
    ap.add_argument("pdfs", nargs="*", type=Path, help="PDF files (default: everything in data/)")
    args = ap.parse_args()
    paths = args.pdfs or sorted(config.DATA_DIR.glob("*.pdf"))
    if not paths:
        raise SystemExit("No PDFs found. Put files in data/ or pass paths.")
    ingest(paths)


if __name__ == "__main__":
    main()
