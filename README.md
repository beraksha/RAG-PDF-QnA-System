# RAG PDF Q&A

Retrieval-augmented generation pipeline that answers questions about your PDFs, with page-level citations.

**Stack:** Python, LangChain, ChromaDB, Hugging Face Transformers (sentence-transformers embeddings), OpenAI.

> You can use any LLM Model and it's respective API Key, for this project I've used OpenAI.

## How it works
1. **Ingest** – `PyPDFLoader` reads each page, `RecursiveCharacterTextSplitter` makes ~1000-char overlapping chunks.
2. **Embed + store** – chunks are embedded locally with a Hugging Face model and persisted in ChromaDB.
3. **Retrieve** – the question is embedded and the top-k most similar chunks are fetched.
4. **Generate** – the chunks go into a prompt that tells the LLM to answer only from context and cite `[file, page]`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
```

## Usage
```bash
# put PDFs in data/ then:
python -m src.rag.ingest
python -m src.rag.cli
pytest
```

## Status / roadmap
- [x] PDF ingestion, chunking, local embeddings, persistent vector store
- [x] Cited answers, CLI
- [ ] Web UI (Streamlit)
- [ ] Re-ranking and hybrid (keyword + vector) search
- [ ] OCR for scanned PDFs
- [ ] Retrieval evaluation set (hit-rate / MRR)
