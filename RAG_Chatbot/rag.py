import os
import re
import pymupdf4llm
import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

EMBED_MODEL = "qwen3-embedding:0.6b"
GENERATE_MODEL = "mistral"
PDF_FOLDER = "data/"
COLLECTION_NAME = "local_pdf_rag"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 300

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

_client = chromadb.Client()
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def get_collection():
    return _collection


def ingest_pdf(file_path: str) -> int:
    """Ingest a single PDF into the vector store. Returns number of chunks added."""
    md_text = pymupdf4llm.to_markdown(file_path, use_ocr=True, ocr_language="eng")
    clean_text = re.sub(r"\*\*==> picture.*?<==\*\*", "", md_text, flags=re.DOTALL)
    chunks = text_splitter.create_documents([clean_text])
    file_name = os.path.basename(file_path)

    for i, chunk in enumerate(chunks):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk.page_content)["embedding"]
        _collection.add(
            ids=[f"{file_name}_chunk_{i}"],
            embeddings=[emb],
            documents=[chunk.page_content],
            metadatas=[{"source": file_name, "chunk": i}]
        )

    return len(chunks)


def ingest_all_pdfs(folder: str = PDF_FOLDER) -> dict:
    """Ingest all PDFs in a folder. Returns a summary dict."""
    results = {}
    if not os.path.isdir(folder):
        return results
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            count = ingest_pdf(path)
            results[file] = count
    return results


def query(question: str, n_results: int = 5) -> dict:
    """Run a RAG query. Returns answer string and source metadata."""
    query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = _collection.query(query_embeddings=[query_emb], n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join(docs)

    prompt = f"""You are a precise teaching assistant.

Answer the question using ONLY the context below.
Give a clear and structured explanation.
If the answer is not in the context, say so honestly.

Context:
{context}

Question:
{question}

Answer:
"""
    output = ollama.generate(model=GENERATE_MODEL, prompt=prompt)
    sources = list({m["source"] for m in metas})

    return {
        "answer": output["response"].strip(),
        "sources": sources,
    }


def clear_collection():
    """Delete and recreate the collection (wipes all ingested data)."""
    global _collection
    try:
        _client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def collection_count() -> int:
    """Return number of chunks currently stored."""
    return _collection.count()
