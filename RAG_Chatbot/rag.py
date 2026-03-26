import os
import re
import uuid
import pymupdf4llm
import ollama
import chromadb
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EMBED_MODEL = "qwen3-embedding:0.6b"
GENERATE_MODEL = "ministral-3"

COLLECTION_NAME = "local_pdf_rag"
PDF_FOLDER = "data/"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 300

TOP_K = 5
MAX_HISTORY_TURNS = 6   # last 3 user+assistant exchanges

# ─────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

_client = chromadb.Client()
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

# in-memory session store (replace with Redis/DB in production)
_sessions: Dict[str, List[Dict]] = {}

# ─────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────

def get_session(session_id: Optional[str]) -> str:
    """Create or retrieve a session."""
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in _sessions:
        _sessions[session_id] = []

    return session_id


def get_history(session_id: str) -> List[Dict]:
    return _sessions.get(session_id, [])


def append_history(session_id: str, role: str, content: str):
    _sessions.setdefault(session_id, []).append({
        "role": role,
        "content": content
    })


def build_history_text(history: List[Dict]) -> str:
    """Convert structured history into prompt text."""
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []

    for msg in recent:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────

def ingest_pdf(file_path: str) -> int:
    """Ingest a single PDF into the vector store."""
    md_text = pymupdf4llm.to_markdown(
        file_path,
        use_ocr=True,
        ocr_language="eng"
    )

    clean_text = re.sub(
        r"\*\*==> picture.*?<==\*\*",
        "",
        md_text,
        flags=re.DOTALL
    )

    chunks = text_splitter.create_documents([clean_text])
    file_name = os.path.basename(file_path)

    for i, chunk in enumerate(chunks):
        emb = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=chunk.page_content
        )["embedding"]

        _collection.add(
            ids=[f"{file_name}_{i}"],
            embeddings=[emb],
            documents=[chunk.page_content],
            metadatas=[{
                "source": file_name,
                "chunk": i,
                "type": "document"
            }]
        )

    return len(chunks)


def ingest_all_pdfs(folder: str = PDF_FOLDER) -> dict:
    results = {}
    if not os.path.isdir(folder):
        return results

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            count = ingest_pdf(path)
            results[file] = count

    return results


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def embed(text: str):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def retrieve(question: str, n_results: int = TOP_K):
    query_emb = embed(question)

    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    filtered_docs = []
    filtered_metas = []

    for doc, meta, dist in zip(docs, metas, distances):
        if dist < 0.7:  # tune this (lower = stricter)
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    return filtered_docs, filtered_metas


def build_context(docs: List[str]) -> str:
    return "\n\n".join(docs)


# ─────────────────────────────────────────────────────────────
# PROMPTING
# ─────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, history_text: str) -> str:
    if context.strip():
        return f"""
You are a helpful assistant.

Use the context below ONLY if it is relevant to the question.

Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""
    else:
        return f"""
You are a helpful assistant.

Conversation:
{history_text}

Question:
{question}

Answer:
"""


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    response = ollama.generate(
        model=GENERATE_MODEL,
        prompt=prompt
    )
    return response["response"].strip()


# ─────────────────────────────────────────────────────────────
# MAIN QUERY PIPELINE
# ─────────────────────────────────────────────────────────────

def query(
    question: str,
    session_id: Optional[str] = None,
    n_results: int = TOP_K
) -> dict:
    # 1. session
    session_id = get_session(session_id)
    history = get_history(session_id)

    # 2. retrieval
    docs, metas = retrieve(question, n_results=n_results)

    if len(docs) == 0:
        context = ""
    else:
        context = build_context(docs)

    # 3. history
    history_text = build_history_text(history)

    # 4. prompt
    prompt = build_prompt(question, context, history_text)

    # 5. generate
    answer = generate(prompt)

    # 6. update memory
    append_history(session_id, "user", question)
    append_history(session_id, "assistant", answer)

    # 7. sources
    sources = list({m["source"] for m in metas})

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def clear_collection():
    global _collection
    try:
        _client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def collection_count() -> int:
    return _collection.count()


def clear_sessions():
    """Reset all conversation memory."""
    global _sessions
    _sessions = {}