import os
import re
import uuid
import pymupdf4llm
import ollama
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
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

TOP_K = 3
MAX_HISTORY_TURNS = 6   # last 3 user+assistant exchanges

# Maximum number of chunks fed to the model for summarization.
# Increase if your model has a large context window.
SUMMARY_CHUNK_LIMIT = 20

# ─────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_store")

_client = chromadb.PersistentClient(path=CHROMA_PATH)
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

EMBED_WORKERS = int(os.environ.get("EMBED_WORKERS", "4"))


def _already_ingested(file_name: str) -> bool:
    results = _collection.get(where={"source": file_name}, limit=1)
    return len(results["ids"]) > 0


def _embed_chunk(args: Tuple[int, str]) -> Tuple[int, list]:
    i, text = args
    emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    return i, emb


def ingest_pdf(file_path: str, force: bool = False) -> int:
    file_name = os.path.basename(file_path)

    if not force and _already_ingested(file_name):
        existing = _collection.get(where={"source": file_name})
        return len(existing["ids"])

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
    texts = [c.page_content for c in chunks]

    embeddings: List[list] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(_embed_chunk, (i, t)): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            i, emb = future.result()
            embeddings[i] = emb

    _collection.add(
        ids=[f"{file_name}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"source": file_name, "chunk": i, "type": "document"}
            for i in range(len(chunks))
        ]
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
    """Standard vector-search retrieval."""
    query_emb = embed(question)

    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    filtered_docs  = []
    filtered_metas = []

    for doc, meta, dist in zip(docs, metas, distances):
        if dist < 0.5:
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    return filtered_docs, filtered_metas


def retrieve_all_chunks_for_source(file_name: str) -> Tuple[List[str], List[Dict]]:
    """
    Fetch every stored chunk for a given source file, ordered by chunk index.
    Used for whole-document summarization — bypasses vector search entirely.
    """
    results = _collection.get(
        where={"source": file_name},
        include=["documents", "metadatas"]
    )
    if not results["ids"]:
        return [], []

    pairs = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1].get("chunk", 0)
    )
    docs  = [p[0] for p in pairs]
    metas = [p[1] for p in pairs]
    return docs, metas


def detect_summarization_target(question: str) -> Optional[str]:
    """
    Return the source file name if the question is a summarization request
    targeting a known ingested document, otherwise return None.

    Matches patterns like:
      "Summarize lecture1.pdf"
      "Give me a summary of Lecture 1"
      "Can you summarise week3.pdf?"
      "What is lecture1 about?"
    """
    q_lower = question.lower()

    is_summary_request = any(kw in q_lower for kw in (
        "summarize", "summarise", "summary", "overview",
        "what is", "what's", "about", "outline", "recap"
    ))
    if not is_summary_request:
        return None

    # Get all known source names from the collection
    all_metas     = _collection.get(include=["metadatas"])["metadatas"]
    known_sources = {m["source"] for m in all_metas if "source" in m}

    # 1. Exact filename match (e.g. "lecture1.pdf")
    for source in known_sources:
        if source.lower() in q_lower:
            return source

    # 2. Stem match — filename without extension (e.g. "lecture1")
    for source in known_sources:
        stem = os.path.splitext(source)[0].lower()
        if stem and stem in q_lower:
            return source

    return None


def build_context(docs: List[str]) -> str:
    return "\n\n".join(docs)


# ─────────────────────────────────────────────────────────────
# PROMPTING
# ─────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, history_text: str) -> str:
    if context.strip():
        return f"""You are a helpful teaching assistant.

Use the context below ONLY if it is relevant to the question.

Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""
    else:
        return f"""You are a helpful teaching assistant.

Conversation:
{history_text}

Question:
{question}

Answer:"""


def build_summary_prompt(file_name: str, context: str, history_text: str) -> str:
    return f"""You are a helpful teaching assistant. A student has asked you to summarize a document.

Document name: {file_name}

Below is the full content of the document split into sections. Read all sections and produce a well-structured summary covering:
- The main topic and purpose of the document
- Key concepts, arguments, or findings
- Any important details a student should know

Conversation:
{history_text}

Document content:
{context}

Summary:"""


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    response = ollama.generate(model=GENERATE_MODEL, prompt=prompt)
    return response["response"].strip()


# ─────────────────────────────────────────────────────────────
# MAIN QUERY PIPELINE
# ─────────────────────────────────────────────────────────────

def query(
    question: str,
    session_id: Optional[str] = None,
    n_results: int = TOP_K
) -> dict:
    """
    Run the full RAG pipeline and return a result dict.

    Automatically routes summarization requests (e.g. "Summarize lecture1.pdf")
    to a full-document retrieval path instead of vector search.
    """
    import time
    t0 = time.time()

    session_id   = get_session(session_id)
    history      = get_history(session_id)
    history_text = build_history_text(history)

    # ── Route: summarization vs. standard retrieval ───────────────────────
    summary_target = detect_summarization_target(question)

    if summary_target:
        docs, metas = retrieve_all_chunks_for_source(summary_target)

        if not docs:
            answer  = (
                f"I couldn't find any content for '{summary_target}' in the knowledge base. "
                "Please make sure the document has been ingested."
            )
            sources = []
            docs    = []
        else:
            # Cap chunks to stay within the model's context window.
            # Uses evenly-spaced sampling to preserve coverage across the doc.
            if len(docs) > SUMMARY_CHUNK_LIMIT:
                step  = len(docs) / SUMMARY_CHUNK_LIMIT
                docs  = [docs[int(i * step)]  for i in range(SUMMARY_CHUNK_LIMIT)]
                metas = [metas[int(i * step)] for i in range(SUMMARY_CHUNK_LIMIT)]

            context = build_context(docs)
            prompt  = build_summary_prompt(summary_target, context, history_text)
            answer  = generate(prompt)
            sources = [summary_target]

        mode = "summarize"

    else:
        # Standard vector-search path
        docs, metas = retrieve(question, n_results=n_results)
        context     = build_context(docs) if docs else ""
        prompt      = build_prompt(question, context, history_text)
        answer      = generate(prompt)
        sources     = list({m["source"] for m in metas})
        mode        = "retrieve"

    latency_ms = (time.time() - t0) * 1000

    append_history(session_id, "user", question)
    append_history(session_id, "assistant", answer)

    return {
        "answer":         answer,
        "sources":        sources,
        "session_id":     session_id,
        "context_chunks": docs,
        "latency_ms":     round(latency_ms, 1),
        "mode":           mode,
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
    global _sessions
    _sessions = {}