from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import rag

app = Flask(__name__, static_folder="frontend")
CORS(app)

UPLOAD_FOLDER = "data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    """Return collection stats."""
    count = rag.collection_count()
    return jsonify({"chunk_count": count, "status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Answer a question using the RAG pipeline."""
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id") or None
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if rag.collection_count() == 0:
        return jsonify({"error": "No documents ingested yet. Please upload PDFs first."}), 400

    try:
        result = rag.query(question, session_id=session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Accepts either:
      - multipart/form-data with file(s) under the key 'files'
      - JSON { "folder": "data/" } to ingest an existing folder
    """
    # File upload path
    if request.files:
        uploaded = []
        for f in request.files.getlist("files"):
            if f.filename.endswith(".pdf"):
                dest = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(dest)
                count = rag.ingest_pdf(dest)
                uploaded.append({"file": f.filename, "chunks": count})
        if not uploaded:
            return jsonify({"error": "No valid PDF files received."}), 400
        return jsonify({"ingested": uploaded})

    # Folder path
    data = request.get_json(force=True, silent=True) or {}
    folder = data.get("folder", UPLOAD_FOLDER)
    results = rag.ingest_all_pdfs(folder)
    if not results:
        return jsonify({"error": f"No PDFs found in '{folder}'."}), 404
    return jsonify({"ingested": [{"file": k, "chunks": v} for k, v in results.items()]})


@app.route("/api/clear", methods=["POST"])
def clear():
    """Wipe the vector store."""
    rag.clear_collection()
    return jsonify({"message": "Collection cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
