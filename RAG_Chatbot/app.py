from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import rag
import eval as ev

app = Flask(__name__, static_folder="frontend")
CORS(app)

UPLOAD_FOLDER = "data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Controls whether every /api/chat response is automatically evaluated.
# Set to False if you only want on-demand or dataset-based evaluation.
AUTO_EVAL = os.environ.get("AUTO_EVAL", "true").lower() == "true"


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
    """
    Answer a question using the RAG pipeline.

    Optional request fields:
      - reference_answer  (str)  Ground-truth answer for context-recall scoring.
      - evaluate          (bool) Override AUTO_EVAL for this single request.
    """
    data = request.get_json(force=True)
    question         = (data.get("question") or "").strip()
    session_id       = data.get("session_id") or None
    reference_answer = data.get("reference_answer") or None
    run_eval         = data.get("evaluate", AUTO_EVAL)

    if not question:
        return jsonify({"error": "No question provided."}), 400

    if rag.collection_count() == 0:
        return jsonify({"error": "No documents ingested yet. Please upload PDFs first."}), 400

    try:
        result = rag.query(question, session_id=session_id)

        response = {
            "answer":     result["answer"],
            "sources":    result["sources"],
            "session_id": result["session_id"],
            "latency_ms": result["latency_ms"],
            "mode":       result.get("mode", "retrieve"),
            "blocked":    result.get("blocked", False),
        }
        if result.get("blocked"):
            response["block_layer"]  = result.get("block_layer")
            response["block_reason"] = result.get("block_reason")

        if run_eval:
            # Run evaluation in a background thread so the user gets
            # the answer immediately without waiting for the judge model.
            def _bg_eval():
                try:
                    ev_result = ev.evaluate(
                        question         = question,
                        answer           = result["answer"],
                        context_chunks   = result["context_chunks"],
                        sources          = result["sources"],
                        reference_answer = reference_answer,
                        session_id       = result["session_id"],
                        latency_ms       = result["latency_ms"],
                    )
                    # Patch eval_id back into the session store so the
                    # feedback endpoint can find it later.  This is a
                    # best-effort in-memory mapping; replace with a DB
                    # in production.
                    _last_eval_by_session[result["session_id"]] = ev_result.eval_id
                except Exception as exc:
                    app.logger.warning(f"Background eval failed: {exc}")

            threading.Thread(target=_bg_eval, daemon=True).start()

            # The eval_id won't be known until the background thread
            # finishes, but we record the session mapping above.
            # For synchronous eval_id return, set AUTO_EVAL=False and
            # call /api/evaluate directly.

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Lightweight in-memory map: session_id → last eval_id
_last_eval_by_session: dict = {}


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


# ── Evaluation routes ─────────────────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Run a synchronous evaluation for a single Q&A pair.

    Request body:
    {
      "question":         "What is Newton's second law?",
      "answer":           "F = ma ...",
      "context_chunks":   ["chunk text 1", "chunk text 2"],   // optional
      "sources":          ["lecture3.pdf"],                    // optional
      "reference_answer": "Force equals mass times ...",       // optional
      "session_id":       "abc-123",                          // optional
      "latency_ms":       412.3                               // optional
    }

    Returns the full EvalResult as JSON.
    """
    data             = request.get_json(force=True)
    question         = (data.get("question") or "").strip()
    answer           = (data.get("answer")   or "").strip()
    context_chunks   = data.get("context_chunks", [])
    sources          = data.get("sources", [])
    reference_answer = data.get("reference_answer") or None
    session_id       = data.get("session_id") or None
    latency_ms       = data.get("latency_ms") or None

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' are required."}), 400

    try:
        result = ev.evaluate(
            question         = question,
            answer           = answer,
            context_chunks   = context_chunks,
            sources          = sources,
            reference_answer = reference_answer,
            session_id       = session_id,
            latency_ms       = latency_ms,
        )
        from dataclasses import asdict
        return jsonify(asdict(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/feedback", methods=["POST"])
def feedback():
    """
    Record thumbs-up / thumbs-down for a previous evaluation.

    Request body:
    {
      "eval_id":  "uuid-of-the-eval-result",
      "feedback": 1    // 1 = thumbs-up, -1 = thumbs-down
    }
    """
    data     = request.get_json(force=True)
    eval_id  = (data.get("eval_id") or "").strip()
    fb_value = data.get("feedback")

    if not eval_id:
        return jsonify({"error": "'eval_id' is required."}), 400
    if fb_value not in (1, -1):
        return jsonify({"error": "'feedback' must be 1 (up) or -1 (down)."}), 400

    updated = ev.add_feedback(eval_id, fb_value)
    if not updated:
        return jsonify({"error": f"eval_id '{eval_id}' not found in log."}), 404

    return jsonify({"message": "Feedback recorded.", "eval_id": eval_id, "feedback": fb_value})


@app.route("/api/evaluate/stats", methods=["GET"])
def eval_stats():
    """Return aggregate evaluation metrics across all logged evaluations."""
    return jsonify(ev.get_aggregate_stats())


@app.route("/api/evaluate/recent", methods=["GET"])
def eval_recent():
    """
    Return the most recent evaluation records.
    Query param: ?limit=20  (default 20)
    """
    limit = request.args.get("limit", 20, type=int)
    records = ev.get_recent_evals(limit=limit)
    return jsonify({"count": len(records), "results": records})


@app.route("/api/evaluate/run-dataset", methods=["POST"])
def run_dataset():
    """
    Run the full eval dataset (eval_dataset.json) through the RAG pipeline.
    This is a long-running operation — consider calling it offline or
    in a background job for large datasets.

    Returns a summary report.
    """
    try:
        summary = ev.run_eval_dataset(rag_query_fn=rag.query)
        # Strip the per-result context to keep the HTTP response small
        summary_lean = {k: v for k, v in summary.items() if k != "results"}
        summary_lean["n_results"] = len(summary.get("results", []))
        return jsonify(summary_lean)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)