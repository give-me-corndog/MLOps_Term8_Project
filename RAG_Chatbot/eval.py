"""
eval.py — RAG Evaluation Module for Teaching Assistant Chatbot
==============================================================

Implements an LLM-as-judge evaluation framework with four metrics:

  1. Faithfulness       — Is the answer grounded in the retrieved context?
  2. Answer Relevancy   — Does the answer address the question?
  3. Context Precision  — Are the retrieved chunks relevant to the question?
  4. Context Recall     — Do retrieved chunks cover the answer? (needs reference)

All judgments are made by the local Ollama model, so no external API is needed.
Results are stored in eval_results.jsonl (one JSON object per line).
"""

import json
import re
import time
import uuid
import os
from dataclasses import dataclass, asdict
from typing import Optional

import ollama

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "ministral-3")   # model used for scoring
EVAL_LOG    = os.environ.get("EVAL_LOG",    "eval_results.jsonl")
EVAL_DATASET= os.environ.get("EVAL_DATASET","eval_dataset.json")

# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    eval_id:            str
    timestamp:          float
    question:           str
    answer:             str
    context:            str             # newline-joined retrieved chunks
    sources:            list[str]

    # Scores — all in [0.0, 1.0].  None means "not computed".
    faithfulness:       Optional[float] = None   # answer grounded in context?
    answer_relevancy:   Optional[float] = None   # answer addresses question?
    context_precision:  Optional[float] = None   # chunks are relevant to question?
    context_recall:     Optional[float] = None   # chunks cover reference answer?

    # Optional fields
    reference_answer:   Optional[str]   = None
    session_id:         Optional[str]   = None
    latency_ms:         Optional[float] = None
    feedback:           Optional[int]   = None   # +1 thumbs-up / -1 thumbs-down


# ─────────────────────────────────────────────────────────────
# LLM JUDGE HELPERS
# ─────────────────────────────────────────────────────────────

def _judge(prompt: str) -> str:
    """Call the judge model and return the raw response text."""
    resp = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
    return resp["response"].strip()


def _extract_score(text: str) -> Optional[float]:
    """
    Parse a numeric score from judge output.
    Accepts patterns like: 'Score: 0.8', '4/5', '4 out of 5', '80%', or a lone digit.
    Always normalises to [0.0, 1.0].
    """
    text = text.lower()

    # Fraction: 4/5
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        return round(float(m.group(1)) / float(m.group(2)), 3)

    # Percentage: 80%
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return round(float(m.group(1)) / 100.0, 3)

    # Decimal already in [0,1]: 0.85
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if m:
        return round(float(m.group(1)), 3)

    # Integer 1–10 scale
    m = re.search(r"\b([1-9]|10)\b", text)
    if m:
        return round(float(m.group(1)) / 10.0, 3)

    return None


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────

def score_faithfulness(question: str, context: str, answer: str) -> float:
    """
    Faithfulness: does every factual claim in the answer appear in the context?
    Score 1.0 means fully grounded; 0.0 means entirely hallucinated.
    """
    if not context.strip():
        # No context was retrieved — we cannot assess grounding.
        return 0.0

    prompt = f"""You are an impartial evaluator assessing whether an AI answer is faithful to a given context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Task: Identify every factual claim in the ANSWER. For each claim, decide whether it is supported by the CONTEXT.

Rules:
- A claim is faithful if it is directly stated or clearly implied by the CONTEXT.
- General world knowledge not found in the CONTEXT counts as unfaithful.
- Ignore stylistic content (greetings, hedges like "I think").

Respond with:
1. A brief bullet list of claims and whether each is Faithful / Unfaithful.
2. A final score from 0.0 to 1.0 on the last line, formatted exactly as: Score: X.XX

If the answer has no factual claims, output Score: 1.0"""

    raw = _judge(prompt)
    score = _extract_score(raw)
    return score if score is not None else 0.5


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: does the answer directly address the question?
    Score 1.0 means perfectly on-topic; 0.0 means totally off-topic.
    """
    prompt = f"""You are an impartial evaluator assessing whether an AI answer is relevant to the question asked.

QUESTION:
{question}

ANSWER:
{answer}

Task: Judge how well the ANSWER addresses the QUESTION.

Scoring guide:
- 1.0: Directly and completely answers the question.
- 0.7–0.9: Mostly answers the question with minor gaps.
- 0.4–0.6: Partially answers; misses key parts.
- 0.1–0.3: Barely related to the question.
- 0.0: Completely off-topic or refuses without reason.

Respond with a single sentence of reasoning, then on the last line: Score: X.XX"""

    raw = _judge(prompt)
    score = _extract_score(raw)
    return score if score is not None else 0.5


def score_context_precision(question: str, context_chunks: list[str]) -> float:
    """
    Context Precision: what fraction of retrieved chunks are actually relevant
    to the question?  A chunk is relevant if it would help answer the question.
    """
    if not context_chunks:
        return 0.0

    relevant = 0
    for chunk in context_chunks:
        prompt = f"""You are an impartial evaluator. Decide if the following text chunk is relevant to answering the question.

QUESTION:
{question}

CHUNK:
{chunk}

Is this chunk relevant? Answer with a single word: YES or NO, then one sentence of reasoning."""

        raw = _judge(prompt).lower()
        if "yes" in raw:
            relevant += 1

    return round(relevant / len(context_chunks), 3)


def score_context_recall(
    context: str,
    reference_answer: str,
    question: str
) -> float:
    """
    Context Recall: does the retrieved context contain the information needed
    to produce the reference answer?  Requires a known ground-truth answer.
    """
    if not context.strip() or not reference_answer.strip():
        return 0.0

    prompt = f"""You are an impartial evaluator. Determine whether the CONTEXT provides enough information to arrive at the REFERENCE ANSWER for the given QUESTION.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CONTEXT:
{context}

Task: Decompose the REFERENCE ANSWER into key factual claims. For each claim, check whether the CONTEXT supports it.

Respond with:
1. Bullet list: each claim → Supported / Not Supported
2. Final line: Score: X.XX  (fraction of claims that are Supported)"""

    raw = _judge(prompt)
    score = _extract_score(raw)
    return score if score is not None else 0.5


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION ENTRY POINT
# ─────────────────────────────────────────────────────────────

def evaluate(
    question:         str,
    answer:           str,
    context_chunks:   list[str],
    sources:          list[str],
    reference_answer: Optional[str] = None,
    session_id:       Optional[str] = None,
    latency_ms:       Optional[float] = None,
) -> EvalResult:
    """
    Run all applicable metrics and return a populated EvalResult.
    Persists results to EVAL_LOG (JSONL).
    """
    context = "\n\n".join(context_chunks)

    result = EvalResult(
        eval_id          = str(uuid.uuid4()),
        timestamp        = time.time(),
        question         = question,
        answer           = answer,
        context          = context,
        sources          = sources,
        reference_answer = reference_answer,
        session_id       = session_id,
        latency_ms       = latency_ms,
    )

    # Always computed
    result.faithfulness      = score_faithfulness(question, context, answer)
    result.answer_relevancy  = score_answer_relevancy(question, answer)
    result.context_precision = score_context_precision(question, context_chunks)

    # Only when ground-truth is available
    if reference_answer:
        result.context_recall = score_context_recall(context, reference_answer, question)

    _log_result(result)
    return result


def add_feedback(eval_id: str, feedback: int) -> bool:
    """
    Patch feedback (+1 or -1) into an existing eval log entry.
    Rewrites the matching line in the JSONL file.
    Returns True if the entry was found and updated.
    """
    if not os.path.exists(EVAL_LOG):
        return False

    lines = []
    found = False
    with open(EVAL_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("eval_id") == eval_id:
                    obj["feedback"] = feedback
                    found = True
                lines.append(json.dumps(obj))
            except json.JSONDecodeError:
                lines.append(line)

    if found:
        with open(EVAL_LOG, "w") as f:
            f.write("\n".join(lines) + "\n")

    return found


# ─────────────────────────────────────────────────────────────
# BATCH EVALUATION (EVAL DATASET)
# ─────────────────────────────────────────────────────────────

def run_eval_dataset(rag_query_fn) -> list[dict]:
    """
    Load eval_dataset.json, run each sample through the RAG pipeline,
    evaluate it, and return a summary report.

    Expected format of eval_dataset.json:
    [
      {
        "question": "What is Newton's second law?",
        "reference_answer": "F = ma …"   // optional
      },
      ...
    ]
    """
    if not os.path.exists(EVAL_DATASET):
        raise FileNotFoundError(
            f"Eval dataset not found at '{EVAL_DATASET}'. "
            "Create a JSON file with a list of {{question, reference_answer}} objects."
        )

    with open(EVAL_DATASET) as f:
        samples = json.load(f)

    results = []
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        reference = sample.get("reference_answer")
        print(f"[{i}/{len(samples)}] Evaluating: {question[:60]}…")

        t0 = time.time()
        rag_result = rag_query_fn(question)
        latency_ms = (time.time() - t0) * 1000

        # rag_query_fn must return the same dict shape as rag.query()
        answer   = rag_result["answer"]
        sources  = rag_result.get("sources", [])
        # Re-retrieve chunks for scoring — stored in rag_result if patched
        chunks   = rag_result.get("context_chunks", [answer])

        ev = evaluate(
            question         = question,
            answer           = answer,
            context_chunks   = chunks,
            sources          = sources,
            reference_answer = reference,
            latency_ms       = latency_ms,
        )
        results.append(asdict(ev))

    # Aggregate summary
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n_samples":          len(results),
        "avg_faithfulness":   avg("faithfulness"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        "avg_latency_ms":     avg("latency_ms"),
        "results":            results,
    }

    report_path = f"eval_report_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved to {report_path}")

    return summary


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────

def get_aggregate_stats() -> dict:
    """Return aggregate metrics across all logged evaluations."""
    if not os.path.exists(EVAL_LOG):
        return {"error": "No evaluation log found."}

    records = []
    with open(EVAL_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {"error": "Evaluation log is empty."}

    def avg(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    feedback_vals = [r["feedback"] for r in records if r.get("feedback") is not None]
    thumbs_up    = sum(1 for v in feedback_vals if v > 0)
    thumbs_down  = sum(1 for v in feedback_vals if v < 0)

    return {
        "n_evaluations":        len(records),
        "avg_faithfulness":     avg("faithfulness"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_context_precision":avg("context_precision"),
        "avg_context_recall":   avg("context_recall"),
        "avg_latency_ms":       avg("latency_ms"),
        "thumbs_up":            thumbs_up,
        "thumbs_down":          thumbs_down,
        "feedback_count":       len(feedback_vals),
    }


def get_recent_evals(limit: int = 20) -> list[dict]:
    """Return the most recent N evaluation records."""
    if not os.path.exists(EVAL_LOG):
        return []

    records = []
    with open(EVAL_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return sorted(records, key=lambda r: r.get("timestamp", 0), reverse=True)[:limit]


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _log_result(result: EvalResult):
    """Append an EvalResult to the JSONL log."""
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
