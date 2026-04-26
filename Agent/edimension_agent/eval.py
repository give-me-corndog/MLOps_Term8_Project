"""
eval.py — RAG Evaluation Module for Telegram Agent
==================================================

Implements an LLM-as-judge evaluation framework with four metrics:

  1. Faithfulness       — Is the answer grounded in the retrieved context?
  2. Answer Relevancy   — Does the answer address the question?
  3. Context Precision  — Are the retrieved chunks relevant to the question?
  4. Context Recall     — Do retrieved chunks cover the answer? (needs reference)

All judgments are made by the local Ollama model.
Results are stored in eval_results.jsonl (one JSON object per line).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
<<<<<<< HEAD
<<<<<<< HEAD
import logging
=======
>>>>>>> e4133e6 (RAG Evaluation set)
=======
import logging
>>>>>>> main
from dataclasses import dataclass, asdict
from typing import Optional

import ollama as _ollama
<<<<<<< HEAD
<<<<<<< HEAD
from . import lmnr_integration

logger = logging.getLogger(__name__)
=======
>>>>>>> e4133e6 (RAG Evaluation set)
=======
from . import lmnr_integration

logger = logging.getLogger(__name__)
>>>>>>> main

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "ministral-3")
EVAL_LOG = os.environ.get("EVAL_LOG", "eval_results.jsonl")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "eval_dataset.json")
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
PUSH_TO_LAMINAR = os.environ.get("PUSH_TO_LAMINAR", "true").lower() == "true"

_ollama_client = _ollama.Client(host=OLLAMA_HOST)
_total_tokens = 0  # Track total tokens across evaluations
<<<<<<< HEAD
=======

_ollama_client = _ollama.Client(host=OLLAMA_HOST)
>>>>>>> e4133e6 (RAG Evaluation set)
=======
>>>>>>> main

# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    eval_id: str
    timestamp: float
    question: str
    answer: str
    context: str
    sources: list[str]

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    reference_answer: Optional[str] = None
    session_id: Optional[str] = None
    latency_ms: Optional[float] = None
    feedback: Optional[int] = None
<<<<<<< HEAD
<<<<<<< HEAD
    token_count: Optional[int] = None  # Total tokens used for this evaluation
    cost_usd: Optional[float] = None  # Estimated cost in USD
=======
>>>>>>> e4133e6 (RAG Evaluation set)
=======
    token_count: Optional[int] = None  # Total tokens used for this evaluation
    cost_usd: Optional[float] = None  # Estimated cost in USD
>>>>>>> main


# ─────────────────────────────────────────────────────────────
# LLM JUDGE HELPERS
# ─────────────────────────────────────────────────────────────


def _judge(prompt: str) -> str:
    resp = _ollama_client.generate(model=JUDGE_MODEL, prompt=prompt)
    return resp["response"].strip()


def _extract_score(text: str) -> Optional[float]:
    text = text.lower()

    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        return round(float(m.group(1)) / float(m.group(2)), 3)

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return round(float(m.group(1)) / 100.0, 3)

    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if m:
        return round(float(m.group(1)), 3)

    m = re.search(r"\b([1-9]|10)\b", text)
    if m:
        return round(float(m.group(1)) / 10.0, 3)

    return None


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────


def score_faithfulness(question: str, context: str, answer: str) -> float:
    if not context.strip():
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


def score_context_recall(context: str, reference_answer: str, question: str) -> float:
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
# TOKEN COUNTING & COST TRACKING
# ─────────────────────────────────────────────────────────────


def _count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 characters for English).
    More accurate if Ollama response includes token counts.
    """
    return max(1, len(text) // 4)


_eval_token_tracker: dict[str, int] = {}


def _track_token_count(eval_id: str, tokens: int) -> None:
    """Track token count for an evaluation."""
    global _total_tokens
    _eval_token_tracker[eval_id] = _eval_token_tracker.get(eval_id, 0) + tokens
    _total_tokens += tokens


def _get_eval_token_count(eval_id: str) -> int:
    """Get total tokens for an evaluation."""
    return _eval_token_tracker.get(eval_id, 0)


# ─────────────────────────────────────────────────────────────
<<<<<<< HEAD
=======
>>>>>>> e4133e6 (RAG Evaluation set)
=======
>>>>>>> main
# MAIN EVALUATION ENTRY POINT
# ─────────────────────────────────────────────────────────────


def evaluate(
    question: str,
    answer: str,
    context_chunks: list[str],
    sources: list[str],
    reference_answer: Optional[str] = None,
    session_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> EvalResult:
    context = "\n\n".join(context_chunks)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
    eval_id = str(uuid.uuid4())
    
    # Track token usage for this evaluation
    _track_token_count(eval_id, _count_tokens_estimate(question + answer + context))

    result = EvalResult(
        eval_id=eval_id,
<<<<<<< HEAD
=======

    result = EvalResult(
        eval_id=str(uuid.uuid4()),
>>>>>>> e4133e6 (RAG Evaluation set)
=======
>>>>>>> main
        timestamp=time.time(),
        question=question,
        answer=answer,
        context=context,
        sources=sources,
        reference_answer=reference_answer,
        session_id=session_id,
        latency_ms=latency_ms,
    )

    result.faithfulness = score_faithfulness(question, context, answer)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
    _track_token_count(eval_id, _count_tokens_estimate(context + answer))

    result.answer_relevancy = score_answer_relevancy(question, answer)
    _track_token_count(eval_id, _count_tokens_estimate(question + answer))

    result.context_precision = score_context_precision(question, context_chunks)
    _track_token_count(eval_id, _count_tokens_estimate(question) * len(context_chunks))

    if reference_answer:
        result.context_recall = score_context_recall(context, reference_answer, question)
        _track_token_count(eval_id, _count_tokens_estimate(context + reference_answer))

    # Set final token count and cost estimate (placeholder for now)
    result.token_count = _get_eval_token_count(eval_id)
    result.cost_usd = None  # Would be calculated based on model pricing
<<<<<<< HEAD
=======
    result.answer_relevancy = score_answer_relevancy(question, answer)
    result.context_precision = score_context_precision(question, context_chunks)

    if reference_answer:
        result.context_recall = score_context_recall(context, reference_answer, question)
>>>>>>> e4133e6 (RAG Evaluation set)
=======
>>>>>>> main

    _log_result(result)
    return result


def add_feedback(eval_id: str, feedback: int) -> bool:
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
    if not os.path.exists(EVAL_DATASET):
        raise FileNotFoundError(
            f"Eval dataset not found at '{EVAL_DATASET}'. "
            "Create a JSON file with a list of {question, reference_answer} objects."
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

        answer = rag_result["answer"]
        sources = rag_result.get("sources", [])
        chunks = rag_result.get("context_chunks", [])

        ev = evaluate(
            question=question,
            answer=answer,
            context_chunks=chunks,
            sources=sources,
            reference_answer=reference,
            latency_ms=latency_ms,
        )
        results.append(asdict(ev))

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n_samples": len(results),
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        "avg_latency_ms": avg("latency_ms"),
        "results": results,
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
    thumbs_up = sum(1 for v in feedback_vals if v > 0)
    thumbs_down = sum(1 for v in feedback_vals if v < 0)

    return {
        "n_evaluations": len(records),
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        "avg_latency_ms": avg("latency_ms"),
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "feedback_count": len(feedback_vals),
    }


def get_recent_evals(limit: int = 20) -> list[dict]:
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


def _log_result(result: EvalResult) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
    """
    Log evaluation result to JSONL file and optionally push to Laminar.
    """
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")

    # Push to Laminar if enabled
    if PUSH_TO_LAMINAR:
        try:
            lmnr_integration.push_eval_result(asdict(result))
            violations = lmnr_integration.check_and_alert(
                asdict(result),
                question=result.question,
                chat_id=None,  # Would need to pass chat_id from context
            )
            if violations:
                logger.warning(f"Quality degradation for eval {result.eval_id}: {violations}")
        except Exception as exc:
            logger.warning(f"Failed to push eval result to Laminar: {exc}")
<<<<<<< HEAD
=======
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
>>>>>>> e4133e6 (RAG Evaluation set)
=======
>>>>>>> main
