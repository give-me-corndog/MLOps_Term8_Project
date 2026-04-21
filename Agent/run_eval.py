#!/usr/bin/env python3
"""
run_eval.py — Offline Evaluation Runner for Telegram RAG
=========================================================

Usage:
    # Run the eval dataset against the Telegram RAG pipeline
    python run_eval.py --chat-id 123456789

    # Evaluate a single question
    python run_eval.py --chat-id 123456789 --question "What is Newton's law?" \
                       --reference "Force equals mass times acceleration."

    # Print aggregate stats from the log
    python run_eval.py --stats

    # Show the last N evaluations
    python run_eval.py --recent 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from dotenv import load_dotenv

from edimension_agent import eval as ev
from edimension_agent import rag_service


def _run_query(chat_id: int, question: str) -> dict:
    return rag_service.query_sync(chat_id=chat_id, question=question)


def cmd_run_dataset(args) -> None:
    print(f"Running eval dataset from '{ev.EVAL_DATASET}' …\n")
    try:
        summary = ev.run_eval_dataset(lambda q: _run_query(args.chat_id, q))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n── Aggregate Results ────────────────────────────────")
    for k, v in summary.items():
        if k != "results":
            print(f"  {k:<28} {v}")
    print("─────────────────────────────────────────────────────")


def cmd_single(args) -> None:
    if not args.question:
        print("ERROR: --question is required for single evaluation.")
        sys.exit(1)

    print(f"Evaluating single Q&A …\n  Q: {args.question}\n")
    rag_result = _run_query(args.chat_id, args.question)

    result = ev.evaluate(
        question=args.question,
        answer=rag_result["answer"],
        context_chunks=rag_result.get("context_chunks", []),
        sources=rag_result.get("sources", []),
        reference_answer=args.reference or None,
        session_id=rag_result.get("session_id"),
        latency_ms=rag_result.get("latency_ms"),
    )

    d = asdict(result)
    print("── Scores ───────────────────────────────────────────")
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = d.get(metric)
        display = f"{val:.3f}" if val is not None else "N/A"
        print(f"  {metric:<28} {display}")
    print(f"\nEval ID: {result.eval_id}")
    print(f"Logged to: {ev.EVAL_LOG}")


def cmd_stats(args) -> None:
    stats = ev.get_aggregate_stats()
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        sys.exit(1)
    print("\n── Aggregate Evaluation Stats ───────────────────────")
    print(json.dumps(stats, indent=2))


def cmd_recent(args) -> None:
    records = ev.get_recent_evals(limit=args.recent)
    if not records:
        print("No evaluation records found.")
        return
    print(f"\n── Last {len(records)} Evaluations ──────────────────────────")
    for r in records:
        import datetime
        ts = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  [{ts}] {r['eval_id'][:8]}…")
        print(f"  Q: {r['question'][:70]}")
        print(f"  Faithfulness:      {r.get('faithfulness')}")
        print(f"  Answer Relevancy:  {r.get('answer_relevancy')}")
        print(f"  Context Precision: {r.get('context_precision')}")
        print(f"  Context Recall:    {r.get('context_recall')}")
        if r.get("feedback") is not None:
            icon = "👍" if r["feedback"] > 0 else "👎"
            print(f"  Feedback: {icon}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Telegram RAG Evaluation CLI")
    parser.add_argument("--chat-id", type=int, help="Telegram chat_id for the user collection")
    parser.add_argument("--question", help="Question for single eval")
    parser.add_argument("--reference", help="Ground-truth answer for context-recall scoring")
    parser.add_argument("--stats", action="store_true", help="Print aggregate stats")
    parser.add_argument("--recent", type=int, metavar="N", help="Show last N evals")
    args = parser.parse_args()

    needs_chat = not (args.stats or args.recent)
    if needs_chat and not args.chat_id:
        print("ERROR: --chat-id is required for dataset or question evaluation.")
        sys.exit(1)

    if args.stats:
        cmd_stats(args)
    elif args.recent:
        cmd_recent(args)
    elif args.question:
        cmd_single(args)
    else:
        cmd_run_dataset(args)


if __name__ == "__main__":
    main()
