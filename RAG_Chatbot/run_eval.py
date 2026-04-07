#!/usr/bin/env python3
"""
run_eval.py — Offline CLI Evaluation Runner
============================================

Usage:
    # Run the built-in eval dataset against the RAG pipeline
    python run_eval.py

    # Evaluate a single question/answer pair
    python run_eval.py --question "What is Newton's law?" \
                       --answer "F = ma" \
                       --reference "Force equals mass times acceleration."

    # Print aggregate stats from the log
    python run_eval.py --stats

    # Show the last N evaluations
    python run_eval.py --recent 10
"""

import argparse
import json
import sys
import rag
import eval as ev
from dataclasses import asdict


def cmd_run_dataset(args):
    print(f"Running eval dataset from '{ev.EVAL_DATASET}' …\n")
    try:
        summary = ev.run_eval_dataset(rag_query_fn=rag.query)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n── Aggregate Results ────────────────────────────────")
    for k, v in summary.items():
        if k != "results":
            print(f"  {k:<28} {v}")
    print("─────────────────────────────────────────────────────")


def cmd_single(args):
    if not args.question or not args.answer:
        print("ERROR: --question and --answer are required for single evaluation.")
        sys.exit(1)

    print(f"Evaluating single Q&A …\n  Q: {args.question}\n  A: {args.answer[:80]}…\n")
    result = ev.evaluate(
        question         = args.question,
        answer           = args.answer,
        context_chunks   = args.context.split("|||") if args.context else [],
        sources          = [],
        reference_answer = args.reference or None,
    )
    d = asdict(result)
    print("── Scores ───────────────────────────────────────────")
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = d.get(metric)
        display = f"{val:.3f}" if val is not None else "N/A"
        print(f"  {metric:<28} {display}")
    print(f"\nEval ID: {result.eval_id}")
    print(f"Logged to: {ev.EVAL_LOG}")


def cmd_stats(args):
    stats = ev.get_aggregate_stats()
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        sys.exit(1)
    print("\n── Aggregate Evaluation Stats ───────────────────────")
    print(json.dumps(stats, indent=2))


def cmd_recent(args):
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


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument("--question",  help="Question for single eval")
    parser.add_argument("--answer",    help="Answer for single eval")
    parser.add_argument("--reference", help="Ground-truth answer for context-recall scoring")
    parser.add_argument("--context",   help="Context chunks separated by '|||'")
    parser.add_argument("--stats",     action="store_true", help="Print aggregate stats")
    parser.add_argument("--recent",    type=int, metavar="N", help="Show last N evals")
    args = parser.parse_args()

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
