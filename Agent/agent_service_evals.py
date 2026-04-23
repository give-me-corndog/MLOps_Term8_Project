from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from browser_use import BrowserSession

try:
    from lmnr import Laminar, evaluate
except ImportError:
    Laminar = None  # type: ignore[assignment]
    evaluate = None  # type: ignore[assignment]

from edimension_agent.agent_service import BrowserTaskRunner
from edimension_agent.config import load_settings
from edimension_agent.db import Database
from edimension_agent.otp_broker import OtpBroker

# TO RUN: uv run python agent_service_evals.py --username "<EDIM_USERNAME>" --password "<EDIM_PASSWORD>" --auth-method okta --group-name "agent_service_success_failure"

logger = logging.getLogger(__name__)

EVAL_QUERIES = [
    # "Assignment 1 due date for MLOps",
    # "Assignment 1 submission criteria for MLOps",
    "Log into the course page",
    "MLOps Assignment 1 score",
    "STEP course Dissertation Due Dates",
    "List MLOps lab topics",
    "List MLOps lecture topics",
    # "Download MLOps Course Handout",
    # "Download MLOps Project Handout which is accessed via a link under the MLOps Course Syllabus document",
    "Download MLOps Week 1 Lectures Notes",
]
DEFAULT_OUTPUT = Path("agent_service_eval_results.jsonl")
EVAL_CHAT_ID = "eval:agent"



@dataclass
class EvalRecord:
    timestamp: str
    task_id: str
    query: str
    status: str
    error: str | None
    summary: str | None
    uploaded_file_count: int
    logs: dict | None


def _initialize_laminar(project_api_key: str) -> None:
    if Laminar is None:
        logger.warning("Laminar SDK is not installed; Laminar evaluation export disabled")
        return
    if not project_api_key:
        logger.warning("LMNR_PROJECT_API_KEY missing; Laminar evaluation export disabled")
        return
    try:
        Laminar.initialize(
            project_api_key=project_api_key,
            base_url="http://localhost",
            http_port=8000,
            grpc_port=8001,
            )
    except Exception as exc:
        logger.warning("Laminar initialization failed; evaluation export disabled: %s", exc)


async def _send_evals_to_laminar(records: list[EvalRecord], project_api_key: str, group_name: str) -> None:
    if evaluate is None:
        return
    if not project_api_key:
        return

    dataset = [
        {
            "data": {
                "task_id": record.task_id,
                "query": record.query,
                "status": record.status,
                "error": record.error,
                "summary": record.summary,
                "uploaded_file_count": record.uploaded_file_count,
                "logs": record.logs or {},
            },
            "target": {"expect_success": 1},
        }
        for record in records
    ]

    if not dataset:
        return

    result = evaluate(
        data=dataset,
        executor=lambda data: {
            "status": data.get("status", "failed"),
            "logs": data.get("logs", {}),
            "error": data.get("error"),
        },
        evaluators={
            "success": lambda output, target: int(output.get("status") == "success"),
            "failure": lambda output, target: int(output.get("status") != "success"),
        },
        project_api_key=project_api_key,
        group_name=group_name
    )
    if hasattr(result, "__await__"):
        await result


async def _otp_listener(otp_broker: OtpBroker) -> None:
    while True:
        challenge = await otp_broker.next_challenge()
        prompt = (
            f"\n[OTP REQUIRED] task_id={challenge.task_id}\n"
            f"Prompt: {challenge.question}\n"
            "Enter OTP code (or leave blank to skip): "
        )
        otp = await asyncio.to_thread(input, prompt)
        otp = otp.strip()
        if not otp:
            logger.warning("No OTP entered for task %s", challenge.task_id)
            continue
        ok, detail = otp_broker.submit_otp(challenge.chat_id, otp)
        if ok:
            logger.info("OTP submitted for task %s", challenge.task_id)
        else:
            logger.warning("OTP submit failed for task %s: %s", challenge.task_id, detail)


def _append_record(path: Path, record: EvalRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


async def run_eval(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = load_settings()
    _initialize_laminar(settings.lmnr_project_api_key)

    db = Database(settings.db_path)
    db.init()
    otp_broker = OtpBroker()
    runner = BrowserTaskRunner(settings=settings, db=db, otp_broker=otp_broker)
    shared_browser_session = BrowserSession(keep_alive = True)

    queries: list[str]
    if args.query:
        queries = args.query
    else:
        queries = EVAL_QUERIES

    if not queries:
        logger.error("No queries found to evaluate.")
        return 1

    output_path = Path(args.output)
    otp_task = asyncio.create_task(_otp_listener(otp_broker))

    total = 0
    succeeded = 0
    failed = 0
    records_for_laminar: list[EvalRecord] = []

    try:
        for query in queries:
            total += 1
            task_id = uuid4().hex
            logger.info("[%d/%d] Running task_id=%s query=%s", total, len(queries), task_id, query)

            status = "success"
            error: str | None = None
            summary: str | None = None
            uploaded_file_count = 0
            logs: dict = {}

            try:
                result = await runner.run_task(
                    task_id=task_id,
                    chat_id=EVAL_CHAT_ID,
                    query=query,
                    username=args.username,
                    password=args.password,
                    auth_method=args.auth_method,
                    browser_session=shared_browser_session,
                    close_browser_session=False,
                    bypass_guardrails=True,
                    live=False # False because doing evaluations
                )
                summary = (result.summary or "").strip()
                uploaded_file_count = len(result.uploaded_files)
                logs = result.logs or {}
                if not bool(logs.get("is_done")) or logs.get("is_successful") is not True:
                    status = "failed"
                    error = logs.get("final_result") or "Agent history indicates unsuccessful completion"
            except Exception as exc:
                status = "failed"
                error = str(exc)
                logger.exception("Task failed task_id=%s", task_id)

            if status == "success":
                succeeded += 1
            else:
                failed += 1

            record = EvalRecord(
                timestamp=datetime.now(UTC).isoformat(),
                task_id=task_id,
                query=query,
                status=status,
                error=error,
                summary=summary,
                uploaded_file_count=uploaded_file_count,
                logs=logs
            )
            records_for_laminar.append(record)
            _append_record(output_path, record)

        await _send_evals_to_laminar(
            records=records_for_laminar,
            project_api_key=settings.lmnr_project_api_key,
            group_name=args.group_name,
        )
    finally:
        otp_task.cancel()
        try:
            await otp_task
        except asyncio.CancelledError:
            pass
        await shared_browser_session.kill()

    logger.info("Eval run complete total=%d succeeded=%d failed=%d output=%s", total, succeeded, failed, output_path)
    return 0 if failed == 0 else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run browser-agent service across multiple queries and log success/failure."
    )
    parser.add_argument("--username", required=True, help="eDimension username")
    parser.add_argument("--password", required=True, help="eDimension password")
    parser.add_argument(
        "--auth-method",
        default="okta",
        choices=["okta", "google_auth", "google-auth", "google"],
        help="MFA method expected by the workflow",
    )
    parser.add_argument("--chat-id", type=int, default=999999, help="Synthetic chat_id for OTP broker")
    parser.add_argument("--query", action="append", help="Single query to run (can be repeated)")
    parser.add_argument(
        "--bypass-guardrails",
        action="store_true",
        help="Bypass browser-agent guardrails for eval runs only.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSONL output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--group-name",
        default="agent_service_success_failure",
        help="Laminar evaluate group name for this eval run",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    # Normalize auth method variants to existing agent prompt values.
    mapping = {
        "google-auth": "google_auth",
        "google": "google_auth",
    }
    args.auth_method = mapping.get(args.auth_method, args.auth_method)

    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
