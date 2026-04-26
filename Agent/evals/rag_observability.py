from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from lmnr import Laminar
except ImportError:
    Laminar = None  # type: ignore[assignment]

try:
    from lmnr import observe
except ImportError:
    observe = None  # type: ignore[assignment]


# Guard against repeated initialize() calls from hot paths.
_laminar_initialized = False


def observe_query(name: Optional[str] = None):
    """Return Laminar observe decorator if available; otherwise return identity decorator."""
    if observe is None:
        def _identity(fn):
            return fn
        return _identity
    return observe(name=name) if name else observe()


def _parse_bool(raw: str, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_laminar_initialized() -> bool:
    """Lazily initialize Laminar so scripts can emit telemetry without app startup hooks."""
    global _laminar_initialized

    if _laminar_initialized:
        return True
    if Laminar is None:
        return False

    lmnr_enabled = _parse_bool(os.getenv("LMNR_ENABLED", "false"), False)
    api_key = os.getenv("LMNR_PROJECT_API_KEY", "").strip()
    if not lmnr_enabled or not api_key:
        return False

    try:
        lmnr_self_hosted = _parse_bool(os.getenv("LMNR_SELF_HOSTED", "true"), True)
        if lmnr_self_hosted:
            http_port = int(os.getenv("LMNR_HTTP_PORT", "8000"))
            grpc_port = int(os.getenv("LMNR_GRPC_PORT", "8001"))
            Laminar.initialize(
                project_api_key=api_key,
                base_url="http://localhost",
                http_port=http_port,
                grpc_port=grpc_port,
            )
        else:
            Laminar.initialize(project_api_key=api_key)

        _laminar_initialized = True
        logger.info("RAG observability connected to Laminar")
        return True
    except Exception as exc:
        logger.warning("RAG observability failed to initialize Laminar: %s", exc)
        return False


def begin_query_trace(
    chat_id: int,
    session_id: str,
    question: str,
    mode: str = "live",
) -> None:
    if not _ensure_laminar_initialized() or Laminar is None:
        return

    try:
        Laminar.set_trace_session_id(session_id)
        Laminar.set_trace_metadata(
            {
                "component": "rag",
                "mode": mode,
                "chat_id": chat_id,
                "question_preview": question[:160],
            }
        )
    except Exception as exc:
        logger.debug("Laminar begin_query_trace skipped: %s", exc)


def log_query_result(
    *,
    chat_id: int,
    session_id: str,
    question: str,
    answer: str,
    blocked: bool,
    block_reason: Optional[str],
    latency_ms: float,
    mode: str,
    sources: list[str],
    context_chunks_count: int,
    faithfulness: Optional[float] = None,
    context_precision: Optional[float] = None,
    trace_status: Optional[str] = None,
) -> None:
    if not _ensure_laminar_initialized() or Laminar is None:
        return

    try:
        Laminar.set_trace_session_id(session_id)
        # For blocked prompts, set terminal trace status immediately and return.
        if blocked:
            Laminar.set_trace_user_id("rag:live:blocked")
        elif trace_status in {"success", "failed"}:
            Laminar.set_trace_user_id(f"rag:live:{trace_status}")

        Laminar.log_event(
            "rag_query_result",
            {
                "chat_id": chat_id,
                "session_id": session_id,
                "question": question,
                "answer_preview": answer[:500],
                "blocked": blocked,
                "block_reason": block_reason,
                "latency_ms": latency_ms,
                "mode": mode,
                "sources": sources,
                "context_chunks_count": context_chunks_count,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "trace_status": trace_status,
            },
        )

        if blocked:
            return
    except Exception as exc:
        logger.debug("Laminar log_query_result skipped: %s", exc)


def log_quality_metrics(
    *,
    chat_id: int,
    session_id: str,
    mode: str,
    question: str,
    sources: list[str],
    latency_ms: float,
    faithfulness: float,
    context_precision: float,
    answer_quality: float,
) -> None:
    if not _ensure_laminar_initialized() or Laminar is None:
        return

    try:
        Laminar.set_trace_session_id(session_id)
        # Final trace status: failed only when BOTH scores are below 0.5.
        is_failed = faithfulness < 0.5 and context_precision < 0.5
        Laminar.set_trace_user_id("rag:live:failed" if is_failed else "rag:live:success")

        Laminar.log_event(
            "rag_quality_metrics",
            {
                "chat_id": chat_id,
                "session_id": session_id,
                "mode": mode,
                "question": question,
                "sources": sources,
                "latency_ms": latency_ms,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "answer_quality": answer_quality,
                "trace_status": "failed" if is_failed else "success",
            },
        )

        # Lightweight alert event for real-time dashboard filtering.
        violations = []
        if faithfulness < 0.7:
            violations.append(f"faithfulness<{0.7}")
        if context_precision < 0.5:
            violations.append(f"context_precision<{0.5}")
        if answer_quality < 0.6:
            violations.append(f"answer_quality<{0.6}")
        if latency_ms > 5000:
            violations.append(f"latency_ms>{5000}")

        if violations:
            Laminar.log_event(
                "rag_quality_alert",
                {
                    "chat_id": chat_id,
                    "question": question,
                    "mode": mode,
                    "violations": violations,
                },
            )
    except Exception as exc:
        logger.debug("Laminar log_quality_metrics skipped: %s", exc)
