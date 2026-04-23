"""
Laminar observability integration for real-time eval quality tracking.

Provides utilities to:
- Push evaluation results to Laminar dashboard
- Fire alerts when metrics fall below thresholds
- Safe fallback if Laminar is unavailable
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from lmnr import Laminar
except ImportError:
    Laminar = None  # type: ignore[assignment]

# Configurable thresholds for alerts
FAITHFULNESS_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.5
LATENCY_THRESHOLD_MS = 5000
CONTEXT_RECALL_THRESHOLD = 0.6
ANSWER_RELEVANCY_THRESHOLD = 0.6

# Alert configuration
ALERTS_ENABLED = True


class LaminarMetricsHandler:
    """
    Safely push metrics to Laminar and handle alerts.
    Fails gracefully if Laminar is unavailable.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and Laminar is not None

    def push_eval_result(self, eval_result: Dict[str, Any]) -> bool:
        """
        Push evaluation result to Laminar dashboard.
        Returns True if successful, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            # Convert EvalResult dataclass to dict if needed
            if hasattr(eval_result, "__dataclass_fields__"):
                result_dict = asdict(eval_result)
            else:
                result_dict = eval_result

            # Create Laminar event with metrics
            event_data = {
                "eval_id": result_dict.get("eval_id"),
                "question": result_dict.get("question"),
                "answer": result_dict.get("answer"),
                "sources": result_dict.get("sources", []),
                "metrics": {
                    "faithfulness": result_dict.get("faithfulness"),
                    "answer_relevancy": result_dict.get("answer_relevancy"),
                    "context_precision": result_dict.get("context_precision"),
                    "context_recall": result_dict.get("context_recall"),
                    "latency_ms": result_dict.get("latency_ms"),
                    "token_count": result_dict.get("token_count"),
                },
                "session_id": result_dict.get("session_id"),
                "timestamp": result_dict.get("timestamp"),
            }

            # Send to Laminar
            Laminar.log_event("eval_result", event_data)
            logger.debug(f"Pushed eval result {result_dict.get('eval_id')} to Laminar")
            return True

        except Exception as exc:
            logger.warning(f"Failed to push eval result to Laminar: {exc}")
            return False

    def check_thresholds(self, eval_result: Dict[str, Any]) -> list[str]:
        """
        Check if metrics violate thresholds and return list of violations.
        Example: ["faithfulness < 0.7", "latency_ms > 5000"]
        """
        if not ALERTS_ENABLED:
            return []

        violations = []

        if hasattr(eval_result, "__dataclass_fields__"):
            result_dict = asdict(eval_result)
        else:
            result_dict = eval_result

        # Check faithfulness
        faithfulness = result_dict.get("faithfulness")
        if faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD:
            violations.append(
                f"Faithfulness too low: {faithfulness:.3f} < {FAITHFULNESS_THRESHOLD}"
            )

        # Check context precision
        context_precision = result_dict.get("context_precision")
        if (
            context_precision is not None
            and context_precision < CONTEXT_PRECISION_THRESHOLD
        ):
            violations.append(
                f"Context precision too low: {context_precision:.3f} < {CONTEXT_PRECISION_THRESHOLD}"
            )

        # Check latency
        latency_ms = result_dict.get("latency_ms")
        if latency_ms is not None and latency_ms > LATENCY_THRESHOLD_MS:
            violations.append(f"Latency too high: {latency_ms:.1f}ms > {LATENCY_THRESHOLD_MS}ms")

        # Check context recall
        context_recall = result_dict.get("context_recall")
        if context_recall is not None and context_recall < CONTEXT_RECALL_THRESHOLD:
            violations.append(
                f"Context recall too low: {context_recall:.3f} < {CONTEXT_RECALL_THRESHOLD}"
            )

        # Check answer relevancy
        answer_relevancy = result_dict.get("answer_relevancy")
        if answer_relevancy is not None and answer_relevancy < ANSWER_RELEVANCY_THRESHOLD:
            violations.append(
                f"Answer relevancy too low: {answer_relevancy:.3f} < {ANSWER_RELEVANCY_THRESHOLD}"
            )

        return violations

    def fire_alert(
        self,
        eval_id: str,
        violations: list[str],
        question: str,
        chat_id: Optional[int] = None,
    ) -> bool:
        """
        Fire alert for threshold violations to Laminar.
        Returns True if alert sent, False otherwise.
        """
        if not self.enabled or not violations:
            return False

        try:
            alert_data = {
                "eval_id": eval_id,
                "severity": "warning" if len(violations) == 1 else "critical",
                "violations": violations,
                "question": question,
                "chat_id": chat_id,
                "message": f"Quality degradation detected: {'; '.join(violations)}",
            }

            Laminar.log_event("quality_alert", alert_data)
            logger.warning(f"Quality alert fired for {eval_id}: {'; '.join(violations)}")
            return True

        except Exception as exc:
            logger.warning(f"Failed to fire alert to Laminar: {exc}")
            return False


# Global handler instance
_handler: Optional[LaminarMetricsHandler] = None


def initialize(enabled: bool = True) -> None:
    """Initialize the Laminar metrics handler."""
    global _handler
    _handler = LaminarMetricsHandler(enabled=enabled)
    logger.info(
        f"Laminar metrics handler initialized (enabled={_handler.enabled})"
    )


def get_handler() -> LaminarMetricsHandler:
    """Get or create the global Laminar metrics handler."""
    global _handler
    if _handler is None:
        _handler = LaminarMetricsHandler(enabled=Laminar is not None)
    return _handler


def push_eval_result(eval_result: Dict[str, Any]) -> bool:
    """Convenience function to push eval result."""
    return get_handler().push_eval_result(eval_result)


def check_and_alert(
    eval_result: Dict[str, Any],
    question: str,
    chat_id: Optional[int] = None,
) -> list[str]:
    """
    Check thresholds and fire alerts if needed.
    Returns list of violations.
    """
    handler = get_handler()
    violations = handler.check_thresholds(eval_result)
    if violations:
        eval_id = (
            asdict(eval_result).get("eval_id")
            if hasattr(eval_result, "__dataclass_fields__")
            else eval_result.get("eval_id")
        )
        handler.fire_alert(eval_id, violations, question, chat_id)
    return violations
