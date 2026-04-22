from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import time
import logging
from typing import Any
from urllib.parse import urlparse

import boto3
from browser_use import ActionResult, Agent, BrowserSession, ChatGoogle, ChatOllama, Tools, BrowserProfile
from langchain_core.messages import HumanMessage

try:
    from lmnr import Laminar, observe
except ImportError:
    Laminar = None  # type: ignore[assignment]

from .config import Settings
from .db import TASK_STATUS_WAITING_OTP, Database
from .otp_broker import OtpBroker, OtpTimeoutError


logger = logging.getLogger(__name__)


GUARDRAIL_REJECTION_MESSAGE = (
    "I can only perform browser tasks for a specific course. "
    "Include the course name or code, and use RAG for document-content questions."
)
GUARDRAIL_MODEL = "gemini-2.5-flash-lite"
BROWSER_LLM_MODEL = "gemini-3-flash-preview" #gemini-3-flash-preview/ministral-3/qwen3.5
BROWSER_PROFILE = BrowserProfile(
    minimum_wait_page_load_time=0.1,
	wait_between_actions=0.1,
    allowed_domains=['ease.sutd.edu.sg', 'edimension.sutd.edu.sg', 'docs.google.com'],
)
ALLOWED_DOMAINS = tuple(BROWSER_PROFILE.allowed_domains or [])
MAX_STEPS = 30


@dataclass
class AgentRunResult:
    summary: str
    uploaded_files: list[dict]
    logs: dict[str, Any]


class BrowserTaskRunner:
    def __init__(self, settings: Settings, db: Database, otp_broker: OtpBroker) -> None:
        self._settings = settings
        self._db = db
        self._otp_broker = otp_broker
        self._spaces = boto3.client(
            "s3",
            region_name=settings.do_spaces_region,
            endpoint_url=settings.do_spaces_endpoint,
            aws_access_key_id=settings.do_spaces_key,
            aws_secret_access_key=settings.do_spaces_secret,
        )

    def _build_browser_llm(self):
        model = BROWSER_LLM_MODEL.strip()
        if model.lower().startswith("gemini"):
            return ChatGoogle(
                model=model,
                temperature=self._settings.google_temperature,
            )
        return ChatOllama(model=model, host=self._settings.browser_ollama_host)

    async def _guardrail_allows_query(self, query: str, task_id: str, chat_id: int) -> tuple[bool, str]:
        # Logging
        Laminar.set_trace_user_id(str(chat_id))
        Laminar.set_trace_session_id(task_id)

        guardrail_prompt = f"""
        You are a strict request classifier for an LMS browser automation agent.

        ALLOW only if the query is limited to:
        - web navigation on the portal
        - downloading files
        - retrieving metadata visible on portal pages (titles, deadlines, listings) only

        REJECT if the query asks to read/summarize/extract/analyze content inside files
        (PDFs, slides, docs, manuals, rubrics, content), even if it also asks to download.

        REJECT if the query does not specify a target course name or course code.

        Output format (exactly one line):
        - ALLOW
        - REJECT[<short user-facing reason>]

        Query:
        {query}
        """.strip()

        try:
            guard_model = ChatGoogle(model=GUARDRAIL_MODEL, temperature=0.0)
            response = await guard_model.ainvoke([HumanMessage(content=guardrail_prompt)])
        except Exception as exc:
            logger.warning("Guardrail model check failed; defaulting to ALLOW: %s", exc)
            return True, ""

        raw = ""
        if isinstance(response, str):
            raw = response
        elif hasattr(response, "content"):
            content = getattr(response, "content")
            if isinstance(content, str):
                raw = content
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(str(item.get("text", "")))
                raw = " ".join(parts)
        if not raw:
            raw = str(response)

        reject_with_reason = re.search(r"REJECT\s*\[(.*?)\]", raw, flags=re.IGNORECASE | re.DOTALL)
        if reject_with_reason:
            reason = reject_with_reason.group(1).strip().strip("\"'")
            return False, reason or GUARDRAIL_REJECTION_MESSAGE

        normalized = raw.strip().upper()
        if re.search(r"\bREJECT\b", normalized):
            return False, GUARDRAIL_REJECTION_MESSAGE
        return True, ""

    @staticmethod
    def _serialize_log_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): BrowserTaskRunner._serialize_log_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [BrowserTaskRunner._serialize_log_value(v) for v in value]
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return BrowserTaskRunner._serialize_log_value(value.model_dump())
            except Exception:
                return str(value)
        if hasattr(value, "dict") and callable(value.dict):
            try:
                return BrowserTaskRunner._serialize_log_value(value.dict())
            except Exception:
                return str(value)
        if hasattr(value, "__dict__"):
            try:
                return BrowserTaskRunner._serialize_log_value(vars(value))
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _collect_agent_logs(result: Any) -> dict[str, Any]:
        def call_method(name: str) -> Any:
            method = getattr(result, name, None)
            if callable(method):
                try:
                    return method()
                except Exception as exc:
                    return f"error calling {name}: {exc}"
            return None

        return {
            "final_result": BrowserTaskRunner._serialize_log_value(call_method("final_result")),
            "is_done": BrowserTaskRunner._serialize_log_value(call_method("is_done")),
            "is_successful": BrowserTaskRunner._serialize_log_value(call_method("is_successful")),
            "has_errors": BrowserTaskRunner._serialize_log_value(call_method("has_errors")),
            # "model_thoughts": BrowserTaskRunner._serialize_log_value(call_method("model_thoughts")),
            # "action_results": BrowserTaskRunner._serialize_log_value(call_method("action_results")),
            "action_history": BrowserTaskRunner._serialize_log_value(call_method("action_history")),
            # "number_of_steps": BrowserTaskRunner._serialize_log_value(call_method("number_of_steps")),
            # "total_duration_seconds": BrowserTaskRunner._serialize_log_value(
            #     call_method("total_duration_seconds")
            # ),
        }

    @staticmethod
    def _is_allowed_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
        if not url:
            return False
        normalized_url = url.strip().lower()
        if normalized_url == "about:blank":
            # Browser sessions often start on about:blank before first navigation.
            return True

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)

    @staticmethod
    def sanitize_key_part(value: str) -> str:
        value = value.strip()
        value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
        return value.strip("-._") or "unknown"

    def build_spaces_key(self, student_id: str, task_id: str, filename: str) -> str:
        safe_student_id = self.sanitize_key_part(student_id)
        safe_filename = self.sanitize_key_part(Path(filename).name)
        safe_task = self.sanitize_key_part(task_id)
        return f"{self._settings.do_spaces_prefix}/{safe_student_id}/{safe_filename}"

    @staticmethod
    def newest_pdf_from_known_locations() -> Path | None:
        candidates: list[Path] = []

        temp_root = Path(tempfile.gettempdir())
        patterns = (
            "browser-use-downloads-*/*.pdf",
            "browser_use_agent_*/browseruse_agent_data/*.pdf",
            "browser_use_agent_*/**/browseruse_agent_data/*.pdf",
        )
        for pattern in patterns:
            candidates.extend(temp_root.glob(pattern))

        if not candidates:
            return None

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = str(candidate.resolve())
            except Exception:
                resolved = str(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_candidates.append(candidate)

        existing_candidates: list[Path] = []
        for candidate in unique_candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    existing_candidates.append(candidate)
            except OSError:
                continue

        if not existing_candidates:
            return None

        return max(existing_candidates, key=lambda p: p.stat().st_mtime)

    def cleanup_temp_and_staging(self, current_task_id: str) -> dict:
        cleaned = {
            "staged_pdf_files_removed": 0,
            "temp_pdf_files_removed": 0,
            "temp_dirs_removed": 0,
            "errors": [],
        }

        # Clean stale staged files but keep current task's files so Telegram can send them.
        try:
            if self._settings.downloads_dir.exists():
                for pdf in self._settings.downloads_dir.glob("*.pdf"):
                    if pdf.name.startswith(f"{current_task_id}-"):
                        continue
                    try:
                        pdf.unlink(missing_ok=True)
                        cleaned["staged_pdf_files_removed"] += 1
                    except Exception as exc:
                        cleaned["errors"].append(f"Failed to delete staged file {pdf}: {exc}")
        except Exception as exc:
            cleaned["errors"].append(
                f"Failed to clean staged artifacts directory {self._settings.downloads_dir}: {exc}"
            )

        temp_root = Path(tempfile.gettempdir())

        # Clean browser-use download directories and PDFs.
        for dl_dir in temp_root.glob("browser-use-downloads-*"):
            if not dl_dir.is_dir():
                continue
            for pdf in dl_dir.glob("*.pdf"):
                try:
                    pdf.unlink(missing_ok=True)
                    cleaned["temp_pdf_files_removed"] += 1
                except Exception as exc:
                    cleaned["errors"].append(f"Failed to delete temp download PDF {pdf}: {exc}")
            try:
                dl_dir.rmdir()
                cleaned["temp_dirs_removed"] += 1
            except OSError:
                pass
            except Exception as exc:
                cleaned["errors"].append(f"Failed to remove temp download dir {dl_dir}: {exc}")

        # Clean PDFs produced by save_as_pdf in browseruse_agent_data folders.
        for agent_data_dir in temp_root.glob("browser_use_agent_*/**/browseruse_agent_data"):
            if not agent_data_dir.is_dir():
                continue
            for pdf in agent_data_dir.glob("*.pdf"):
                try:
                    pdf.unlink(missing_ok=True)
                    cleaned["temp_pdf_files_removed"] += 1
                except Exception as exc:
                    cleaned["errors"].append(f"Failed to delete agent temp PDF {pdf}: {exc}")
            try:
                agent_data_dir.rmdir()
                cleaned["temp_dirs_removed"] += 1
            except OSError:
                pass
            except Exception as exc:
                cleaned["errors"].append(f"Failed to remove agent temp dir {agent_data_dir}: {exc}")

        return cleaned

    @observe()
    async def run_task(
        self,
        task_id: str,
        chat_id: int,
        query: str,
        username: str,
        password: str,
        auth_method: str,
        browser_profile: BrowserProfile = BROWSER_PROFILE,
        browser_session: BrowserSession | None = None,
        close_browser_session: bool = True,
        bypass_guardrails: bool = False,
        live: bool = True
    ) -> AgentRunResult:
        if bypass_guardrails:
            logger.warning("Guardrails bypassed for task_id=%s chat_id=%s", task_id, chat_id)
        else:
            allowed, rejection_message = await self._guardrail_allows_query(query, task_id, chat_id)
            logger.info(
                "Guardrail verdict task_id=%s chat_id=%s verdict=%s",
                task_id,
                chat_id,
                "ALLOW" if allowed else "REJECT",
            )
            if not allowed:
                logger.info(
                    "Guardrail rejection task_id=%s reason=%s",
                    task_id,
                    rejection_message,
                )
                return AgentRunResult(
                    summary=rejection_message,
                    uploaded_files=[],
                    logs={
                        "final_result": rejection_message,
                        "is_done": True,
                        "is_successful": False,
                        "has_errors": False,
                        "model_thoughts": [],
                        "action_results": [],
                        "action_history": [],
                        "number_of_steps": 0,
                        "total_duration_seconds": 0,
                        "guardrail_rejected": True,
                    },
                )

        
        # Logging
        Laminar.set_trace_user_id(str(chat_id))
        Laminar.set_trace_session_id(task_id)
        Laminar.set_trace_metadata({"live": live, "agent": True})

        self._settings.downloads_dir.mkdir(parents=True, exist_ok=True)

        tools = Tools()
        uploaded_files: list[dict] = []
        summary = "" # As response to Telegram bot
        offsite_violation: dict[str, Any] | None = None


        @tools.action(description="Ask human for the OTP")
        async def ask_human_for_otp(question: str) -> ActionResult:
            self._db.set_task_status(task_id, TASK_STATUS_WAITING_OTP)
            try:
                otp = await self._otp_broker.request_otp(chat_id, task_id, question)
            except OtpTimeoutError as exc:
                return ActionResult(error=str(exc))
            return ActionResult(
                extracted_content=f"The human responded with OTP: {otp}",
                include_in_memory=True,
            )

        @tools.action(description="Upload the newest downloaded PDF to DigitalOcean Spaces")
        async def upload_newest_pdf_to_spaces(params: dict | None = None) -> ActionResult:
            _ = params
            if not self._settings.do_spaces_bucket:
                return ActionResult(error="DO_SPACES_BUCKET is not configured")

            pdf_path = self.newest_pdf_from_known_locations()
            if pdf_path is None:
                return ActionResult(
                    error=(
                        f"No PDF found in browser-use temp directories under {tempfile.gettempdir()}"
                    )
                )

            staged_pdf_path = self._settings.downloads_dir / f"{task_id}-AF{pdf_path.name}"
            try:
                if pdf_path.resolve() != staged_pdf_path.resolve():
                    shutil.copy2(pdf_path, staged_pdf_path)
                    pdf_path = staged_pdf_path
            except Exception as exc:
                return ActionResult(error=f"Failed to stage PDF for upload: {exc}")

            object_key = self.build_spaces_key(username, task_id, pdf_path.name)
            try:
                self._spaces.upload_file(str(pdf_path), self._settings.do_spaces_bucket, object_key)
            except Exception as exc:
                return ActionResult(error=f"Spaces upload failed: {exc}")

            url = (
                f"https://{self._settings.do_spaces_bucket}."
                f"{self._settings.do_spaces_region}.digitaloceanspaces.com/{object_key}"
            )
            payload = {
                "uploaded": True,
                "local_path": str(pdf_path),
                "bucket": self._settings.do_spaces_bucket,
                "key": object_key,
                "url": url,
            }
            uploaded_files.append(payload)
            return ActionResult(extracted_content=json.dumps(payload), include_in_memory=True)

        @tools.action(description="Clean browser-use temporary download folders and stale staged uploads")
        async def cleanup_temp_browser_and_staging(params: dict | None = None) -> ActionResult:
            _ = params
            cleanup_result = self.cleanup_temp_and_staging(current_task_id=task_id)
            return ActionResult(extracted_content=json.dumps(cleanup_result), include_in_memory=True)

        

        task_prompt = f"""
                You are an agent helping a user navigate eDimension. 

                If the query is allowed, follow this workflow:
                1. FIRST ACTION (mandatory): navigate to https://edimension.sutd.edu.sg/ and wait for page load, even if already logged in or currently on another page.

                If you are not logged in:
                1. Press "OK" if you see a "Privacy, cookies and terms of use" popup to expose the login page.
                2. Click on the "SUTD EASE ID" to access the EASE login page.
                3. On reaching EASE login page, log in using the placeholder credentials: use `username` for the username/email field and `password` for the password field. Click Login after entering credentials.
                4. After logging, depending on the user's {auth_method}, select the correct auth method to trigger the OTP input.
                5. When you see a text input for OTP, call the tool "Ask human for the OTP" to get the OTP required. Recall this tool if OTP fails, and strictly use the same auth method: {auth_method}.

                At this point, if you are already logged in:
                1. Navigate to courses page at ```https://edimension.sutd.edu.sg/ultra/course``` and find by either scrolling through all the course options or searching for the full name of the course. (E.g user might say MLOps but means Machine Learning Operations)
                2. Click the course page and when it loads, click on the Content tab to be redirected to the course-specific contents that contains directories like Assignments, Labs etc
                3. According to the user's query, click on the most appropriate directory to look for the relevant information or files. It may not be named exactly such as Syllabus being called Course Handout.
                4. If the query requires file download(s), download and call the tool 'Upload the newest downloaded PDF to DigitalOcean Spaces' after each downloaded PDF.
                5. If the query asks for information visible directly on the eDimension portal UI (such as assignment deadlines, listing of lab topics, or listing of lecture topics), summarize those findings clearly. DO NOT open or read the actual files/PDFs to gather this information.
                6. After successful completion, call 'Clean browser-use temporary download folders and stale staged uploads'.

                User query:
                {query}
                """.strip()

        llm = self._build_browser_llm()
        if browser_session is None:
            browser_session = BrowserSession()
        agent = Agent(
            task=task_prompt,
            llm=llm,
            tools=tools,
            browser_profile=browser_profile,
            sensitive_data={"username": username, "password": password},
            browser_session=browser_session,
            use_vision = True if self._settings.browser_llm_provider.strip().lower() in {"google", "gemini", "googlechatmodel", "chatgoogle"} else False,
            calculate_cost=True,
        )

        # Defined callbacks per step
        async def fail_on_offsite(current_agent: Agent) -> None:
            nonlocal offsite_violation
            try:
                url = await current_agent.browser_session.get_current_page_url()
            except Exception as exc:
                logger.warning("Offsite check failed to read URL: %s", exc)
                return

            if self._is_allowed_url(url, ALLOWED_DOMAINS):
                return

            offsite_violation = {
                "allowed_domains": list(ALLOWED_DOMAINS),
                "offsite_url": url,
            }
            logger.error("Offsite navigation blocked task_id=%s url=%s", task_id, url)
            current_agent.stop()

        try:
            # start = time.perf_counter()
            result = await agent.run(on_step_start=fail_on_offsite, max_steps = MAX_STEPS)
            # latency = time.perf_counter() - start

            # ## Build summary string
            # usage_summary = await agent.token_cost_service.get_usage_summary()
            # cost_line = f"Cost: ${usage_summary.total_cost:.6f} | Latency: {latency:.2f}s"
            # summary = f"{cost_line}" if summary else cost_line
            # summary = summary[:500]

            if hasattr(result, "final_result") and callable(result.final_result):
                final = result.final_result()
                if isinstance(final, str):
                    summary += final.strip()

            # Fallback to the most recent extracted content if final_result is empty.
            

            # Keep Telegram responses concise.
            summary = summary[:500]

            logs = self._collect_agent_logs(result)
            if offsite_violation is not None:
                logs["offsite_violation"] = offsite_violation

            return AgentRunResult(
                summary=summary,
                uploaded_files=uploaded_files,
                logs=logs,
            )
        finally:
            try: 
                if close_browser_session:
                    await browser_session.kill()
            except Exception as exc:
                logger.warning("Failed to close browser session cleanly: %s", exc)
