from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import time
import logging

import boto3
from browser_use import ActionResult, Agent, BrowserSession, ChatGoogle, ChatOllama, Tools

from .config import Settings
from .db import TASK_STATUS_WAITING_OTP, Database
from .otp_broker import OtpBroker, OtpTimeoutError


logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    summary: str
    uploaded_files: list[dict]


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
        provider = self._settings.browser_llm_provider.strip().lower()
        if provider in {"google", "gemini", "googlechatmodel", "chatgoogle"}:
            return ChatGoogle(
                model=self._settings.google_model,
                temperature=self._settings.google_temperature,
            )
        if provider in {"ollama", "qwen", "qwen3.5"}:
            return ChatOllama(
                model=self._settings.browser_ollama_model,
                host=self._settings.browser_ollama_host,
            )

        raise ValueError(
            "Unsupported BROWSER_LLM_PROVIDER. Use 'google' or 'ollama'."
        )

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

    async def run_task(
        self,
        task_id: str,
        chat_id: int,
        query: str,
        username: str,
        password: str,
        auth_method: str,
    ) -> AgentRunResult:
        self._settings.downloads_dir.mkdir(parents=True, exist_ok=True)

        tools = Tools()
        uploaded_files: list[dict] = []

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
                You are helping a user on eDimension. Follow this workflow:
                1. Go to https://edimension.sutd.edu.sg/. Press "OK" if you see a "Privacy, cookies and terms of use" popup to expose the login page.
                2. Click on the "SUTD EASE ID" to access the EASE login page.
                3. On reaching EASE login page, log in using the placeholder credentials: use `username` for the username/email field and `password` for the password field. Click Login after entering credentials.
                4. After logging, depending on the user's  {auth_method}, select the correct auth method to trigger the OTP input.
                5. When you see a text input for OTP, call the tool "Ask human for the OTP" to get the OTP required. Recall this tool if OTP fails, and strictly use the same auth method: {auth_method}.
                6. Navigate to courses page at ```https://edimension.sutd.edu.sg/ultra/course``` and find by either scrolling through all the course options or searching for the full name of the course. (E.g user might say MLOps but means Machine Learning Operations)
                7. Click the course page and when it loads, click on the Content tab to be redirected to the course-specific contents that contains directories like Assignments, Labs etc
                8. According to the user's query, click on the most appropriate directory to look for the relevant information or files. 
                9. If the query requires file download(s), download and call the tool 'Upload the newest downloaded PDF to DigitalOcean Spaces' after each downloaded PDF.
                10. If no file is needed, summarize findings clearly as required by the user within 5 sentences.
                11. After successful completion, call 'Clean browser-use temporary download folders and stale staged uploads'.

                User query:
                {query}
                """.strip()

        llm = self._build_browser_llm()
        browser_session = BrowserSession()

        agent = Agent(
            allowed_domains=[
            "edimension.sutd.edu.sg",
            "ease.sutd.edu.sg",      # SUTD EASE SSO domain
            ],
            task=task_prompt,
            llm=llm,
            tools=tools,
            sensitive_data={"username": username, "password": password},
            browser_session=browser_session,
            use_vision = True if self._settings.browser_llm_provider.strip().lower() in {"google", "gemini", "googlechatmodel", "chatgoogle"} else False,
            calculate_cost=True,
        )

        start = time.perf_counter()
        result = await agent.run()
        latency = time.perf_counter() - start

        ## Build summary string
        usage_summary = await agent.token_cost_service.get_usage_summary()
        cost_line = f"Cost: ${usage_summary.total_cost:.6f} | Latency: {latency:.2f}s"
        summary = f"{summary}\n{cost_line}" if summary else cost_line
        summary = summary[:500]

        if hasattr(result, "final_result") and callable(result.final_result):
            final = result.final_result()
            if isinstance(final, str):
                summary += final.strip()

        # Fallback to the most recent extracted content if final_result is empty.
        if not summary and hasattr(result, "extracted_content") and callable(result.extracted_content):
            extracted = result.extracted_content()
            if isinstance(extracted, list) and extracted:
                last_item = extracted[-1]
                if isinstance(last_item, str):
                    summary = last_item.strip()

        # Keep Telegram responses concise.
        summary = summary[:500]

        return AgentRunResult(summary=summary, uploaded_files=uploaded_files)
