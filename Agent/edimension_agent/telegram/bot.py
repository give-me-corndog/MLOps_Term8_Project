from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from pathlib import Path

import boto3
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Document, FSInputFile, Message, Update

from ..agent_service import BrowserTaskRunner
from ..config import Settings
from ..crypto import CredentialCipher
from ..db import (
    TASK_STATUS_RUNNING,
    USER_STATUS_READY,
    USER_STATUS_WAITING_AUTH_METHOD,
    USER_STATUS_WAITING_PASSWORD,
    USER_STATUS_WAITING_USERNAME,
    Database,
)
from ..otp_broker import OtpBroker
from .. import rag_service

logger = logging.getLogger(__name__)

# Per-chat RAG session IDs (in-memory; survives only for the process lifetime).
# Maps chat_id → session_id string.
_rag_sessions: dict[int, str] = {}

# Per-chat asyncio locks — ensures only one RAG query runs at a time per user,
# preventing race conditions on session history and parallel Ollama calls.
_rag_locks: dict[int, asyncio.Lock] = {}


def _get_rag_lock(chat_id: int) -> asyncio.Lock:
    """Return (creating if needed) the per-user RAG query lock."""
    if chat_id not in _rag_locks:
        _rag_locks[chat_id] = asyncio.Lock()
    return _rag_locks[chat_id]


class TelegramAgentBot:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        cipher: CredentialCipher,
        task_runner: BrowserTaskRunner,
        otp_broker: OtpBroker,
    ) -> None:
        self.settings = settings
        self.db = db
        self.cipher = cipher
        self.task_runner = task_runner
        self.otp_broker = otp_broker

        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)

        self._otp_listener_task: asyncio.Task[None] | None = None
        self._register_handlers()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.bot.set_webhook(
            url=self.settings.telegram_webhook_url,
            secret_token=self.settings.telegram_webhook_secret,
            allowed_updates=["message"],
        )
        self._otp_listener_task = asyncio.create_task(self._otp_listener_loop())

    async def stop(self) -> None:
        if self._otp_listener_task is not None:
            self._otp_listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._otp_listener_task
        await self.bot.delete_webhook(drop_pending_updates=False)
        await self.bot.session.close()

    async def handle_update(self, update: dict) -> None:
        parsed = Update.model_validate(update)
        if parsed.message is not None:
            self.db.ensure_user(parsed.message.chat.id)
        await self.dp.feed_update(self.bot, parsed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_authorized(self, chat_id: int) -> bool:
        return self.db.ensure_user(chat_id).authorized

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list[str]:
        """
        Split text into chunks that fit within Telegram's character limit.
        Tries to break on newlines first, then on spaces, then hard-cuts as
        a last resort so no chunk ever exceeds `limit` characters.
        """
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            # Prefer a newline boundary within the limit.
            cut = text.rfind("\n", 0, limit)
            if cut <= 0:
                # Fall back to the last space within the limit.
                cut = text.rfind(" ", 0, limit)
            if cut <= 0:
                # Hard cut — no whitespace found.
                cut = limit
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        return chunks

    async def _send(self, chat_id: int, text: str) -> None:
        """Send a message, splitting it across multiple messages if needed."""
        for chunk in self._split_message(text):
            await self.bot.send_message(chat_id=chat_id, text=chunk)

    async def _reply(self, message: Message, text: str) -> None:
        """Reply to a Message object, splitting if needed."""
        for chunk in self._split_message(text):
            await message.answer(chunk)

    def _spaces_client(self):
        return boto3.client(
            "s3",
            region_name=self.settings.do_spaces_region,
            endpoint_url=self.settings.do_spaces_endpoint,
            aws_access_key_id=self.settings.do_spaces_key,
            aws_secret_access_key=self.settings.do_spaces_secret,
        )

    # ── Handler registration ──────────────────────────────────────────────────

    def _register_handlers(self) -> None:
        @self.router.message(Command("start"))
        async def _(m: Message) -> None:
            await self._handle_start(m)

        @self.router.message(Command("setup"))
        async def _(m: Message) -> None:
            await self._handle_setup(m)

        @self.router.message(Command("otp"))
        async def _(m: Message) -> None:
            await self._handle_otp(m)

        @self.router.message(Command("status"))
        async def _(m: Message) -> None:
            await self._handle_status(m)

        @self.router.message(Command("help"))
        async def _(m: Message) -> None:
            await self._handle_help(m)

        # ── RAG commands ──────────────────────────────────────────────────────

        @self.router.message(Command("get"))
        async def _(m: Message) -> None:
            await self._handle_get(m)

        @self.router.message(Command("ingest"))
        async def _(m: Message) -> None:
            await self._handle_ingest(m)

        @self.router.message(Command("docs"))
        async def _(m: Message) -> None:
            await self._handle_docs(m)

        @self.router.message(Command("clearchat"))
        async def _(m: Message) -> None:
            await self._handle_clearchat(m)

        @self.router.message(Command("cleardocs"))
        async def _(m: Message) -> None:
            await self._handle_cleardocs(m)

        # ── Catch-all text / document handler ────────────────────────────────

        @self.router.message()
        async def _(m: Message) -> None:
            await self._handle_text(m)

    # ── Existing handlers (unchanged) ─────────────────────────────────────────

    async def _handle_start(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer(
                "Your user is registered but not authorized yet. "
                "Ask an admin to set users.authorized = 1 in the database."
            )
            return
        await message.answer(
            "Welcome! Use /setup to configure your eDimension credentials.\n\n"
            "Send any text to query your documents, or use /get <query> "
            "to run a browser task on eDimension.\n"
            "Use /help to see all commands."
        )

    async def _handle_setup(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet. Ask an admin to enable your account in DB.")
            return
        self.db.set_user_status(chat_id, USER_STATUS_WAITING_USERNAME)
        await message.answer("Setup step 1/3: send your eDimension username.")

    async def _handle_otp(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet.")
            return
        payload = (message.text or "").strip().split(maxsplit=1)
        if len(payload) < 2:
            await message.answer("Usage: /otp 123456")
            return
        ok, detail = self.otp_broker.submit_otp(chat_id, payload[1].strip())
        await message.answer("OTP received. Resuming browser task." if ok else detail)

    async def _handle_status(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet.")
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 2:
            task = self.db.get_task(parts[1].strip())
            if task is None or task["chat_id"] != chat_id:
                await message.answer("Task not found.")
                return
            await self._reply(message,
                f"Task {parts[1].strip()}: {task['status']}\n"
                f"Query: {task['query']}\n"
                f"Error: {task.get('error_message') or '-'}"
            )
            return
        tasks = self.db.list_recent_tasks(chat_id)
        if not tasks:
            await message.answer("No tasks yet.")
            return
        lines = ["Recent tasks:"]
        for item in tasks:
            lines.append(f"- {item['task_id']}: {item['status']} | {item['query']}")
        await self._reply(message, "\n".join(lines))

    async def _handle_help(self, message: Message) -> None:
        await message.answer(
            "Commands:\n"
            "/start        – initialize bot access\n"
            "/setup        – store your eDimension credentials securely\n"
            "/otp <code>   – reply to OTP request during a browser task\n"
            "/status [id]  – inspect browser task status\n"
            "\n"
            "RAG / Document assistant:\n"
            "/ingest           – pull & index your files from eDimension / Spaces\n"
            "/docs             – list documents currently in your knowledge base\n"
            "/clearchat        – clear your conversation history (keeps documents)\n"
            "/cleardocs        – wipe all documents from your knowledge base\n"
            "\n"
            "You can also send a PDF directly to Telegram to ingest it.\n"
            "Plain text → answered by the document assistant.\n"
            "/get <query>      – run a browser task on eDimension."
        )

    # ── RAG handlers ──────────────────────────────────────────────────────────

    async def _handle_get(self, message: Message) -> None:
        """Trigger a browser task on eDimension."""
        chat_id = message.chat.id
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet.")
            return

        parts = (message.text or "").strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await message.answer("Usage: /get <your eDimension query>")
            return

        user = self.db.get_user(chat_id)
        if user is None or not user.username_encrypted or not user.password_encrypted or not user.auth_method:
            await message.answer("Credentials not set up yet. Run /setup first.")
            return

        query   = parts[1].strip()
        task_id = self.db.create_task(chat_id, query)
        await message.answer(f"Task queued: {task_id}. I will update you when done.")
        asyncio.create_task(self._run_task(task_id=task_id, chat_id=chat_id, query=query))

    async def _handle_ingest(self, message: Message) -> None:
        """
        Pull PDFs from DO Spaces for this user and ingest them.

        The Spaces prefix used is  <DO_SPACES_PREFIX>/<username>/
        which is exactly where the browser agent uploads files.
        """
        chat_id = message.chat.id
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet.")
            return

        if not self.settings.do_spaces_bucket:
            await message.answer("DigitalOcean Spaces is not configured on this server.")
            return

        user = self.db.get_user(chat_id)
        if not user or not user.username_encrypted:
            await message.answer("Credentials not set up yet. Run /setup first.")
            return

        username = self.cipher.decrypt(user.username_encrypted)
        prefix   = f"{self.settings.do_spaces_prefix}/{username}/"

        await message.answer(f"Fetching PDFs from Spaces (prefix: `{prefix}`)… ⏳")

        try:
            results = await rag_service.ingest_spaces_async(
                chat_id       = chat_id,
                spaces_client = self._spaces_client(),
                bucket        = self.settings.do_spaces_bucket,
                prefix        = prefix,
            )
        except Exception as exc:
            logger.exception("Spaces ingest failed for chat %d", chat_id)
            await message.answer(f"Ingest failed: {exc}")
            return

        if not results:
            await message.answer(
                f"No PDFs found under `{prefix}`. "
                "Run a browser task first to download course files."
            )
            return

        lines = ["✅ Ingestion complete:"]
        for fname, count in results.items():
            status = f"{count} chunks" if count >= 0 else "❌ failed"
            lines.append(f"  • {fname} → {status}")
        await self._reply(message, "\n".join(lines))

    async def _handle_docs(self, message: Message) -> None:
        """List documents in the user's knowledge base."""
        chat_id = message.chat.id
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet.")
            return

        files = rag_service.list_ingested_files(chat_id)
        if not files:
            await message.answer(
                "Your knowledge base is empty. Use /ingest or send a PDF."
            )
            return

        lines = [f"📚 {len(files)} document(s) in your knowledge base:"]
        for f in files:
            lines.append(f"  • {f}")
        await self._reply(message, "\n".join(lines))

    async def _handle_clearchat(self, message: Message) -> None:
        """Clear conversation history for this user (keeps documents)."""
        chat_id    = message.chat.id
        session_id = _rag_sessions.pop(chat_id, None)
        if session_id:
            rag_service.clear_session(session_id)
        await message.answer("Conversation history cleared. Your documents are still indexed.")

    async def _handle_cleardocs(self, message: Message) -> None:
        """Wipe all ingested documents for this user."""
        chat_id    = message.chat.id
        session_id = _rag_sessions.pop(chat_id, None)
        if session_id:
            rag_service.clear_session(session_id)
        rag_service.clear_user_collection(chat_id)
        await message.answer("✅ Your knowledge base has been cleared.")

    # ── Text / document catch-all ─────────────────────────────────────────────

    async def _handle_text(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet. Ask an admin to enable your account in DB.")
            return

        # If the user sent a PDF document, ingest it directly.
        if message.document and message.document.file_name.lower().endswith(".pdf"):
            await self._ingest_uploaded_pdf(message)
            return

        user = self.db.ensure_user(chat_id)
        text = (message.text or "").strip()
        if not text:
            await message.answer("Please send a text query or a PDF file.")
            return

        # ── Setup flow ────────────────────────────────────────────────────────
        if user.status == USER_STATUS_WAITING_USERNAME:
            self.db.save_username(chat_id, self.cipher.encrypt(text))
            self.db.set_user_status(chat_id, USER_STATUS_WAITING_PASSWORD)
            await message.answer("Setup step 2/3: send your eDimension password.")
            return

        if user.status == USER_STATUS_WAITING_PASSWORD:
            self.db.save_password(chat_id, self.cipher.encrypt(text))
            self.db.set_user_status(chat_id, USER_STATUS_WAITING_AUTH_METHOD)
            await message.answer(
                "Setup step 3/3: choose auth method by replying exactly with 'Okta' or 'Google Auth'."
            )
            return

        if user.status == USER_STATUS_WAITING_AUTH_METHOD:
            normalized = text.strip().lower()
            if normalized == "okta":
                selected = "Okta"
            elif normalized in {"google auth", "google", "googleauth"}:
                selected = "Google Auth"
            else:
                await message.answer("Invalid option. Please reply with exactly 'Okta' or 'Google Auth'.")
                return
            self.db.save_auth_method(chat_id, selected)
            self.db.set_user_status(chat_id, USER_STATUS_READY)
            await message.answer(
                f"Setup complete. Auth method saved as {selected}. "
                "You can now send questions directly, or use /get <query> for browser tasks."
            )
            return

        if user.status != USER_STATUS_READY:
            await message.answer("Please run /setup first.")
            return

        # ── RAG query (plain text → document assistant) ───────────────────────
        if rag_service.collection_count(chat_id) == 0:
            await message.answer(
                "Your knowledge base is empty. "
                "Use /ingest to pull your course files from Spaces, "
                "or send a PDF directly to this chat.\n\n"
                "To run a browser task on eDimension, use /get <query>."
            )
            return

        lock = _get_rag_lock(chat_id)
        if lock.locked():
            await message.answer(
                "⏳ Still thinking about your previous question — please wait."
            )
            return

        async with lock:
            await message.answer("Thinking… ⏳")
            session_id = _rag_sessions.get(chat_id)

            try:
                result = await rag_service.query(chat_id, text, session_id=session_id)
            except Exception as exc:
                logger.exception("RAG query failed for chat %d", chat_id)
                await message.answer(f"Something went wrong: {exc}")
                return

            _rag_sessions[chat_id] = result["session_id"]

            answer  = result["answer"]
            sources = result.get("sources", [])
            mode    = result.get("mode", "retrieve")

            if result.get("blocked"):
                await self._reply(message, answer)
                return

            footer = ""
            if sources:
                footer = "\n\n📄 Sources: " + ", ".join(sources)
            if mode == "summarize":
                footer += "\n_(summary mode)_"

            await self._reply(message, answer + footer)

    async def _ingest_uploaded_pdf(self, message: Message) -> None:
        """Download a PDF sent directly to the chat and ingest it."""
        chat_id  = message.chat.id
        doc: Document = message.document
        fname    = doc.file_name or "upload.pdf"

        # Telegram's getFile API enforces a 20 MB limit.
        TELEGRAM_MAX_BYTES = 20 * 1024 * 1024
        if doc.file_size and doc.file_size > TELEGRAM_MAX_BYTES:
            size_mb = doc.file_size / (1024 * 1024)
            await message.answer(
                f"❌ '{fname}' is {size_mb:.1f} MB — Telegram only allows bots to "
                f"download files up to 20 MB.\n\n"
                f"To ingest this file, either:\n"
                f"  • Compress or split the PDF and resend it, or\n"
                f"  • Upload it to your eDimension course and use /ingest to pull it from Spaces."
            )
            return

        await message.answer(f"Received '{fname}'. Ingesting… ⏳")

        try:
            file_info = await self.bot.get_file(doc.file_id)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            await self.bot.download_file(file_info.file_path, destination=tmp_path)

            # Rename so ChromaDB stores the original filename as source.
            named_path = Path(tmp_path).parent / fname
            Path(tmp_path).rename(named_path)

            count = await rag_service.ingest_pdf_async(chat_id, str(named_path))
            await message.answer(
                f"✅ '{fname}' ingested — {count} chunks added to your knowledge base.\n"
                "Just send a message to ask questions about it."
            )
        except Exception as exc:
            logger.exception("PDF ingest failed for chat %d", chat_id)
            await message.answer(f"Failed to ingest PDF: {exc}")
        finally:
            with contextlib.suppress(OSError):
                named_path.unlink(missing_ok=True)

    # ── Browser task runner (unchanged) ──────────────────────────────────────

    async def _run_task(self, task_id: str, chat_id: int, query: str) -> None:
        self.db.set_task_status(task_id, TASK_STATUS_RUNNING)
        user = self.db.get_user(chat_id)
        if user is None or not user.username_encrypted or not user.password_encrypted or not user.auth_method:
            self.db.fail_task(task_id, "User credentials are missing. Run /setup again.")
            await self._send(chat_id, f"Task failed ({task_id}): credentials missing.")
            return

        username    = self.cipher.decrypt(user.username_encrypted)
        password    = self.cipher.decrypt(user.password_encrypted)
        auth_method = user.auth_method

        try:
            result = await self.task_runner.run_task(
                task_id     = task_id,
                chat_id     = chat_id,
                query       = query,
                username    = username,
                password    = password,
                auth_method = auth_method
            )
        except Exception as exc:
            logger.exception("Task %s failed", task_id)
            self.db.fail_task(task_id, str(exc))
            await self._send(chat_id, f"Task failed ({task_id}): {exc}")
            return

        payload = {"summary": result.summary, "uploaded_files": result.uploaded_files, "logs": result.logs}
        self.db.complete_task(task_id, payload)

        if result.uploaded_files:
            await self._send(chat_id, f"Task completed: {task_id}. Sending file(s)…")
            for uploaded in result.uploaded_files:
                local_path = uploaded.get("local_path")
                if local_path:
                    try:
                        await self.bot.send_document(chat_id, FSInputFile(path=local_path))
                    except Exception as exc:
                        logger.warning("Failed to send file for task %s: %s", task_id, exc)
                    if self.settings.cleanup_after_upload:
                        with contextlib.suppress(OSError):
                            Path(local_path).unlink(missing_ok=True)
            return

        summary = result.summary.strip() if result.summary else "Task completed with no textual output."
        await self._send(chat_id, f"Task completed: {task_id}\n\n{summary}")

    async def _otp_listener_loop(self) -> None:
        while True:
            challenge = await self.otp_broker.next_challenge()
            await self._send(
                challenge.chat_id,
                (
                    f"Task {challenge.task_id} requires OTP.\n"
                    f"Prompt: {challenge.question}\n"
                    "Reply with /otp <code>"
                ),
            )