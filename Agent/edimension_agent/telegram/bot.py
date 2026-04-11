from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message, Update

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

logger = logging.getLogger(__name__)


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

    def _is_authorized(self, chat_id: int) -> bool:
        user = self.db.ensure_user(chat_id)
        return user.authorized

    async def _send(self, chat_id: int, text: str) -> None:
        await self.bot.send_message(chat_id=chat_id, text=text)

    def _register_handlers(self) -> None:
        @self.router.message(Command("start"))
        async def start_handler(message: Message) -> None:
            await self._handle_start(message)

        @self.router.message(Command("setup"))
        async def setup_handler(message: Message) -> None:
            await self._handle_setup(message)

        @self.router.message(Command("otp"))
        async def otp_handler(message: Message) -> None:
            await self._handle_otp(message)

        @self.router.message(Command("status"))
        async def status_handler(message: Message) -> None:
            await self._handle_status(message)

        @self.router.message(Command("help"))
        async def help_handler(message: Message) -> None:
            await self._handle_help(message)

        @self.router.message()
        async def text_handler(message: Message) -> None:
            await self._handle_text(message)

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
            "Welcome. Please run /setup to configure your eDimension credentials before any other action."
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
            await message.answer("You are not authorized yet. Ask an admin to enable your account in DB.")
            return

        payload = (message.text or "").strip().split(maxsplit=1)
        if len(payload) < 2:
            await message.answer("Usage: /otp 123456")
            return

        otp = payload[1].strip()
        ok, detail = self.otp_broker.submit_otp(chat_id, otp)
        if ok:
            await message.answer("OTP received. Resuming browser task.")
        else:
            await message.answer(detail)

    async def _handle_status(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet. Ask an admin to enable your account in DB.")
            return

        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 2:
            task_id = parts[1].strip()
            task = self.db.get_task(task_id)
            if task is None or task["chat_id"] != chat_id:
                await message.answer("Task not found.")
                return
            await message.answer(
                f"Task {task_id}: {task['status']}\n"
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
        await message.answer("\n".join(lines))

    async def _handle_help(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        await message.answer(
            "Commands:\n"
            "/start - initialize bot access\n"
            "/setup - store your credentials securely\n"
            "/otp <code> - reply to OTP request\n"
            "/status [task_id] - inspect task status\n"
            "Setup step 3 includes selecting auth method: Okta or Google Auth.\n"
            "After setup, send any eDimension navigation query as plain text."
        )

    async def _handle_text(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.ensure_user(chat_id)
        if not self._is_authorized(chat_id):
            await message.answer("You are not authorized yet. Ask an admin to enable your account in DB.")
            return

        user = self.db.ensure_user(chat_id)
        text = (message.text or "").strip()
        if not text:
            await message.answer("Please send a text query.")
            return

        if user.status == USER_STATUS_WAITING_USERNAME:
            self.db.save_username(chat_id, self.cipher.encrypt(text))
            self.db.set_user_status(chat_id, USER_STATUS_WAITING_PASSWORD)
            await message.answer("Setup step 2/3: send your eDimension password.")
            return

        if user.status == USER_STATUS_WAITING_PASSWORD:
            self.db.save_password(chat_id, self.cipher.encrypt(text))
            self.db.set_user_status(chat_id, USER_STATUS_WAITING_AUTH_METHOD)
            await message.answer("Setup step 3/3: choose auth method by replying exactly with 'Okta' or 'Google Auth'.")
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
            await message.answer(f"Setup complete. Auth method saved as {selected}. You can now send your query.")
            return

        if user.status != USER_STATUS_READY:
            await message.answer("Please run /setup first.")
            return

        task_id = self.db.create_task(chat_id, text)
        await message.answer(f"Task queued: {task_id}. I will update you when done.")
        asyncio.create_task(self._run_task(task_id=task_id, chat_id=chat_id, query=text))

    async def _run_task(self, task_id: str, chat_id: int, query: str) -> None:
        self.db.set_task_status(task_id, TASK_STATUS_RUNNING)
        user = self.db.get_user(chat_id)
        if user is None or not user.username_encrypted or not user.password_encrypted or not user.auth_method:
            self.db.fail_task(task_id, "User credentials are missing. Run /setup again.")
            await self._send(chat_id, f"Task failed ({task_id}): credentials missing.")
            return

        username = self.cipher.decrypt(user.username_encrypted)
        password = self.cipher.decrypt(user.password_encrypted)
        auth_method = user.auth_method

        try:
            result = await self.task_runner.run_task(
                task_id=task_id,
                chat_id=chat_id,
                query=query,
                username=username,
                password=password,
                auth_method=auth_method,
            )
        except Exception as exc:
            logger.exception("Task %s failed", task_id)
            self.db.fail_task(task_id, str(exc))
            await self._send(chat_id, f"Task failed ({task_id}): {exc}")
            return

        payload = {"summary": result.summary, "uploaded_files": result.uploaded_files}
        self.db.complete_task(task_id, payload)

        if result.uploaded_files:
            await self._send(chat_id, f"Task completed: {task_id}. Sending file(s)...")
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
