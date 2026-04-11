from __future__ import annotations

import asyncio
from dataclasses import dataclass


class OtpTimeoutError(Exception):
    pass


@dataclass
class OtpChallenge:
    chat_id: int
    task_id: str
    question: str


class OtpBroker:
    def __init__(self) -> None:
        self._pending: dict[int, tuple[str, asyncio.Future[str]]] = {}
        self._queue: asyncio.Queue[OtpChallenge] = asyncio.Queue()

    async def request_otp(self, chat_id: int, task_id: str, question: str, timeout_seconds: int = 180) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending[chat_id] = (task_id, future)
        await self._queue.put(OtpChallenge(chat_id=chat_id, task_id=task_id, question=question))
        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError as exc:
            self._pending.pop(chat_id, None)
            raise OtpTimeoutError(f"OTP timed out for task {task_id}") from exc

    def submit_otp(self, chat_id: int, otp: str) -> tuple[bool, str]:
        pending = self._pending.get(chat_id)
        if pending is None:
            return False, "No pending OTP challenge for this chat."

        task_id, future = pending
        if future.done():
            self._pending.pop(chat_id, None)
            return False, "OTP challenge already resolved."

        future.set_result(otp)
        self._pending.pop(chat_id, None)
        return True, f"OTP received for task {task_id}."

    async def next_challenge(self) -> OtpChallenge:
        return await self._queue.get()
