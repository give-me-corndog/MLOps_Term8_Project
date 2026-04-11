from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


USER_STATUS_NEW = "new"
USER_STATUS_WAITING_USERNAME = "waiting_username"
USER_STATUS_WAITING_PASSWORD = "waiting_password"
USER_STATUS_WAITING_AUTH_METHOD = "waiting_auth_method"
USER_STATUS_READY = "ready"

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_WAITING_OTP = "waiting_otp"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"


@dataclass
class UserRecord:
    chat_id: int
    status: str
    username_encrypted: str | None
    password_encrypted: str | None
    auth_method: str | None
    authorized: bool


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    chat_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    username_encrypted TEXT,
                    password_encrypted TEXT,
                    auth_method TEXT,
                    authorized INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES users(chat_id)
                )
                """
            )
            user_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "auth_method" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN auth_method TEXT")
            if "authorized" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN authorized INTEGER NOT NULL DEFAULT 0")

            conn.execute("DROP TABLE IF EXISTS inbound_messages")
            conn.commit()
        finally:
            conn.close()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def get_user(self, chat_id: int) -> UserRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT chat_id, status, username_encrypted, password_encrypted, auth_method, authorized
                FROM users
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
            if row is None:
                return None
            return UserRecord(
                chat_id=row["chat_id"],
                status=row["status"],
                username_encrypted=row["username_encrypted"],
                password_encrypted=row["password_encrypted"],
                auth_method=row["auth_method"],
                authorized=bool(row["authorized"]),
            )
        finally:
            conn.close()

    def ensure_user(self, chat_id: int) -> UserRecord:
        existing = self.get_user(chat_id)
        if existing is not None:
            return existing

        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO users(chat_id, status, username_encrypted, password_encrypted, auth_method, authorized, created_at, updated_at)
                VALUES (?, ?, NULL, NULL, NULL, 0, ?, ?)
                """,
                (chat_id, USER_STATUS_NEW, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_user(chat_id)  # type: ignore[return-value]

    def set_user_status(self, chat_id: int, status: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET status = ?, updated_at = ? WHERE chat_id = ?",
                (status, now, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def save_username(self, chat_id: int, username_encrypted: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET username_encrypted = ?, updated_at = ? WHERE chat_id = ?",
                (username_encrypted, now, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def save_password(self, chat_id: int, password_encrypted: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET password_encrypted = ?, updated_at = ? WHERE chat_id = ?",
                (password_encrypted, now, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def save_auth_method(self, chat_id: int, auth_method: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET auth_method = ?, updated_at = ? WHERE chat_id = ?",
                (auth_method, now, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def create_task(self, chat_id: int, query: str) -> str:
        task_id = uuid4().hex
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO tasks(task_id, chat_id, query, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, chat_id, query, TASK_STATUS_QUEUED, now, now),
            )
            conn.commit()
            return task_id
        finally:
            conn.close()

    def set_task_status(self, task_id: str, status: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                (status, now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def complete_task(self, task_id: str, result: dict[str, Any]) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, result_json = ?, error_message = NULL, updated_at = ?
                WHERE task_id = ?
                """,
                (TASK_STATUS_COMPLETED, json.dumps(result), now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def fail_task(self, task_id: str, error_message: str) -> None:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (TASK_STATUS_FAILED, error_message, now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT task_id, chat_id, query, status, result_json, error_message, created_at, updated_at
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def list_recent_tasks(self, chat_id: int, limit: int = 5) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT task_id, query, status, created_at, updated_at
                FROM tasks
                WHERE chat_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

