# Architecture Summary

## 1. Goal

This project was refactored from a single hard-coded browser task script into a Telegram-driven automation service that:

1. Enforces setup-before-use for each Telegram user.
2. Stores eDimension credentials encrypted at rest.
3. Runs dynamic browser-use tasks from free-form user queries.
4. Handles OTP challenges through Telegram.
5. Uploads downloaded PDFs to DigitalOcean Spaces and returns results to users.

## 2. High-Level Runtime

The runtime is now a webhook-based FastAPI service.

1. Telegram sends updates to `POST /telegram/webhook`.
2. The app validates Telegram webhook secret.
3. Aiogram routes commands/messages.
4. Setup state and tasks are persisted in SQLite.
5. Browser-use task execution runs asynchronously per submitted query.
6. OTP waits are bridged by an in-memory async OTP broker.

## 3. Module Structure

## Entry and App Composition

1. `main.py`
   1. Thin launcher that adds `src` to `sys.path` and starts server.
2. `src/edimension_agent/server.py`
   1. Starts Uvicorn with configured host/port.
3. `src/edimension_agent/app.py`
   1. Loads env.
   2. Builds shared services (`Settings`, `Database`, `CredentialCipher`, `OtpBroker`, `BrowserTaskRunner`, `TelegramAgentBot`).
   3. Registers startup/shutdown hooks and API routes.

## Core Services

1. `src/edimension_agent/config.py`
   1. Centralized environment parsing and validation.
   2. Supports strict chat allowlist from one or many chat IDs.
2. `src/edimension_agent/crypto.py`
   1. Fernet-based credential encryption/decryption.
   2. Normalizes non-Fernet keys by SHA-256 deriving a valid Fernet key.
3. `src/edimension_agent/db.py`
   1. SQLite repository layer for users and tasks.
   2. Tracks setup status and task lifecycle states.
4. `src/edimension_agent/otp_broker.py`
   1. Async queue/future broker for OTP challenge -> user response mapping.
5. `src/edimension_agent/agent_service.py`
   1. Browser-use task runner for dynamic user queries.
   2. Registers browser tools:
      1. Ask human for OTP.
      2. Upload newest downloaded PDF to Spaces.
   3. Scopes downloads per `chat_id/task_id` to improve isolation.

## Telegram Interface

1. `src/edimension_agent/telegram/bot.py`
   1. Aiogram handlers for `/start`, `/setup`, `/otp`, `/status`, `/help`, plus plain text queries.
   2. Enforces authorization (allowlist).
   3. Enforces setup-first workflow.
   4. Dispatches async task execution and status updates.
   5. Sends downloaded files back to Telegram and also returns Spaces key/URL.

## 4. User and Task Flows

## Setup and Access Control

1. User must be in allowlist (`AUTHORIZED_TELEGRAM_CHAT_ID` or `AUTHORIZED_TELEGRAM_CHAT_IDS`).
2. User runs `/setup`.
3. Bot asks for username then password in sequence.
4. Credentials are encrypted before writing to DB.
5. User status moves to `ready`.

## Query Execution

1. Ready user sends plain text query.
2. Task record is created as `queued` then `running`.
3. Browser-use agent executes dynamic task prompt with user credentials in sensitive memory only.
4. If OTP is required:
   1. Task status becomes `waiting_otp`.
   2. Bot prompts user.
   3. User replies `/otp <code>`.
   4. Runner resumes.
5. On success:
   1. Task stores summary and uploaded file metadata.
   2. Bot sends completion response.

## File Download Path

1. Browser task downloads PDF(s).
2. Tool locates newest PDF from task and browser temp locations.
3. File is uploaded to Spaces with key pattern:

`{DO_SPACES_PREFIX}/{student_id}/{task_id}/{filename}`

4. Bot sends:
   1. Telegram document upload (local staged file).
   2. Spaces object key and public URL.

## Non-File Path

1. Browser task completes without file upload.
2. Bot sends textual summary from browser-use result.

## 5. Persistence Model

## users table

1. `chat_id` (PK)
2. `status` (`new`, `waiting_username`, `waiting_password`, `ready`)
3. `username_encrypted`
4. `password_encrypted`
5. timestamps

## tasks table

1. `task_id` (PK)
2. `chat_id` (FK)
3. `query`
4. `status` (`queued`, `running`, `waiting_otp`, `completed`, `failed`)
5. `result_json`
6. `error_message`
7. timestamps

## 6. API Surface

1. `GET /health`
   1. Liveness endpoint.
2. `POST /telegram/webhook`
   1. Validates `X-Telegram-Bot-Api-Secret-Token`.
   2. Feeds update to aiogram dispatcher.

## 7. Configuration and Dependency Updates

## Environment

Updated `.env.example` with:

1. Webhook settings (`TELEGRAM_WEBHOOK_URL`, `TELEGRAM_WEBHOOK_SECRET`).
2. Host/port (`APP_HOST`, `APP_PORT`).
3. Multi-chat allowlist option (`AUTHORIZED_TELEGRAM_CHAT_IDS`).
4. Browser model tuning (`GOOGLE_MODEL`, `GOOGLE_TEMPERATURE`).
5. Upload cleanup behavior (`CLEANUP_AFTER_UPLOAD`).

## Python Dependencies

`pyproject.toml` was expanded to include runtime packages for:

1. Telegram handling (`aiogram`).
2. Webhook server (`fastapi`, `uvicorn`).
3. Encryption (`cryptography`).
4. Storage/upload and config support (`boto3`, `python-dotenv`, `pydantic`).

## 8. Notable Implementation Decisions

1. Kept architecture simple for MVP: in-process async execution and OTP broker.
2. Kept DB lightweight with SQLite.
3. Added per-task download directory isolation.
4. Preserved browser-use + Spaces integration pattern from original script while making it dynamic.

## 9. Current Known Gaps

1. OTP broker is in-memory only (no persistence across process restarts).
2. No worker queue backend yet (Redis/Celery not added).
3. Limited automated tests in this pass.
4. Cleanup of browser temp folders is not globally orchestrated anymore; cleanup is focused on local staged files used for Telegram return.

## 10. Documentation Update

`README.md` now reflects:

1. Bot command usage.
2. Setup gating behavior.
3. Webhook run instructions.
4. Implemented file/non-file response behavior.
