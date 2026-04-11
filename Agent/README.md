# eDimension Telegram Browser Agent

This project runs a Telegram bot that triggers browser-use to log in to eDimension and execute free-form navigation tasks.

## Implemented Flow

1. New users must run `/setup` before they can submit tasks.
2. Setup collects eDimension username/password and stores both encrypted at rest in SQLite.
3. After setup, users can send plain-text queries.
4. Tasks run with browser-use and can request OTP via Telegram (`/otp <code>`).
5. For download flows, PDFs are uploaded to DigitalOcean Spaces and also sent back in Telegram.
6. For non-download flows, the bot sends a text summary.

## Commands

- `/start` initialize access and setup reminder
- `/setup` start credential setup wizard
- `/otp <code>` provide OTP to resume waiting task
- `/status [task_id]` check one task or list recent tasks
- `/help` show command list

## Environment

Copy `.env.example` to `.env` and fill values:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_URL`
- `TELEGRAM_WEBHOOK_SECRET`
- `AUTHORIZED_TELEGRAM_CHAT_ID` or `AUTHORIZED_TELEGRAM_CHAT_IDS`
- `APP_ENCRYPTION_KEY`
- `GOOGLE_API_KEY`
- `DO_SPACES_*`

## Run

```bash
uv sync
uv run main.py
```

Server endpoints:

- `GET /health`
- `POST /telegram/webhook`

Telegram should be configured to call the webhook URL in your environment settings.
