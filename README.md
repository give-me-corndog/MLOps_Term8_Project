# eDimension Telegram Agent

A Telegram bot for SUTD students that combines two capabilities:

- **RAG Document Assistant** — ask questions about your course materials in plain text, powered by Ollama (local LLMs) and ChromaDB
- **eDimension Browser Agent** — automate navigation on the eDimension LMS, download course files, and upload them to DigitalOcean Spaces, powered by `browser-use` with either Google Gemini or Ollama

---

## Project structure

```
.
|-- main.py                         # Entry point
|-- run_eval.py                     # RAG evaluation CLI
|-- eval_dataset.json               # Static RAG eval questions and references
|-- eval_results.jsonl              # RAG eval log output
|-- eval_report_*.json              # Generated RAG aggregate reports
|-- pyproject.toml
|-- evals/
|   |-- agent_service_evals.py      # Static browser-agent eval runner
|   |-- app.db                      # Eval-local SQLite database
|   `-- artifacts/                  # Eval downloads and generated artifacts
|-- artifacts/                      # Runtime artifacts
|-- chroma_store/                   # Persistent ChromaDB vector store
`-- edimension_agent/
    |-- __init__.py
    |-- app.py                      # FastAPI app factory + webhook endpoint
    |-- server.py                   # uvicorn runner
    |-- config.py                   # Settings loaded from environment
    |-- db.py                       # SQLite database (users + tasks)
    |-- crypto.py                   # Fernet credential encryption
    |-- otp_broker.py               # Async OTP request/response coordination
    |-- agent_service.py            # browser-use task runner + Spaces upload
    |-- rag_service.py              # RAG pipeline (ingest, retrieve, generate)
    |-- eval.py                     # RAG LLM-as-judge metrics and reporting
    |-- lmnr_integration.py         # Laminar metric/event helpers
    `-- telegram/
        `-- bot.py                  # aiogram handlers for all commands
```

---

## Architecture

```
Telegram ──► FastAPI webhook ──► bot.py
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
            RAG pipeline     Browser agent    OTP broker
            (rag_service)    (agent_service)  (otp_broker)
                   │               │
                   ▼               ▼
              ChromaDB       DigitalOcean
              + Ollama           Spaces
```

| Component | Technology |
|---|---|
| Bot framework | aiogram 3 |
| Web server | FastAPI + uvicorn |
| Browser automation | browser-use 0.12 |
| LLM (browser agent) | Google Gemini or Ollama (via `browser-use`) |
| LLM (RAG) | Ollama — `ministral-3` |
| Embeddings | Ollama — `qwen3-embedding:0.6b` |
| Vector store | ChromaDB (persistent, per-user collections) |
| PDF parsing | pymupdf4llm |
| File storage | DigitalOcean Spaces (S3-compatible) |
| Database | SQLite |
| Credential encryption | Fernet (AES-128) |

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager and runner)
- [Ollama](https://ollama.com) running and accessible
- A [Telegram bot token](https://t.me/BotFather)
- A Google Gemini API key (for the browser agent)
- A DigitalOcean Spaces bucket
- A public HTTPS URL for the Telegram webhook (e.g. [ngrok](https://ngrok.com) for local dev)

---

## Installation

```bash
# Clone the repo and enter the project directory
git clone <repo-url>
cd <project-dir>/Agent

# Create/sync virtual environment and install dependencies from pyproject.toml
uv sync

# For monitoring
uv pip install --upgrade 'lmnr[all]'

# Pull the required Ollama models
ollama pull ministral-3
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the required values:

```dotenv
# =========================
# App
# =========================
APP_ENV=development
LOG_LEVEL=INFO
APP_BASE_URL=http://127.0.0.1:8000

# =========================
# Server
# =========================
APP_HOST=0.0.0.0
APP_PORT=8000

# =========================
# Telegram
# =========================
TELEGRAM_BOT_TOKEN=<your-telegram-bot-token>
TELEGRAM_WEBHOOK_URL=<your-public-url>/telegram/webhook
TELEGRAM_WEBHOOK_SECRET=<random-secret>        

# =========================
# Encryption
# =========================
APP_ENCRYPTION_KEY=<random-key>              

# =========================
# Database
# =========================
SQLALCHEMY_DATABASE_URL=sqlite:///./app.db

# =========================
# Browser agent LLM (Google Gemini or Ollama)
# =========================
BROWSER_LLM_PROVIDER=google              # google | ollama
GOOGLE_API_KEY=<your-google-api-key>
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0.2
BROWSER_OLLAMA_MODEL=qwen3.5

# =========================
# DigitalOcean Spaces
# =========================
DO_SPACES_KEY=<your-spaces-key>
DO_SPACES_SECRET=<your-spaces-secret>
DO_SPACES_REGION=sgp1
DO_SPACES_BUCKET=<your-bucket-name>
DO_SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
DO_SPACES_PREFIX=edimension
CLEANUP_AFTER_UPLOAD=true

# =========================
# Local paths
# =========================
LOCAL_ARTIFACT_ROOT=./artifacts
DOWNLOADS_DIR=./artifacts/downloads
SCREENSHOTS_DIR=./artifacts/screenshots
VIDEOS_DIR=./artifacts/videos
TRACES_DIR=./artifacts/traces

# =========================
# Ollama
# =========================
OLLAMA_HOST=http://localhost:11434   # or http://<remote-ip>:11434
RAG_EMBED_MODEL=qwen3-embedding:0.6b
RAG_GENERATE_MODEL=ministral-3
RAG_GUARD_MODEL=ministral-3

# =========================
# RAG Settings
# =========================
CHROMA_PATH=./chroma_store
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=300
RAG_TOP_K=2
RAG_MAX_HISTORY=6
RAG_SUMMARY_LIMIT=20
EMBED_WORKERS=4
RERANK_MODEL=ministral-3
RERANK_TOP_N=9

# =========================
# Observability & Evaluation
# =========================
LMNR_ENABLED=true
LMNR_SELF_HOSTED=true
LMNR_PROJECT_API_KEY=<your-laminar-project-api-key-generate-from-localhost-5667>
```

---

## Laminar setup (self-hosted)

Before starting this application:

1. Turn on Docker Desktop.
2. Run:

```bash
git clone https://github.com/lmnr-ai/lmnr
cd lmnr
docker compose up -d
```

Laminar dashboard: http://localhost:5667/

---

## Running

**If Browser Agent uses Ollama (`BROWSER_LLM_PROVIDER=ollama`):**

```bash
# Start Ollama server
ollama serve
```

**Development (with ngrok):**

```bash
# Terminal 1 - Ollama
ollama serve

# Terminal 2 - expose local port
ngrok http 3000
# Copy the https://xxxx.ngrok-free.app URL into TELEGRAM_WEBHOOK_URL in .env

# Terminal 3 - start the bot (from Agent/)
uv run python -m edimension_agent.server
```

**Production:**

Set `TELEGRAM_WEBHOOK_URL` to your real domain, then:

```bash
uv run python -m edimension_agent.server
```

The server listens on `APP_HOST` and `APP_PORT` from `.env`.
The server listens on the host/port passed to uvicorn (default above: `0.0.0.0:8000`).

---

## First-time setup

After starting the bot, **authorize your Telegram account** in the database. The bot requires an explicit authorization flag before responding to anyone — send `/start` to the bot first to create your user row, then run:

```bash
sqlite3 app.db "UPDATE users SET authorized = 1 WHERE chat_id = <your-chat-id>;"
```

You can find your chat ID in the server logs when you send `/start`.

Then, inside Telegram:

```
/setup: enter your eDimension username, password, and MFA method (Okta or Google Auth)
```

---

## Bot commands

### General
| Command | Description |
|---|---|
| `/start` | Register and check authorization status |
| `/setup` | Store your eDimension credentials securely |
| `/help` | Show all available commands |

### Browser agent
| Command | Description |
|---|---|
| `/get <query>` | Run a browser task on eDimension (navigate, download files, etc.) |
| `/otp <code>` | Submit an MFA code when the browser agent requests one |
| `/status [task_id]` | Check the status of recent or specific tasks |

### Document assistant (RAG)
| Command | Description |
|---|---|
| Plain text | Ask a question — answered from your ingested documents |
| `/ingest` | Pull and index PDFs from your DigitalOcean Spaces prefix |
| `/docs` | List all documents in your knowledge base |
| `/clearchat` | Clear conversation history (documents stay indexed) |
| `/cleardocs` | Wipe your entire knowledge base |
| Send a PDF | Upload and ingest a PDF directly from Telegram |

---

## How the RAG pipeline works

1. **Ingest**: PDFs are parsed with `pymupdf4llm`, split into overlapping chunks, embedded with `qwen3-embedding:0.6b` via Ollama, and stored in a per-user ChromaDB collection
2. **Query**: the question is embedded and the top-k most similar chunks are retrieved (distance threshold `< 0.5`)
3. **Guardrails**: an input guardrail checks the question is academic before retrieval; an output guardrail checks the answer before it is sent
4. **Generation**: `ministral-3` generates an answer grounded in the retrieved context, with the last 6 conversation turns included for follow-up questions
5. **Summarisation**: if the question targets a specific document (e.g. "summarise lecture3.pdf"), all chunks for that file are fetched instead of doing similarity search

Each user has their own isolated ChromaDB collection (documents ingested by one user are never visible to another)

---

## How the browser agent works

1. The user sends `/get <query>` describing what to do on eDimension
2. The agent logs in using the user's stored (encrypted) credentials
3. If MFA is required, the bot sends the user an OTP prompt; the user replies with `/otp <code>`
4. The agent navigates the LMS, downloads any requested PDFs, and uploads them to DigitalOcean Spaces
5. Downloaded files are sent back to the user via Telegram
6. Temporary browser files are cleaned up after the task completes

---

## Static evaluations

The `evals/` folder contains static evaluations for the browser-agent service. These evals run a fixed set of eDimension tasks through `BrowserTaskRunner`, record each task as success or failure, and export aggregate metrics to Laminar. The current eval set lives in `evals/agent_service_evals.py` as `EVAL_QUERIES` and includes login, course navigation, assignment-score lookup, topic listing, and file-download tasks.

Each run records:

- `success` / `failure` for each browser-agent task
- `duration_seconds` from the browser-use task logs, when available
- `total_cost` from the browser-use task logs, when available
- task summaries, uploaded-file counts, final errors, and raw logs

Before running the evals, make sure Laminar is running and `.env` contains a valid `LMNR_PROJECT_API_KEY`. For self-hosted Laminar, open http://localhost:5667/, create or select a project, and copy the project API key into `.env`.

Run the static browser-agent evals from `Agent/`:

```bash
uv run python evals/agent_service_evals.py \
  --username "<EDIM_USERNAME>" \
  --password "<EDIM_PASSWORD>" \
  --auth-method okta \
  --group-name agent_service_success_failure
```

Use `--auth-method google_auth` if the test account uses Google Authenticator. If MFA is required during the run, the eval process prints an OTP prompt in the terminal; enter the current code to continue.

Useful options:

```bash
# Run one or more custom queries instead of the default static list
uv run python evals/agent_service_evals.py \
  --username "<EDIM_USERNAME>" \
  --password "<EDIM_PASSWORD>" \
  --auth-method okta \
  --query "Download MLOps Week 1 Lectures Notes"

# Write JSONL results to a custom path
uv run python evals/agent_service_evals.py \
  --username "<EDIM_USERNAME>" \
  --password "<EDIM_PASSWORD>" \
  --output evals/agent_service_eval_results.jsonl
```

To view the run in Laminar, open the dashboard at http://localhost:5667/, select the project that owns `LMNR_PROJECT_API_KEY`, then look for the evaluation group named by `--group-name` (default: `agent_service_success_failure`). The group shows the exported `success`, `failure`, `duration_seconds`, and `total_cost` evaluator values for the static run. The browser-agent traces are also tagged as eval traces by `agent_service.py`, which makes successful and failed eval tasks easier to filter from live Telegram traffic.

The repo also includes `run_eval.py`, which evaluates the RAG assistant against `eval_dataset.json` using local LLM-as-judge metrics. Run it from `Agent/` with:

```bash
uv run python run_eval.py --chat-id <telegram-chat-id>
uv run python run_eval.py --stats
uv run python run_eval.py --recent 10
```

---

## Security notes

- eDimension credentials are encrypted with Fernet (AES-128-CBC + HMAC) before being written to SQLite. The encryption key is never stored in the database.
- The Telegram webhook is protected by a secret token validated on every request.
- Users must be explicitly authorized in the database by an admin — the bot will not respond to unauthorized accounts.
- The RAG pipeline runs input and output guardrails on every request to prevent misuse of the assistant.
- DigitalOcean Spaces credentials are only used server-side; they are never sent to the user.
