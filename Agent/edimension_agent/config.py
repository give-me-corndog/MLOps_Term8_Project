from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(raw: str, default: bool = False) -> bool:
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_webhook_url: str
    telegram_webhook_secret: str
    bind_host: str
    bind_port: int

    app_encryption_key: str
    db_path: Path

    downloads_dir: Path

    browser_llm_provider: str

    google_model: str
    google_temperature: float

    browser_ollama_model: str
    browser_ollama_host: str
    lmnr_enabled: bool
    lmnr_project_api_key: str
    lmnr_self_hosted: bool

    do_spaces_key: str
    do_spaces_secret: str
    do_spaces_region: str
    do_spaces_bucket: str
    do_spaces_endpoint: str
    do_spaces_prefix: str
    cleanup_after_upload: bool



def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
    webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    bind_host = os.getenv("APP_HOST", "0.0.0.0").strip()
    bind_port = int(os.getenv("APP_PORT", "8000"))

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    if not webhook_url:
        raise ValueError("TELEGRAM_WEBHOOK_URL is required for webhook mode")
    if not webhook_secret:
        raise ValueError("TELEGRAM_WEBHOOK_SECRET is required")

    encryption_key = os.getenv("APP_ENCRYPTION_KEY", "").strip()
    if not encryption_key:
        raise ValueError("APP_ENCRYPTION_KEY is required")

    db_url = os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///./app.db").strip()
    if not db_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite URLs are currently supported")
    db_path = Path(db_url.replace("sqlite:///", "", 1)).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    downloads_dir = Path(os.getenv("DOWNLOADS_DIR", "./artifacts/downloads")).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    do_spaces_region = os.getenv("DO_SPACES_REGION", "sgp1").strip()
    do_spaces_endpoint = os.getenv(
        "DO_SPACES_ENDPOINT",
        f"https://{do_spaces_region}.digitaloceanspaces.com",
    ).strip()

    return Settings(
        telegram_bot_token=token,
        telegram_webhook_url=webhook_url,
        telegram_webhook_secret=webhook_secret,
        bind_host=bind_host,
        bind_port=bind_port,
        app_encryption_key=encryption_key,
        db_path=db_path,
        downloads_dir=downloads_dir,
        browser_llm_provider=os.getenv("BROWSER_LLM_PROVIDER", "google").strip().lower(),
        google_model=os.getenv("GOOGLE_MODEL", "gemini-flash-latest").strip(),
        google_temperature=float(os.getenv("GOOGLE_TEMPERATURE", "0.2")),
        browser_ollama_model=os.getenv("BROWSER_OLLAMA_MODEL", "qwen3.5").strip(),
        browser_ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").strip(),
        lmnr_enabled=_parse_bool(os.getenv("LMNR_ENABLED", "false"), False),
        lmnr_project_api_key=os.getenv("LMNR_PROJECT_API_KEY", "").strip(),
        lmnr_self_hosted=os.getenv("LMNR_SELF_HOSTED", "true").strip(),
        do_spaces_key=os.getenv("DO_SPACES_KEY", "").strip(),
        do_spaces_secret=os.getenv("DO_SPACES_SECRET", "").strip(),
        do_spaces_region=do_spaces_region,
        do_spaces_bucket=os.getenv("DO_SPACES_BUCKET", "").strip(),
        do_spaces_endpoint=do_spaces_endpoint,
        do_spaces_prefix=os.getenv("DO_SPACES_PREFIX", "edimension").strip(),
        cleanup_after_upload=_parse_bool(os.getenv("CLEANUP_AFTER_UPLOAD", "true"), True),
    )
