from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

try:
    from lmnr import Laminar
except ImportError:
    Laminar = None  # type: ignore[assignment]

from .agent_service import BrowserTaskRunner
from .config import Settings, load_settings
from .crypto import CredentialCipher
from .db import Database
from .otp_broker import OtpBroker
from .telegram.bot import TelegramAgentBot
try:
    from evals import lmnr_integration
except ImportError:
    try:
        from Agent.evals import lmnr_integration
    except ImportError:
        lmnr_integration = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _initialize_laminar(settings: Settings) -> None:
    if not settings.lmnr_enabled:
        logger.info("Laminar disabled (LMNR_ENABLED=false)")
        return
    if Laminar is None:
        logger.warning("Laminar SDK is not installed; observability disabled")
        return
    if not settings.lmnr_project_api_key:
        logger.warning("LMNR_ENABLED is true but LMNR_PROJECT_API_KEY is empty; observability disabled")
        return

    try:
        if settings.lmnr_self_hosted and settings.lmnr_project_api_key:
            Laminar.initialize(
                project_api_key=settings.lmnr_project_api_key,
                base_url="http://localhost",
                http_port=settings.lmnr_http_port,
                grpc_port=settings.lmnr_grpc_port,
            )
            logger.info(f"Laminar initialized with self-hosted endpoints (http:{settings.lmnr_http_port} grpc:{settings.lmnr_grpc_port})")
        else:
            Laminar.initialize(project_api_key=settings.lmnr_project_api_key)
            logger.info("Laminar initialized with Laminar Cloud endpoint")
    except Exception as exc:
        logger.warning("Laminar initialization failed; observability disabled: %s", exc)


def create_app() -> FastAPI:
    load_dotenv()

    settings: Settings = load_settings()
    db = Database(settings.db_path)
    db.init()
    cipher = CredentialCipher(settings.app_encryption_key)
    otp_broker = OtpBroker()
    task_runner = BrowserTaskRunner(settings=settings, db=db, otp_broker=otp_broker)
    telegram_bot = TelegramAgentBot(
        settings=settings,
        db=db,
        cipher=cipher,
        task_runner=task_runner,
        otp_broker=otp_broker,
    )

    app = FastAPI(title="eDimension Telegram Agent")
    app.state.settings = settings
    app.state.telegram_bot = telegram_bot

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("App startup begin")
        _initialize_laminar(settings)
        lmnr_integration.initialize(enabled=settings.lmnr_enabled)
        await telegram_bot.start()
        logger.info("Telegram webhook configured")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await telegram_bot.stop()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/telegram/webhook")
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> dict[str, bool]:
        if x_telegram_bot_api_secret_token != settings.telegram_webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

        update = await request.json()
        await telegram_bot.handle_update(update)
        return {"ok": True}

    return app
