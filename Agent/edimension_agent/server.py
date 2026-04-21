from __future__ import annotations
 
import logging
import os
import uvicorn
from dotenv import load_dotenv
from .app import create_app
from .config import load_settings
 
load_dotenv()


def main() -> None:
    settings = load_settings()
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting server on %s:%s", settings.bind_host, settings.bind_port)
    app = create_app()
    uvicorn.run(app, host=settings.bind_host, port=settings.bind_port)
 
 
if __name__ == "__main__":
    main()