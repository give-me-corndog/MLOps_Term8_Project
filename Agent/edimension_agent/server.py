from __future__ import annotations
 
import uvicorn
from dotenv import load_dotenv
from .app import create_app
from .config import load_settings
 
load_dotenv()
def main() -> None:
    settings = load_settings()
    app = create_app()
    uvicorn.run(app, host=settings.bind_host, port=settings.bind_port)
 
 
if __name__ == "__main__":
    main()