"""FastAPI entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.db import init_db
from app.routers import sessions

settings = get_settings()

app = FastAPI(title=settings.app_name, version=__version__)
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(sessions.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def root() -> dict:
    return {"service": "interview-service", "version": __version__}
