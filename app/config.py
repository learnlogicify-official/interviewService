"""Runtime settings for the AI interview service."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Nex AI Interview"
    database_url: str = "sqlite:///./interview.db"
    shared_secret: str = "dev-change-me-interview-secret"
    token_ttl_seconds: int = 60 * 60 * 4
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    default_duration_minutes: int = 30
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
