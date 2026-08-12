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
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_tts_base_url: str = ""  # default: same as openai_base_url (use api.openai.com for TTS)
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "alloy"
    openai_stt_base_url: str = ""
    openai_stt_model: str = "whisper-1"
    openai_realtime_model: str = "gpt-realtime"
    openai_realtime_voice: str = "alloy"
    # Prefer realtime WebRTC voice when client requests a token.
    voice_mode: str = "realtime"  # realtime|legacy
    default_duration_minutes: int = 30
    cors_origins: str = "*"
    # How many Q&A exchanges before coding (LLM may move earlier/later).
    qa_target_exchanges: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
