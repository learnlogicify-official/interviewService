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
    openai_base_url: str = "https://api.openai.com/v1"
    openai_tts_base_url: str = ""  # default: same as openai_base_url (use api.openai.com for TTS)
    openai_tts_model: str = "tts-1-hd"
    openai_tts_voice: str = "coral"
    openai_stt_base_url: str = ""
    openai_stt_model: str = "whisper-1"
    openai_realtime_model: str = "gpt-realtime"
    openai_realtime_voice: str = "coral"
    # Prefer realtime WebRTC voice when client requests a token.
    voice_mode: str = "realtime"  # realtime|legacy
    # Gladia live STT — get key at https://app.gladia.io (API host: api.gladia.io)
    gladia_api_key: str = ""
    gladia_api_base: str = "https://api.gladia.io"
    # auto = Gladia when key set, else OpenAI Whisper for batch /stt
    stt_provider: str = "auto"  # auto|gladia|openai
    default_duration_minutes: int = 17
    cors_origins: str = "*"
    # Share of the session: technical Q&A / coding / spoken wrap.
    qa_share: float = 0.30
    coding_share: float = 0.65
    wrap_share: float = 0.05
    qa_seconds: int = 0  # 0 = derive from qa_share * duration
    coding_seconds: int = 0
    wrap_seconds: int = 0
    qa_target_exchanges: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
