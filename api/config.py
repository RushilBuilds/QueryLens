from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Reads from environment variables first, then .env file. Pydantic-settings
    chosen over dynaconf or plain os.getenv because it validates types at
    startup — a missing DATABASE_URL surfaces immediately rather than on the
    first query 30 seconds into the process.
    """

    DATABASE_URL: str = "postgresql+psycopg2://localhost:5432/querylens"
    REDPANDA_BROKERS: str = "localhost:9092"
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Module-level factory so FastAPI Depends() can override in tests."""
    return Settings()
