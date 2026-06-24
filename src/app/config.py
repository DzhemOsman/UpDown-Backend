from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
MEASUREMENT = "ohlcv_data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    INFLUXDB_HOST: str
    INFLUXDB_TOKEN: str
    INFLUXDB_DATABASE: str
    CORS_ORIGINS: list[str] = ["http://localhost:8184"]
    OPTIMIZER_MAX_WORKERS: int = 3

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str] | Any:
        if isinstance(v, str) and not v.startswith("["):
            return [item.strip() for item in v.split(",")]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
