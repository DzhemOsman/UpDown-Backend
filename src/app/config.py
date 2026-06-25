from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
MEASUREMENT = "ohlcv_data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    INFLUXDB_HOST: str
    INFLUXDB_TOKEN: SecretStr
    INFLUXDB_DATABASE: str
    OPTIMIZER_MAX_WORKERS: int = 3

    CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:8184"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
