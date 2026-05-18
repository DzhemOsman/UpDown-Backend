from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "UpDown Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS — Vite dev server runs on 5173 by default
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # InfluxDB Core 3
    influxdb_host: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_database: str = "updown"


settings = Settings()
