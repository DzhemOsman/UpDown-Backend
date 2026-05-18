from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    INFLUXDB_HOST: str
    INFLUXDB_TOKEN: str
    INFLUXDB_DATABASE: str
    CORS_ORIGINS: list[str] = ["*"] 

settings = Settings()