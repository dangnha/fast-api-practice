from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Inference Platform"
    environment: str = "development"
    database_url: str = "sqlite:///./fastapi_ai.db"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "change-me"
    access_token_minutes: int = 30
    cors_origins: list[str] = ["http://localhost:3000"]
    model_path: str = "models/mnist_cnn.pt"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
