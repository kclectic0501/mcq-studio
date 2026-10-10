from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: str = "huggingface"
    llm_model: str = "Qwen/Qwen2.5-72B-Instruct"
    hf_token: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    max_retries: int = Field(default=2, ge=0, le=5)
    request_attempts: int = Field(default=3, ge=1, le=6)
    max_concurrency: int = Field(default=3, ge=1, le=8)
    model_timeout: int = Field(default=90, ge=10, le=300)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "mcqgen"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=7860, ge=1024, le=65535)

    @field_validator(
        "hf_token",
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "langsmith_api_key",
        mode="before",
    )
    @classmethod
    def empty_secret(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value
