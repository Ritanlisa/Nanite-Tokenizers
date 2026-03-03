from __future__ import annotations

from typing import Optional, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, yaml_file="settings.yaml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            init_settings,
            file_secret_settings,
        )

    ENV: Literal["dev", "test", "prod"] = Field("dev", validation_alias="ENV")

    OPENAI_API_KEY: str = Field(..., validation_alias="OPENAI_API_KEY")
    OPENAI_API_URL: Optional[str] = Field(None, validation_alias="OPENAI_API_URL")
    LLM_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = Field(0.0, ge=0.0, le=2.0)
    EMBED_MODEL: str = "text-embedding-3-small"
    EMBED_DIM: Optional[int] = None

    CHUNK_SIZE: int = Field(512, ge=100, le=2048)
    CHUNK_OVERLAP: int = Field(50, ge=0, le=200)
    SIMILARITY_TOP_K: int = Field(5, ge=1, le=20)
    ENABLE_RERANK: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = Field(3, ge=1)
    RAG_CONFIDENCE_THRESHOLD: float = Field(0.8, ge=0.0, le=1.0)
    MIN_RAG_SOURCES: int = Field(1, ge=1)
    VECTOR_STORE_TYPE: Literal["chroma", "faiss"] = "chroma"
    FAISS_INDEX_TYPE: str = "Flat"
    SAMPLE_FOR_TRAINING: int = Field(1000, ge=1)
    PERSIST_DIR: str = "./storage"
    DATA_DIR: str = "./data"
    VECTOR_STORE_FALLBACK_WARNING: bool = True

    CACHE_TYPE: Literal["memory", "redis"] = "memory"
    REDIS_URL: Optional[str] = Field(None, validation_alias="REDIS_URL")
    CACHE_TTL: int = Field(3600, ge=60)
    MEMORY_CACHE_MAXSIZE: int = Field(1000, ge=10)

    MCP_FETCH_SERVER_COMMAND: str = "npx -y @modelcontextprotocol/server-fetch"
    MCP_TIMEOUT: int = Field(30, ge=5)
    MCP_MAX_RESTART: int = Field(3, ge=0, le=10)

    MCP_RETRY_TIMES: int = Field(2, ge=0, le=5)
    MCP_RETRY_DELAY: float = Field(1.0, ge=0.1)

    AGENT_VERBOSE: bool = Field(False, validation_alias="AGENT_VERBOSE")
    MAX_ITERATIONS: int = Field(5, ge=1, le=20)
    MAX_TOTAL_TOKENS: int = Field(4000, ge=1000)
    MAX_HISTORY_ROUNDS: int = Field(10, ge=1, le=50)

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5

    BATCH_CONCURRENCY: int = Field(5, ge=1)

    OCR_API_KEY: Optional[str] = Field(None, validation_alias="OCR_API_KEY")
    OCR_API_URL: Optional[str] = Field(None, validation_alias="OCR_API_URL")
    OCR_MODEL: Optional[str] = Field(None, validation_alias="OCR_MODEL")
    OCR_PROMPT: str = Field(
        "<image>\n<|grounding|>Convert the document to markdown.",
        validation_alias="OCR_PROMPT",
    )
    OCR_TIMEOUT: int = Field(120, ge=10)

    @field_validator("RERANK_TOP_N")
    @classmethod
    def validate_rerank_top_n(cls, value: int, info):
        if value > info.data.get("SIMILARITY_TOP_K", 5):
            raise ValueError("RERANK_TOP_N must be <= SIMILARITY_TOP_K")
        return value

    @field_validator("OPENAI_API_URL")
    @classmethod
    def normalize_openai_api_url(cls, value: Optional[str]):
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_rerank_config(self):
        if not self.OPENAI_API_URL and len(self.OPENAI_API_KEY) < 20:
            raise ValueError("OPENAI_API_KEY must be at least 20 characters unless OPENAI_API_URL is set")
        if self.RERANK_TOP_N > self.SIMILARITY_TOP_K:
            raise ValueError("RERANK_TOP_N must be <= SIMILARITY_TOP_K")
        return self

    def update(self, **kwargs) -> "Settings":
        data = self.model_dump()
        data.update(kwargs)
        return Settings(**data)


settings = Settings()  # pyright: ignore[reportCallIssue]
