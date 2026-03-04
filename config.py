from __future__ import annotations

from typing import Optional, Literal
import os

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
            init_settings,                     # 最高优先级：通过 __init__ 传入的值
            env_settings,                       # 次高：环境变量
            YamlConfigSettingsSource(settings_cls),  # 基础：YAML 文件
            file_secret_settings,                # 最低：密钥文件
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
    ENABLE_RAG: bool = True
    RAG_TOOL_TIMEOUT: int = Field(90, ge=5, le=600)
    RAG_REGEX_RETRIEVE_TIMEOUT: int = Field(90, ge=5, le=600)
    RAG_VECTOR_RETRIEVE_TIMEOUT: int = Field(90, ge=5, le=600)
    RAG_SYNTHESIZE_ANSWER: bool = False
    RAG_SYNTHESIS_TIMEOUT: int = Field(45, ge=5, le=300)
    OFFLINE_ONLY: bool = False
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    RERANK_LOCAL_DIR: str = "./models/rerank"
    RERANK_TOP_N: int = Field(3, ge=1)
    RAG_CONFIDENCE_THRESHOLD: float = Field(0.8, ge=0.0, le=1.0)
    MIN_RAG_SOURCES: int = Field(1, ge=1)
    VECTOR_STORE_TYPE: Literal["chroma", "faiss"] = "chroma"
    FAISS_INDEX_TYPE: str = "Flat"
    SAMPLE_FOR_TRAINING: int = Field(1000, ge=1)
    PERSIST_DIR: str = "./database"
    RAG_DB_NAME: Optional[str] = None
    RAG_DB_NAMES: list[str] = Field(default_factory=list)
    DATA_DIR: str = "./data"
    VECTOR_STORE_FALLBACK_WARNING: bool = True

    CACHE_TYPE: Literal["memory", "redis"] = "memory"
    REDIS_URL: Optional[str] = Field(None, validation_alias="REDIS_URL")
    CACHE_TTL: int = Field(3600, ge=60)
    MEMORY_CACHE_MAXSIZE: int = Field(1000, ge=10)

    MCP_FETCH_SERVER_COMMAND: str = "python -m mcp_server_fetch"
    MCP_SERVER_URLS: list[str] = Field(default_factory=list)
    MCP_INIT_TIMEOUT: int = Field(60, ge=5)
    MCP_TIMEOUT: int = Field(30, ge=5)
    MCP_MAX_RESTART: int = Field(3, ge=0, le=10)

    MCP_RETRY_TIMES: int = Field(2, ge=0, le=5)
    MCP_RETRY_DELAY: float = Field(1.0, ge=0.1)

    AGENT_VERBOSE: bool = Field(False, validation_alias="AGENT_VERBOSE")
    MAX_ITERATIONS: int = Field(5, ge=1, le=20)
    RECURSION_LIMIT: int = Field(50, ge=5, le=1000)
    LLM_REQUEST_TIMEOUT: int = Field(120, ge=10, le=600)
    AGENT_INVOKE_TIMEOUT: int = Field(180, ge=10, le=1200)
    AGENT_LLM_RETRY_TIMES: int = Field(2, ge=0, le=5)
    AGENT_LLM_RETRY_DELAY: float = Field(1.0, ge=0.1, le=30.0)
    MAX_TOTAL_TOKENS: int = Field(4000, ge=1000)
    MAX_HISTORY_ROUNDS: int = Field(10, ge=1, le=50)
    SYSTEM_PROMPT: Optional[str] = None

    ENABLE_AGENT_SKILLS: bool = True
    ENABLE_SHELL_SKILL: bool = True
    ENABLE_FILE_IO_SKILL: bool = True
    ENABLE_SEARCH_SKILL: bool = True
    SHELL_SKILL_TIMEOUT: int = Field(20, ge=1, le=120)
    SKILL_MAX_FILE_BYTES: int = Field(200_000, ge=1_024, le=5_000_000)
    SKILL_OUTPUT_MAX_CHARS: int = Field(12_000, ge=500, le=100_000)
    SEARCH_TOP_K: int = Field(5, ge=1, le=10)
    SEARCH_TIMEOUT: int = Field(15, ge=3, le=90)
    SEARCH_USER_AGENT: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    SEARCH_WEBDRIVER_KIND: Literal["edge", "chrome", "firefox"] = "edge"
    SEARCH_WEBDRIVER_PATH: str = "./edgedriver_linux64/msedgedriver"
    SEARCH_URL: str = "https://cn.bing.com/"
    SEARCH_BOX_XPATH: str = "/html/body/div[1]/div/div[3]/div[2]/form"
    SEARCH_BUTTON_XPATH: str = "/html/body/div[1]/div/div[3]/div[2]/form/label"
    SEARCH_RESULT_REGEX: str = (
        r'<li[^>]*class="[^\"]*b_algo[^\"]*"[^>]*>.*?<h2[^>]*>.*?<a[^>]+href="(?P<link>[^\"]+)"[^>]*>(?P<title>.*?)</a>.*?</h2>(?:.*?<p[^>]*>(?P<snippet>.*?)</p>)?'
    )
    SEARCH_PAGE_PARAM: str = "first"
    SEARCH_PAGE_SIZE: int = Field(10, ge=1, le=100)

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
        os.environ["OFFLINE_ONLY"] = "1" if self.OFFLINE_ONLY else "0"
        if self.OFFLINE_ONLY:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
        if not self.OPENAI_API_URL and len(self.OPENAI_API_KEY) < 20:
            raise ValueError("OPENAI_API_KEY must be at least 20 characters unless OPENAI_API_URL is set")
        if self.RERANK_TOP_N > self.SIMILARITY_TOP_K:
            raise ValueError("RERANK_TOP_N must be <= SIMILARITY_TOP_K")
        self.RAG_DB_NAMES = [item.strip() for item in self.RAG_DB_NAMES if item and item.strip()]
        deduped_db_names: list[str] = []
        for item in self.RAG_DB_NAMES:
            if item not in deduped_db_names:
                deduped_db_names.append(item)
        self.RAG_DB_NAMES = deduped_db_names
        if self.RAG_DB_NAMES and not self.RAG_DB_NAME:
            self.RAG_DB_NAME = self.RAG_DB_NAMES[0]
        self.MCP_SERVER_URLS = [item.strip() for item in self.MCP_SERVER_URLS if item and item.strip()]
        return self

    def update(self, **kwargs) -> "Settings":
        data = self.model_dump()
        data.update(kwargs)
        return Settings(**data)


settings = Settings()  # pyright: ignore[reportCallIssue]


def get_rag_persist_dir() -> str:
    base_dir = settings.PERSIST_DIR
    name = (settings.RAG_DB_NAME or "").strip()
    if name:
        return os.path.join(base_dir, name)
    return base_dir
