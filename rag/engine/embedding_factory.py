from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI as OpenAIClient,
)
from pydantic import Field, PrivateAttr

import config
from exceptions import RAGError

logger = logging.getLogger(__name__)

def _should_use_openai_embedding(model_name: str, api_base: Optional[str]) -> bool:
    if api_base and "api.openai.com" not in api_base:
        return False
    return model_name.startswith("text-embedding-") or model_name == "text-embedding-ada-002"


class OpenAICompatibleEmbedding(BaseEmbedding):
    model_name: str = Field(default="unknown", description="Embedding model name.")
    api_key: Optional[str] = Field(default=None, exclude=True)
    api_base: Optional[str] = Field(default=None, exclude=True)
    _client: OpenAIClient = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str],
        api_base: Optional[str],
    ) -> None:
        super().__init__(model_name=model_name)
        self.api_key = api_key
        self.api_base = api_base
        self._client = OpenAIClient(api_key=api_key, base_url=api_base)

    _MAX_EMBED_CHARS: int = 4000

    def _get_text_embedding(self, text: str) -> Embedding:
        if len(text) > self._MAX_EMBED_CHARS:
            text = text[: self._MAX_EMBED_CHARS]

        retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = self._client.embeddings.create(input=[text], model=self.model_name)
                return response.data[0].embedding
            except BadRequestError as exc:
                message = str(exc).lower()
                if "context length" in message or "input length" in message:
                    text = text[: len(text) // 2]
                    if not text:
                        raise
                    logger.debug("Truncating text to %d chars for embedding", len(text))
                    continue
                raise
            except InternalServerError as exc:
                last_error = exc
                message = str(exc).lower()
                retriable = "model failed to load" in message or "resource limitations" in message
                if not retriable or attempt == retries:
                    raise
                time.sleep(min(1.5 * attempt, 4.0))
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                if attempt == retries:
                    raise
                time.sleep(min(0.8 * attempt, 2.4))
        if last_error is not None:
            try:
                msg = str(last_error).lower()
            except Exception:
                msg = ""
            if isinstance(last_error, InternalServerError) and "model failed to load" in msg:
                logger.warning(
                    "Embedding model load failed repeatedly; using fallback sha256-based embedding for query"
                )
                h = hashlib.sha256(text.encode("utf-8")).digest()
                vec: list[float] = []
                dim = getattr(config.settings, "EMBED_DIM", 1536)
                while len(vec) < dim:
                    h = hashlib.sha256(h).digest()
                    for b in h:
                        if len(vec) >= dim:
                            break
                        vec.append((b / 255.0) * 2.0 - 1.0)
                return vec
            raise last_error
        raise RAGError("Embedding request failed")

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)


class RAGEmbeddingFactory:
    """Factory for the embedding model and embedding-dimension management."""

    def __init__(self, engine) -> None:
        self._engine = engine

    @staticmethod
    def create_embed_model() -> BaseEmbedding:
        api_base = config.settings.OPENAI_API_URL
        if _should_use_openai_embedding(config.settings.EMBED_MODEL, api_base):
            try:
                return OpenAIEmbedding(
                    api_key=config.settings.OPENAI_API_KEY,
                    model=config.settings.EMBED_MODEL,
                    dimensions=config.settings.EMBED_DIM,
                    api_base=api_base,
                )
            except ValueError as exc:
                logger.warning(
                    "OpenAI embedding model '%s' not recognized; using compatible client: %s",
                    config.settings.EMBED_MODEL,
                    exc,
                )
                return OpenAICompatibleEmbedding(
                    model_name=config.settings.EMBED_MODEL,
                    api_key=config.settings.OPENAI_API_KEY,
                    api_base=api_base,
                )
        else:
            return OpenAICompatibleEmbedding(
                model_name=config.settings.EMBED_MODEL,
                api_key=config.settings.OPENAI_API_KEY,
                api_base=api_base,
            )

    @property
    def embed_dim(self) -> int:
        if self._engine._embed_dim is None:
            self._engine._embed_dim = self._get_embedding_dim()
        return self._engine._embed_dim

    def _get_embedding_dim(self) -> int:
        if config.settings.EMBED_DIM:
            return config.settings.EMBED_DIM
        sample_embed = self._engine.embed_model.get_text_embedding("test")
        return len(sample_embed)
