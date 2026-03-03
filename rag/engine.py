from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, cast
import atexit

import faiss
import numpy as np
from cachetools import TTLCache
from huggingface_hub import snapshot_download
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from redis import Redis
from redis.exceptions import RedisError
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, OpenAI as OpenAIClient
from pydantic import Field, PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

import config
from exceptions import APIThrottlingError, QueryTimeoutError, RAGError
from monitoring import rag_cache_hit_ratio, rag_query_count, rag_query_latency
from rag.documents import (
    RAG_DB_Document,
    load_chunk_documents_from_paths,
    load_chunk_documents_from_data_dir,
    load_chunk_documents_from_persist_dir,
    load_rag_documents_from_persist_dir,
    stable_doc_id,
)
from rag.vector_store import get_vector_store
from rag.line_profiler_instrument import profile_if_enabled, start_profiler, stop_profiler

logger = logging.getLogger(__name__)

SUPPORTED_RAG_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".xlsx",
    ".xls",
    ".csv",
}
INDEX_METADATA_MAX_VALUE_LENGTH = 320
INDEX_METADATA_DROP_KEYS = {"native_catalog", "style_catalog", "font_catalog"}


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

    def _get_text_embedding(self, text: str) -> Embedding:
        retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = self._client.embeddings.create(input=[text], model=self.model_name)
                return response.data[0].embedding
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


class DocumentRegistry:
    def __init__(self, persist_dir: str) -> None:
        self.path = os.path.join(persist_dir, "doc_registry.json")
        self.cache: Optional[Set[str]] = None
        self.last_load_time = 0.0
        self.cache_ttl = 300

    def _lock_shared(self, handle) -> None:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_SH)

    def _lock_exclusive(self, handle) -> None:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)

    def _unlock(self, handle) -> None:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_UN)

    def _load_with_lock(self) -> Set[str]:
        now = time.time()
        if self.cache and (now - self.last_load_time) < self.cache_ttl:
            return self.cache
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as handle:
            self._lock_shared(handle)
            handle.seek(0)
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = []
            finally:
                self._unlock(handle)
        self.cache = set(data)
        self.last_load_time = now
        return self.cache

    def _save_with_lock(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            self._lock_exclusive(handle)
            json.dump(sorted(self.cache or set()), handle)
            self._unlock(handle)

    def get_existing_ids(self) -> Set[str]:
        if self.cache is None:
            return self._load_with_lock()
        return self.cache

    def set_all(self, doc_ids: Set[str]) -> None:
        self.cache = set(doc_ids)
        self._save_with_lock()

    def add_ids(self, doc_ids: Set[str]) -> None:
        if self.cache is None:
            self._load_with_lock()
        if self.cache is None:
            self.cache = set()
        self.cache.update(doc_ids)
        self._save_with_lock()


class RAGEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        api_base = config.settings.OPENAI_API_URL
        self.llm = OpenAI(
            model=config.settings.LLM_MODEL,
            api_key=config.settings.OPENAI_API_KEY,
            api_base=api_base,
        )
        if _should_use_openai_embedding(config.settings.EMBED_MODEL, api_base):
            try:
                self.embed_model = OpenAIEmbedding(
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
                self.embed_model = OpenAICompatibleEmbedding(
                    model_name=config.settings.EMBED_MODEL,
                    api_key=config.settings.OPENAI_API_KEY,
                    api_base=api_base,
                )
        else:
            self.embed_model = OpenAICompatibleEmbedding(
                model_name=config.settings.EMBED_MODEL,
                api_key=config.settings.OPENAI_API_KEY,
                api_base=api_base,
            )
        self.embed_dim = self._get_embedding_dim()
        self.index = None
        self.query_engine = None
        self._rerank_processor: Optional[SentenceTransformerRerank] = None
        self._rerank_model_key: Optional[str] = None
        self._rerank_device_key: Optional[str] = None
        self.cache = self._init_cache()
        self._active_persist_dir = config.get_rag_persist_dir()
        self.doc_registry = DocumentRegistry(self._active_persist_dir)
        # Optional line-by-line profiler (requires `line_profiler` package).
        if getattr(config.settings, "ENABLE_LINE_PROFILE", False):
            prof_path = getattr(config.settings, "LINE_PROFILE_OUTPUT", None)
            started = start_profiler(prof_path)
            if started:
                logger.info("Line profiler started, output=%s", started)
                try:
                    atexit.register(stop_profiler)
                except Exception:
                    pass
            else:
                logger.warning("Line profiler requested but line_profiler is unavailable")

    def _resolve_local_rerank_model(self, model_name: str) -> str:
        if os.path.isdir(model_name):
            return model_name
        local_root = config.settings.RERANK_LOCAL_DIR.strip()
        if not local_root:
            if config.settings.OFFLINE_ONLY:
                raise RAGError(
                    "OFFLINE_ONLY is enabled but RERANK_LOCAL_DIR is empty"
                )
            return model_name
        repo_name = model_name.replace("/", "__")
        local_dir = os.path.abspath(os.path.join(local_root, repo_name))
        if os.path.isdir(local_dir):
            return local_dir
        if config.settings.OFFLINE_ONLY:
            raise RAGError(
                f"Rerank model not found locally: {local_dir}. "
                "Disable OFFLINE_ONLY or pre-download the model."
            )
        os.makedirs(local_dir, exist_ok=True)
        logger.info("Downloading rerank model %s to %s", model_name, local_dir)
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        return local_dir

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and ("cuda" in message or "accelerator" in message)

    @staticmethod
    def _is_embedding_backend_failure(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "model failed to load" in message
            or "resource limitations" in message
            or "openai.internalservererror" in message
            or "error code: 500" in message
        )

    @staticmethod
    def _is_readonly_db_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "readonly database" in message
            or "read-only database" in message
            or "attempt to write a readonly database" in message
        )

    @staticmethod
    def _clear_chroma_system_cache() -> None:
        try:
            from chromadb.api.client import SharedSystemClient  # type: ignore

            SharedSystemClient.clear_system_cache()
        except Exception as exc:
            logger.debug("Failed to clear Chroma shared system cache: %s", exc)

    def invalidate_runtime_state(self, *, clear_chroma_cache: bool = False) -> None:
        self.index = None
        self.query_engine = None
        if clear_chroma_cache and config.settings.VECTOR_STORE_TYPE == "chroma":
            self._clear_chroma_system_cache()

    def _reset_persisted_index_artifacts(self, persist_dir: str) -> None:
        if config.settings.VECTOR_STORE_TYPE == "chroma":
            self._clear_chroma_system_cache()
        removable_paths = [
            os.path.join(persist_dir, "chroma"),
            os.path.join(persist_dir, "faiss.index"),
            os.path.join(persist_dir, "docstore.json"),
            os.path.join(persist_dir, "index_store.json"),
            os.path.join(persist_dir, "graph_store.json"),
            os.path.join(persist_dir, "image__vector_store.json"),
        ]
        for path in removable_paths:
            if not os.path.exists(path):
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as reset_exc:
                logger.warning("Failed to remove stale index artifact %s: %s", path, reset_exc)

    def _build_sentence_reranker(
        self,
        model: str,
        *,
        device: Optional[str] = None,
    ) -> SentenceTransformerRerank:
        kwargs: Dict[str, Any] = {
            "model": model,
            "top_n": config.settings.RERANK_TOP_N,
        }
        if device:
            kwargs["device"] = device
        try:
            return SentenceTransformerRerank(**kwargs)
        except TypeError:
            if "device" in kwargs:
                kwargs.pop("device", None)
                return SentenceTransformerRerank(**kwargs)
            raise

    def _build_sentence_reranker_cpu_safe(self, model: str) -> SentenceTransformerRerank:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            return SentenceTransformerRerank(
                model=model,
                top_n=config.settings.RERANK_TOP_N,
            )
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def _get_sentence_reranker(self) -> SentenceTransformerRerank:
        model_name = config.settings.RERANK_MODEL
        device_pref = config.settings.RERANK_DEVICE
        if (
            self._rerank_processor is not None
            and self._rerank_model_key == model_name
            and self._rerank_device_key == device_pref
        ):
            return self._rerank_processor
        if config.settings.OFFLINE_ONLY:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        resolved_model = model_name
        try:
            resolved_model = self._resolve_local_rerank_model(model_name)
        except Exception as exc:
            logger.warning(
                "Failed to prepare local rerank model %s, fallback to direct load: %s",
                model_name,
                exc,
            )
        target_device: Optional[str]
        if device_pref == "auto":
            target_device = None
        else:
            target_device = device_pref

        try:
            self._rerank_processor = self._build_sentence_reranker(
                resolved_model,
                device=target_device,
            )
        except Exception as exc:
            if device_pref == "cpu":
                logger.warning(
                    "Reranker device=cpu direct init failed, retrying CPU-safe mode: %s",
                    exc,
                )
                self._rerank_processor = self._build_sentence_reranker_cpu_safe(resolved_model)
            elif self._is_cuda_oom_error(exc):
                logger.warning(
                    "CUDA OOM while loading reranker (%s), fallback to CPU",
                    model_name,
                )
                try:
                    self._rerank_processor = self._build_sentence_reranker(
                        resolved_model,
                        device="cpu",
                    )
                except Exception:
                    self._rerank_processor = self._build_sentence_reranker_cpu_safe(resolved_model)
            else:
                raise
        self._rerank_model_key = model_name
        self._rerank_device_key = device_pref
        return self._rerank_processor

    def _ensure_db_context(self) -> None:
        current_persist_dir = config.get_rag_persist_dir()
        if current_persist_dir == self._active_persist_dir:
            return
        self._active_persist_dir = current_persist_dir
        self.invalidate_runtime_state(clear_chroma_cache=(config.settings.VECTOR_STORE_TYPE == "chroma"))
        self.doc_registry = DocumentRegistry(current_persist_dir)
        logger.info("Switched RAG context to %s", current_persist_dir)

    def _get_embedding_dim(self) -> int:
        if config.settings.EMBED_DIM:
            return config.settings.EMBED_DIM
        sample_embed = self.embed_model.get_text_embedding("test")
        return len(sample_embed)

    def _init_cache(self):
        if config.settings.CACHE_TYPE == "redis" and config.settings.REDIS_URL:
            try:
                redis_client = Redis.from_url(
                    config.settings.REDIS_URL, decode_responses=True
                )
                redis_client.ping()
                return redis_client
            except RedisError as exc:
                logger.warning("Redis unavailable, falling back to memory: %s", exc)
        return TTLCache(
            maxsize=config.settings.MEMORY_CACHE_MAXSIZE,
            ttl=config.settings.CACHE_TTL,
        )

    def _cache_key(self, question: str) -> str:
        normalized = " ".join(question.lower().split())
        db_scope = config.get_rag_persist_dir()
        payload = f"{db_scope}::{normalized}"
        return f"rag:v2:{hashlib.md5(payload.encode()).hexdigest()}"

    @staticmethod
    def _sanitize_metadata_for_indexing(metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if key in INDEX_METADATA_DROP_KEYS:
                continue
            if isinstance(value, str):
                sanitized[key] = value[:INDEX_METADATA_MAX_VALUE_LENGTH]
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                sanitized[key] = value
                continue
            text = str(value)
            if text:
                sanitized[key] = text[:INDEX_METADATA_MAX_VALUE_LENGTH]
        return sanitized

    def _sanitize_documents_for_indexing(self, docs: List[Document]) -> List[Document]:
        sanitized_docs: List[Document] = []
        for doc in docs:
            sanitized_docs.append(
                Document(
                    text=doc.text,
                    metadata=self._sanitize_metadata_for_indexing(dict(doc.metadata or {})),
                    doc_id=doc.doc_id,
                )
            )
        return sanitized_docs

    def _get_selected_db_names(self, db_names: Optional[List[str]] = None) -> List[str]:
        names: List[str] = []
        source_names = db_names if db_names is not None else config.settings.RAG_DB_NAMES
        for item in source_names:
            value = (item or "").strip()
            if value and value not in names:
                names.append(value)
        if db_names is not None:
            return names
        fallback = (config.settings.RAG_DB_NAME or "").strip()
        if not names and fallback:
            names.append(fallback)
        return names

    def _query_multi_db(self, question: str, db_names: List[str]) -> Dict[str, Any]:
        db_scope = ",".join(db_names)
        cache_key = self._cache_key(f"[multi:{db_scope}] {question}")
        cached = self._get_cached(cache_key)
        if cached:
            rag_cache_hit_ratio.labels(hit="true").inc()
            rag_query_count.labels(success="true").inc()
            return cached

        rag_cache_hit_ratio.labels(hit="false").inc()
        original_db_name = config.settings.RAG_DB_NAME
        original_db_names = list(config.settings.RAG_DB_NAMES)
        results: List[Dict[str, Any]] = []

        try:
            for db_name in db_names:
                config.settings = config.settings.update(RAG_DB_NAME=db_name, RAG_DB_NAMES=[])
                single = self.query(question)
                for source in single.get("sources", []) or []:
                    metadata = source.get("metadata") or {}
                    if isinstance(metadata, dict):
                        metadata.setdefault("rag_db", db_name)
                        source["metadata"] = metadata
                results.append(single)
        finally:
            config.settings = config.settings.update(
                RAG_DB_NAME=original_db_name,
                RAG_DB_NAMES=original_db_names,
            )

        all_sources: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        for result in results:
            sources = result.get("sources", []) or []
            all_sources.extend(sources)
            top_score = sources[0].get("score") if sources else None
            numeric_score = top_score if isinstance(top_score, (int, float)) else float("-inf")
            if numeric_score > best_score:
                best_score = numeric_score
                best_result = result

        def _source_score(item: Dict[str, Any]) -> float:
            score = item.get("score")
            return float(score) if isinstance(score, (int, float)) else float("-inf")

        all_sources.sort(key=_source_score, reverse=True)
        merged_sources = all_sources[: max(config.settings.SIMILARITY_TOP_K, 1)]
        answer = (best_result or {}).get("answer", "")
        result = {
            "answer": answer,
            "sources": merged_sources,
            "timestamp": datetime.now().isoformat(),
        }
        self._set_cache(cache_key, result)
        rag_query_count.labels(success="true").inc()
        return result

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        if isinstance(self.cache, Redis):
            cached = self.cache.get(key)
            if cached:
                return json.loads(cast(str, cached))
            return None
        return self.cache.get(key)

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        if isinstance(self.cache, Redis):
            self.cache.setex(key, config.settings.CACHE_TTL, json.dumps(value))
        else:
            self.cache[key] = value

    def _clear_cache(self) -> None:
        if isinstance(self.cache, Redis):
            try:
                for key in self.cache.scan_iter(match="rag:v2:*"):
                    self.cache.delete(key)
            except Exception as exc:
                logger.warning("Failed to clear redis RAG cache: %s", exc)
            return
        try:
            self.cache.clear()
        except Exception as exc:
            logger.warning("Failed to clear in-memory RAG cache: %s", exc)

    def clear_query_cache(self) -> None:
        self._clear_cache()

    def _load_sample_documents(self, limit: int) -> List[Document]:
        selected_db = (config.settings.RAG_DB_NAME or "").strip()
        if selected_db:
            selected_docs = load_chunk_documents_from_persist_dir(
                config.get_rag_persist_dir(),
                SUPPORTED_RAG_EXTENSIONS,
            )
            if selected_docs:
                return selected_docs[:limit]
        reader = SimpleDirectoryReader(
            input_dir=config.settings.DATA_DIR,
            recursive=True,
            filename_as_id=True,
            num_files_limit=limit,
        )
        return reader.load_data()

    @profile_if_enabled
    def _build_faiss_index_with_training(self, embed_dim: int) -> faiss.Index:
        if "IVF" in config.settings.FAISS_INDEX_TYPE:
            nlist = 100
            index_type = config.settings.FAISS_INDEX_TYPE
            if index_type.startswith("IVF"):
                digits = "".join(ch for ch in index_type if ch.isdigit())
                if digits:
                    nlist = int(digits)
            quantizer = faiss.IndexFlatL2(embed_dim)
            index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist)

            sample_docs = self._load_sample_documents(config.settings.SAMPLE_FOR_TRAINING)
            if len(sample_docs) < nlist:
                logger.warning(
                    "IVF training samples (%s) below clusters (%s), using Flat index",
                    len(sample_docs),
                    nlist,
                )
                return faiss.IndexFlatL2(embed_dim)


            sample_embeds = np.array(
                [self.embed_model.get_text_embedding(doc.text) for doc in sample_docs],
                dtype="float32",
            )
            cast(Any, index).train(sample_embeds)
            logger.info("IVF index trained with %s samples", len(sample_docs))
            return index
        return faiss.IndexFlatL2(embed_dim)

    @profile_if_enabled
    def _load_all_documents(self) -> List[Document]:
        docs = load_chunk_documents_from_data_dir(
            config.settings.DATA_DIR,
            SUPPORTED_RAG_EXTENSIONS,
        )
        logger.info("Loaded %s documents", len(docs))
        return docs

    def _build_index(self) -> None:
        docs = self._load_all_documents()
        if not docs:
            raise ValueError(f"No documents found in {config.settings.DATA_DIR}")
        self._build_index_from_docs(docs)

    def _build_index_for_selected_db(self) -> bool:
        docs = load_chunk_documents_from_persist_dir(
            config.get_rag_persist_dir(),
            SUPPORTED_RAG_EXTENSIONS,
        )
        if not docs:
            self.index = None
            self.query_engine = None
            self._save_doc_registry(set())
            return False
        self._build_index_from_docs(docs)
        return True

    @profile_if_enabled
    def _build_index_from_docs(self, docs: List[Document]) -> None:
        docs = self._sanitize_documents_for_indexing(docs)
        parser = SentenceSplitter.from_defaults(
            chunk_size=config.settings.CHUNK_SIZE,
            chunk_overlap=config.settings.CHUNK_OVERLAP,
        )
        try:
            nodes = parser.get_nodes_from_documents(docs)
        except RecursionError:
            logger.warning("Sentence splitter recursion overflow, switching to token splitter")
            token_parser = TokenTextSplitter.from_defaults(
                chunk_size=config.settings.CHUNK_SIZE,
                chunk_overlap=config.settings.CHUNK_OVERLAP,
                separator=" ",
                backup_separators=["\n", "。", ".", "，", ",", "；", ";", "：", ":"],
            )
            nodes = token_parser.get_nodes_from_documents(docs)
        if not nodes:
            raise ValueError("Document parsing produced no nodes")

        persist_dir = config.get_rag_persist_dir()

        def _build_once() -> None:
            vector_store = get_vector_store(
                store_type=config.settings.VECTOR_STORE_TYPE,
                persist_dir=persist_dir,
                embed_dim=self.embed_dim,
                embed_model=self.embed_model,
                index_builder=self._build_faiss_index_with_training,
            )
            if (
                getattr(vector_store, "fallback", False)
                and config.settings.VECTOR_STORE_FALLBACK_WARNING
            ):
                logger.warning("Vector store fallback active; performance may be degraded")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )

        try:
            _build_once()
        except Exception as exc:
            if config.settings.VECTOR_STORE_TYPE == "chroma" and self._is_readonly_db_error(exc):
                logger.warning(
                    "Detected read-only Chroma storage under %s, resetting persisted index artifacts and retrying once",
                    persist_dir,
                )
                self.invalidate_runtime_state(clear_chroma_cache=True)
                self._reset_persisted_index_artifacts(persist_dir)
                _build_once()
            else:
                raise
        if self.index is None:
            raise RAGError("RAG index build failed")
        self.index.storage_context.persist(persist_dir=config.get_rag_persist_dir())
        self._persist_faiss_index()
        self._save_doc_registry({doc.doc_id for doc in docs})
        self._clear_cache()
        logger.info("Built and persisted index")

    def _persist_faiss_index(self) -> None:
        if not self.index:
            return
        vector_store = self.index.storage_context.vector_store
        if isinstance(vector_store, FaissVectorStore):
            faiss_index = getattr(vector_store, "faiss_index", None) or getattr(
                vector_store, "_faiss_index", None
            )
            if faiss_index is None:
                return
            faiss_path = os.path.join(config.get_rag_persist_dir(), "faiss.index")
            faiss.write_index(faiss_index, faiss_path)

    @profile_if_enabled
    def _load_or_build_index(self) -> None:
        self._ensure_db_context()
        persist_dir = config.get_rag_persist_dir()
        os.makedirs(persist_dir, exist_ok=True)
        selected_db_mode = bool((config.settings.RAG_DB_NAME or "").strip())

        try:
            vector_store = get_vector_store(
                store_type=config.settings.VECTOR_STORE_TYPE,
                persist_dir=persist_dir,
                embed_dim=self.embed_dim,
                embed_model=self.embed_model,
                index_builder=self._build_faiss_index_with_training,
            )
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=vector_store,
            )
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model,
            )
            logger.info("Loaded index from storage")
        except FileNotFoundError:
            logger.info("No persisted index found, building a new one")
            self.invalidate_runtime_state(clear_chroma_cache=(config.settings.VECTOR_STORE_TYPE == "chroma"))
            if selected_db_mode:
                built = self._build_index_for_selected_db()
                if not built:
                    logger.info("Selected DB has no docs/index: %s", config.settings.RAG_DB_NAME)
            else:
                self._build_index()
        except Exception as exc:
            logger.error("Failed to load index, rebuilding: %s", exc)
            if config.settings.VECTOR_STORE_TYPE == "chroma" and self._is_readonly_db_error(exc):
                self.invalidate_runtime_state(clear_chroma_cache=True)
                self._reset_persisted_index_artifacts(persist_dir)
            if selected_db_mode:
                built = self._build_index_for_selected_db()
                if not built:
                    logger.info("Selected DB has no docs/index after rebuild: %s", config.settings.RAG_DB_NAME)
            else:
                self._build_index()

    @profile_if_enabled
    def _add_documents_incremental(self, new_docs: List[Document]) -> None:
        self._ensure_db_context()
        if self.index is None:
            self._load_or_build_index()
        if self.index is None:
            raise RAGError("RAG index is unavailable")

        existing_ids = self._get_existing_doc_ids()

        parser = SentenceSplitter.from_defaults(
            chunk_size=config.settings.CHUNK_SIZE,
            chunk_overlap=config.settings.CHUNK_OVERLAP,
        )
        token_parser = TokenTextSplitter.from_defaults(
            chunk_size=config.settings.CHUNK_SIZE,
            chunk_overlap=config.settings.CHUNK_OVERLAP,
            separator=" ",
            backup_separators=["\n", "。", ".", "，", ",", "；", ";", "：", ":"],
        )
        inserted = 0
        inserted_ids: Set[str] = set()
        for doc in new_docs:
            doc_id = doc.doc_id or stable_doc_id(doc)
            if doc_id in existing_ids:
                continue
            insert_doc = Document(
                text=doc.text,
                metadata=self._sanitize_metadata_for_indexing(dict(doc.metadata or {})),
                doc_id=doc_id,
            )
            try:
                nodes = parser.get_nodes_from_documents([insert_doc])
            except RecursionError:
                logger.warning(
                    "Sentence splitter recursion overflow for doc %s, fallback to token splitter",
                    doc_id,
                )
                nodes = token_parser.get_nodes_from_documents([insert_doc])
            self.index.insert_nodes(nodes)
            inserted += 1
            inserted_ids.add(doc_id)
        if inserted:
            self.index.storage_context.persist(persist_dir=config.get_rag_persist_dir())
            self._persist_faiss_index()
            self.doc_registry.add_ids(inserted_ids)
            self._clear_cache()
        logger.info("Incrementally added %s documents", inserted)

    def _get_existing_doc_ids(self) -> Set[str]:
        if self.index is None:
            return set()
        vector_store = self.index.storage_context.vector_store
        if config.settings.VECTOR_STORE_TYPE == "chroma":
            collection = getattr(vector_store, "_collection", None)
            if collection:
                data = collection.get(include=["metadatas"])
                return {
                    meta.get("doc_id")
                    for meta in data.get("metadatas", [])
                    if meta and meta.get("doc_id")
                }
        return self.doc_registry.get_existing_ids()

    def _save_doc_registry(self, doc_ids: Set[str]) -> None:
        try:
            self.doc_registry.set_all(doc_ids)
        except Exception as exc:
            logger.warning("Failed to write doc registry: %s", exc)

    def add_documents(self, docs: List[Document]) -> None:
        self._ensure_db_context()
        self._add_documents_incremental(docs)

    def add_documents_from_paths(self, paths: List[str]) -> int:
        self._ensure_db_context()
        docs = load_chunk_documents_from_paths(paths, SUPPORTED_RAG_EXTENSIONS)
        if not docs:
            return 0
        self._add_documents_incremental(docs)
        return len(docs)

    def rebuild_index_from_paths(self, paths: List[str]) -> int:
        self._ensure_db_context()
        self.index = None
        self.query_engine = None
        if not paths:
            self._save_doc_registry(set())
            return 0
        docs = load_chunk_documents_from_paths(paths, SUPPORTED_RAG_EXTENSIONS)
        if not docs:
            self._save_doc_registry(set())
            return 0
        self._build_index_from_docs(docs)
        return len(docs)

    def rebuild_index(self) -> int:
        self._ensure_db_context()
        self.index = None
        self.query_engine = None
        docs = self._load_all_documents()
        if not docs:
            raise ValueError(f"No documents found in {config.settings.DATA_DIR}")
        self._build_index_from_docs(docs)
        return len(docs)

    def get_query_engine(self) -> RetrieverQueryEngine:
        self._ensure_db_context()
        if self.query_engine is None:
            self._load_or_build_index()
            if self.index is None:
                raise RAGError("RAG index is unavailable")

            retriever = VectorIndexRetriever(
                index=cast(VectorStoreIndex, self.index),
                similarity_top_k=config.settings.SIMILARITY_TOP_K,
            )

            node_postprocessors = []
            if config.settings.ENABLE_RERANK:
                if config.settings.RERANK_MODEL.startswith("cross-encoder"):
                    rerank = self._get_sentence_reranker()
                else:
                    rerank = LLMRerank(llm=self.llm, top_n=config.settings.RERANK_TOP_N)
                node_postprocessors.append(rerank)

            try:
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    llm=self.llm,
                    node_postprocessors=node_postprocessors,
                )
            except Exception as exc:
                logger.error("RetrieverQueryEngine initialization failed: %s", exc)
                raise RAGError(f"Query engine init failed: {exc}") from exc
        return self.query_engine

    @profile_if_enabled
    def query(self, question: str, db_names: Optional[List[str]] = None) -> Dict[str, Any]:
        selected_db_names = self._get_selected_db_names(db_names)
        if not selected_db_names:
            return {
                "answer": "No RAG database selected.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
            }
        if len(selected_db_names) > 1:
            return self._query_multi_db(question, selected_db_names)

        original_db_name = config.settings.RAG_DB_NAME
        original_db_names = list(config.settings.RAG_DB_NAMES)
        if selected_db_names:
            config.settings = config.settings.update(
                RAG_DB_NAME=selected_db_names[0],
                RAG_DB_NAMES=selected_db_names,
            )

        try:
            cache_key = self._cache_key(question)
            cached = self._get_cached(cache_key)
            if cached:
                rag_cache_hit_ratio.labels(hit="true").inc()
                rag_query_count.labels(success="true").inc()
                return cached

            rag_cache_hit_ratio.labels(hit="false").inc()
            with rag_query_latency.time():
                try:
                    self._ensure_db_context()
                    self._load_or_build_index()
                    if self.index is None:
                        raise RAGError("RAG index is unavailable")
                    retrieval_top_k = min(
                        max(config.settings.SIMILARITY_TOP_K * 4, config.settings.SIMILARITY_TOP_K),
                        40,
                    )
                    retriever = VectorIndexRetriever(
                        index=cast(VectorStoreIndex, self.index),
                        similarity_top_k=retrieval_top_k,
                    )
                    source_nodes = retriever.retrieve(question)
                    source_nodes = self._route_nodes_by_structure(source_nodes)
                    if config.settings.ENABLE_RERANK:
                        if config.settings.RERANK_MODEL.startswith("cross-encoder"):
                            rerank = self._get_sentence_reranker()
                            source_nodes = rerank.postprocess_nodes(
                                source_nodes,
                                query_bundle=QueryBundle(question),
                            )
                        else:
                            rerank = LLMRerank(
                                llm=self.llm,
                                top_n=config.settings.RERANK_TOP_N,
                            )
                            source_nodes = rerank.postprocess_nodes(
                                source_nodes,
                                query_bundle=QueryBundle(question),
                            )
                    source_nodes = source_nodes[: config.settings.SIMILARITY_TOP_K]
                except RateLimitError as exc:
                    rag_query_count.labels(success="false").inc()
                    raise APIThrottlingError("OpenAI rate limit") from exc
                except (APITimeoutError, APIConnectionError) as exc:
                    rag_query_count.labels(success="false").inc()
                    raise QueryTimeoutError("OpenAI request timeout") from exc
                except Exception as exc:
                    if self._is_embedding_backend_failure(exc):
                        logger.warning("Embedding backend unavailable, fallback to lexical retrieval: %s", exc)
                        # Pull a larger set of candidates via lexical scan (no embeddings)
                        fallback = self.regex_retrieve(
                            regex=None,
                            chunk=None,
                            limit=min(max(config.settings.SIMILARITY_TOP_K * 6, 24), 200),
                        )
                        fallback_results = fallback.get("results", []) if isinstance(fallback, dict) else []
                        if fallback_results:
                            # Score candidates by simple token overlap with the question
                            q_tokens = set(re.findall(r"\w+", (question or "").lower()))
                            scored: List[Dict[str, Any]] = []
                            for item in fallback_results:
                                text = (item.get("text") or "")[:4000]
                                text_lower = text.lower()
                                text_tokens = set(re.findall(r"\w+", text_lower))
                                overlap = 0.0
                                if q_tokens:
                                    overlap = len(q_tokens & text_tokens) / float(len(q_tokens))
                                combined_score = (item.get("score") or 0.0) + overlap
                                scored.append({"score": combined_score, "item": item})

                            scored.sort(key=lambda r: r["score"], reverse=True)
                            top_items = [s["item"] for s in scored[: config.settings.SIMILARITY_TOP_K]]
                            sources = [
                                {
                                    "text": str(item.get("text", ""))[:1200],
                                    "score": item.get("score"),
                                    "metadata": item.get("metadata") or {},
                                }
                                for item in top_items
                            ]
                            answer = sources[0]["text"] if sources else "No relevant knowledge found."
                            result = {
                                "answer": answer,
                                "sources": sources,
                                "timestamp": datetime.now().isoformat(),
                            }
                            self._set_cache(cache_key, result)
                            rag_query_count.labels(success="true").inc()
                            return result
                    rag_query_count.labels(success="false").inc()
                    raise RAGError(f"Query failed: {exc}") from exc

            source_text_limit = 1200
            sources = [
                {
                    "text": node.node.get_content()[:source_text_limit],
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                for node in source_nodes
            ]
            answer = ""
            if sources and config.settings.RAG_SYNTHESIZE_ANSWER:
                try:
                    client = OpenAIClient(
                        api_key=config.settings.OPENAI_API_KEY,
                        base_url=config.settings.OPENAI_API_URL,
                    )
                    context = "\n\n".join(
                        f"[{idx + 1}] {item['text']}" for idx, item in enumerate(sources[:5])
                    )
                    completion = client.chat.completions.create(
                        model=config.settings.LLM_MODEL,
                        temperature=config.settings.TEMPERATURE,
                        timeout=config.settings.RAG_SYNTHESIS_TIMEOUT,
                        messages=[
                            {
                                "role": "system",
                                "content": "Answer based only on provided context. If insufficient, say so briefly.",
                            },
                            {
                                "role": "user",
                                "content": f"Question: {question}\n\nContext:\n{context}",
                            },
                        ],
                    )
                    answer = (
                        (completion.choices[0].message.content or "").strip()
                        if completion.choices
                        else ""
                    )
                except Exception as exc:
                    logger.warning("RAG answer synthesis fallback: %s", exc)
                    answer = sources[0]["text"]
            elif sources:
                answer = sources[0]["text"]
            if not answer:
                answer = "No relevant knowledge found."

            result = {
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
            }
            self._set_cache(cache_key, result)
            rag_query_count.labels(success="true").inc()
            return result
        finally:
            if selected_db_names:
                config.settings = config.settings.update(
                    RAG_DB_NAME=original_db_name,
                    RAG_DB_NAMES=original_db_names,
                )

    def _route_nodes_by_structure(self, source_nodes: List[Any]) -> List[Any]:
        if not source_nodes:
            return source_nodes

        section_scores: Dict[str, float] = {}
        for node_with_score in source_nodes:
            metadata = getattr(node_with_score.node, "metadata", {}) or {}
            section_key = metadata.get("section_id") or metadata.get("section_path")
            if not section_key:
                page = metadata.get("section_start_page") or metadata.get("page")
                if page is not None:
                    section_key = f"page:{page}"
            if not section_key:
                continue
            score = node_with_score.score if isinstance(node_with_score.score, (int, float)) else 0.0
            section_scores[section_key] = section_scores.get(section_key, 0.0) + float(score)

        if not section_scores:
            return source_nodes

        top_sections = {
            item[0]
            for item in sorted(section_scores.items(), key=lambda pair: pair[1], reverse=True)[:3]
        }
        routed: List[Any] = []
        for node_with_score in source_nodes:
            metadata = getattr(node_with_score.node, "metadata", {}) or {}
            section_key = metadata.get("section_id") or metadata.get("section_path")
            if not section_key:
                page = metadata.get("section_start_page") or metadata.get("page")
                if page is not None:
                    section_key = f"page:{page}"
            if section_key in top_sections:
                routed.append(node_with_score)

        if len(routed) >= config.settings.SIMILARITY_TOP_K:
            return routed
        return source_nodes

    def _empty_retrieve_result(
        self,
        *,
        query_text: str,
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        regex: Optional[str],
        chunk: Optional[str],
        doc_name: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "query": query_text,
            "filters": {
                "doc_name": doc_name,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "regex": regex,
                "chunk": chunk,
            },
            "count": 0,
            "results": [],
            "timestamp": datetime.now().isoformat(),
        }

    def _retrieve_documents(
        self,
        *,
        query_text: str,
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        regex: Optional[str],
        chunk: Optional[str],
        doc_name: Optional[str],
        limit: int,
        use_vector: bool,
    ) -> Dict[str, Any]:
        selected_db_names = self._get_selected_db_names()
        if not selected_db_names:
            return self._empty_retrieve_result(
                query_text=query_text,
                section=section,
                page_start=page_start,
                page_end=page_end,
                regex=regex,
                chunk=chunk,
                doc_name=doc_name,
            )
        self._ensure_db_context()
        limit = max(1, min(int(limit), 50))
        rag_docs = load_rag_documents_from_persist_dir(
            config.get_rag_persist_dir(),
            SUPPORTED_RAG_EXTENSIONS,
        )
        if not rag_docs:
            return self._empty_retrieve_result(
                query_text=query_text,
                section=section,
                page_start=page_start,
                page_end=page_end,
                regex=regex,
                chunk=chunk,
                doc_name=doc_name,
            )

        regex_pattern = (regex or "").strip()
        compiled = None
        if regex_pattern:
            try:
                compiled = re.compile(regex_pattern, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                raise RAGError(f"Invalid regex: {exc}") from exc

        vector_section_scores: Dict[str, Dict[str, float]] = {}
        if use_vector:
            if not query_text:
                return self._empty_retrieve_result(
                    query_text=query_text,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    regex=regex,
                    chunk=chunk,
                    doc_name=doc_name,
                )
            self._load_or_build_index()
            if self.index is None:
                raise RAGError("RAG index is unavailable")
            retriever = VectorIndexRetriever(
                index=cast(VectorStoreIndex, self.index),
                similarity_top_k=min(max(limit * 6, 24), 80),
            )
            source_nodes = retriever.retrieve(query_text)
            for item in source_nodes:
                metadata = getattr(item.node, "metadata", {}) or {}
                candidate_doc_name = str(metadata.get("doc_name") or "").strip()
                section_id = str(metadata.get("section_id") or "").strip()
                if not candidate_doc_name or not section_id:
                    continue
                score = float(item.score) if isinstance(item.score, (int, float)) else 0.0
                score_by_section = vector_section_scores.setdefault(candidate_doc_name, {})
                previous = score_by_section.get(section_id)
                score_by_section[section_id] = score if previous is None else max(previous, score)

        matched_doc_names: Optional[Set[str]] = None
        if doc_name is not None:
            available_doc_names = {
                rag_doc.doc_name
                for rag_doc in rag_docs
                if isinstance(rag_doc.doc_name, str) and rag_doc.doc_name.strip()
            }
            matched_doc_names = RAG_DB_Document.resolve_doc_name_matches(
                doc_name,
                available_doc_names,
                data_dir=str(config.settings.DATA_DIR or ""),
            )
            if not matched_doc_names:
                return self._empty_retrieve_result(
                    query_text=query_text,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    regex=regex,
                    chunk=chunk,
                    doc_name=doc_name,
                )

        scoped_docs = [
            rag_doc
            for rag_doc in rag_docs
            if matched_doc_names is None or rag_doc.doc_name in matched_doc_names
        ]

        results: List[Dict[str, Any]] = []
        for rag_doc in scoped_docs:
            if use_vector:
                section_scores = vector_section_scores.get(rag_doc.doc_name) or {}
                if not section_scores:
                    continue
                results.extend(
                    rag_doc.retrieve_by_vector(
                        query_text=query_text,
                        section_scores=section_scores,
                        compiled_regex=compiled,
                        section=section,
                        page_start=page_start,
                        page_end=page_end,
                        chunk=chunk,
                    )
                )
                continue
            results.extend(
                rag_doc.retrieve_by_regex(
                    compiled_regex=compiled,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    chunk=chunk,
                )
            )

        results.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        payload = self._empty_retrieve_result(
            query_text=query_text,
            section=section,
            page_start=page_start,
            page_end=page_end,
            regex=regex,
            chunk=chunk,
            doc_name=doc_name,
        )
        payload["count"] = len(results)
        payload["results"] = results[:limit]
        return payload

    @profile_if_enabled
    def regex_retrieve(
        self,
        regex: Optional[str] = None,
        section: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        chunk: Optional[str] = None,
        doc_name: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        return self._retrieve_documents(
            query_text="",
            section=section,
            page_start=page_start,
            page_end=page_end,
            regex=regex,
            chunk=chunk,
            doc_name=doc_name,
            limit=limit,
            use_vector=False,
        )

    @profile_if_enabled
    def vector_retrieve(
        self,
        query: str,
        section: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        regex: Optional[str] = None,
        chunk: Optional[str] = None,
        doc_name: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        return self._retrieve_documents(
            query_text=(query or "").strip(),
            section=section,
            page_start=page_start,
            page_end=page_end,
            regex=regex,
            chunk=chunk,
            doc_name=doc_name,
            limit=limit,
            use_vector=True,
        )
    

    def list_documents(self) -> List[Dict[str, Any]]:
        """返回当前数据库中的所有文档标题（文件名）及其估计页数。"""
        self._ensure_db_context()
        rag_docs = load_rag_documents_from_persist_dir(
            config.get_rag_persist_dir(),
            SUPPORTED_RAG_EXTENSIONS,
        )
        by_doc_name: Dict[str, Dict[str, Any]] = {}
        for rag_doc in rag_docs:
            payload = rag_doc.list_payload()
            doc_name = str(payload.get("doc_name") or "").strip()
            if not doc_name:
                continue
            existing = by_doc_name.get(doc_name)
            if existing is None:
                by_doc_name[doc_name] = payload
                continue
            existing["page_count"] = max(
                int(existing.get("page_count") or 0),
                int(payload.get("page_count") or 0),
            )
            existing["chunk_count"] = max(
                int(existing.get("chunk_count") or 0),
                int(payload.get("chunk_count") or 0),
            )
        return sorted(by_doc_name.values(), key=lambda d: str(d.get("title") or ""))


    def get_document_catalog(self, doc_name: str) -> List[Dict[str, Any]]:
        """返回指定文档的目录：章节路径及起始页码。"""
        self._ensure_db_context()
        rag_docs = load_rag_documents_from_persist_dir(
            config.get_rag_persist_dir(),
            SUPPORTED_RAG_EXTENSIONS,
        )
        available_doc_names = {
            rag_doc.doc_name
            for rag_doc in rag_docs
            if isinstance(rag_doc.doc_name, str) and rag_doc.doc_name.strip()
        }
        matched_doc_names = RAG_DB_Document.resolve_doc_name_matches(
            doc_name,
            available_doc_names,
            data_dir=str(config.settings.DATA_DIR or ""),
        )
        if not matched_doc_names:
            return []

        catalog: Dict[str, Dict[str, Any]] = {}
        for rag_doc in rag_docs:
            if rag_doc.doc_name not in matched_doc_names:
                continue
            for item in rag_doc.catalog_payload():
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                page = int(item.get("page") or 0)
                end_page = max(page, int(item.get("end_page") or page))
                existing = catalog.get(title)
                if existing is None:
                    catalog[title] = {
                        "title": title,
                        "page": page,
                        "end_page": end_page,
                    }
                    continue
                existing["page"] = min(int(existing.get("page") or page), page)
                existing["end_page"] = max(int(existing.get("end_page") or end_page), end_page)

        return sorted(
            catalog.values(),
            key=lambda item: (
                int(item.get("page") or 0),
                int(item.get("end_page") or item.get("page") or 0),
                str(item.get("title") or ""),
            ),
        )