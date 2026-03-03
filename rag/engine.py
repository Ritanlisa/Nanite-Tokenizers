from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import faiss
import numpy as np
from cachetools import TTLCache
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from redis import Redis
from redis.exceptions import RedisError
from openai import APIConnectionError, APITimeoutError, RateLimitError

import config
from exceptions import APIThrottlingError, QueryTimeoutError, RAGError
from monitoring import rag_cache_hit_ratio, rag_query_count, rag_query_latency
from rag.ocr import OCRPDFReader, ocr_enabled
from rag.preprocessor import clean_document
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


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
        self.embed_model = OpenAIEmbedding(
            api_key=config.settings.OPENAI_API_KEY,
            model=config.settings.EMBED_MODEL,
            dimensions=config.settings.EMBED_DIM,
            api_base=api_base,
        )
        self.embed_dim = self._get_embedding_dim()
        self.index = None
        self.query_engine = None
        self.cache = self._init_cache()
        self.doc_registry = DocumentRegistry(config.settings.PERSIST_DIR)

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
        return f"rag:v1:{hashlib.md5(normalized.encode()).hexdigest()}"

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        if isinstance(self.cache, Redis):
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
            return None
        return self.cache.get(key)

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        if isinstance(self.cache, Redis):
            self.cache.setex(key, config.settings.CACHE_TTL, json.dumps(value))
        else:
            self.cache[key] = value

    def _load_sample_documents(self, limit: int) -> List[Document]:
        reader = SimpleDirectoryReader(
            input_dir=config.settings.DATA_DIR,
            recursive=True,
            filename_as_id=True,
            num_files_limit=limit,
        )
        return reader.load_data()

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
            index.train(sample_embeds)
            logger.info("IVF index trained with %s samples", len(sample_docs))
            return index
        return faiss.IndexFlatL2(embed_dim)

    def _load_all_documents(self) -> List[Document]:
        from llama_index.readers.file import DocxReader, PandasExcelReader, PDFReader

        pdf_reader = OCRPDFReader() if ocr_enabled() else PDFReader()
        readers = {
            ".pdf": pdf_reader,
            ".docx": DocxReader(),
            ".xlsx": PandasExcelReader(),
        }
        reader = SimpleDirectoryReader(
            input_dir=config.settings.DATA_DIR,
            recursive=True,
            filename_as_id=True,
            file_extractor=readers,
        )
        docs = reader.load_data()
        for doc in docs:
            if not doc.doc_id:
                doc.doc_id = self._stable_doc_id(doc)
            doc.text = clean_document(doc.text)
        logger.info("Loaded %s documents", len(docs))
        return docs

    def _stable_doc_id(self, doc: Document) -> str:
        source = doc.metadata.get("file_name") or doc.text[:200]
        return hashlib.md5(source.encode()).hexdigest()

    def _build_index(self) -> None:
        docs = self._load_all_documents()
        if not docs:
            raise ValueError(f"No documents found in {config.settings.DATA_DIR}")
        parser = SimpleNodeParser.from_defaults(
            chunk_size=config.settings.CHUNK_SIZE,
            chunk_overlap=config.settings.CHUNK_OVERLAP,
        )
        nodes = parser.get_nodes_from_documents(docs)
        if not nodes:
            raise ValueError("Document parsing produced no nodes")

        vector_store = get_vector_store(
            store_type=config.settings.VECTOR_STORE_TYPE,
            persist_dir=config.settings.PERSIST_DIR,
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
        self.index.storage_context.persist(persist_dir=config.settings.PERSIST_DIR)
        self._persist_faiss_index()
        self._save_doc_registry({doc.doc_id for doc in docs})
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
            faiss_path = os.path.join(config.settings.PERSIST_DIR, "faiss.index")
            faiss.write_index(faiss_index, faiss_path)

    def _load_or_build_index(self) -> None:
        persist_dir = config.settings.PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)

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
            self._build_index()
        except Exception as exc:
            logger.error("Failed to load index, rebuilding: %s", exc)
            self._build_index()

    def _add_documents_incremental(self, new_docs: List[Document]) -> None:
        if self.index is None:
            self._load_or_build_index()

        existing_ids = self._get_existing_doc_ids()

        parser = SimpleNodeParser.from_defaults(
            chunk_size=config.settings.CHUNK_SIZE,
            chunk_overlap=config.settings.CHUNK_OVERLAP,
        )
        inserted = 0
        for doc in new_docs:
            if not doc.doc_id:
                doc.doc_id = self._stable_doc_id(doc)
            if doc.doc_id in existing_ids:
                continue
            nodes = parser.get_nodes_from_documents([doc])
            for node in nodes:
                self.index.insert(node)
            inserted += 1
        if inserted:
            self.index.storage_context.persist(persist_dir=config.settings.PERSIST_DIR)
            self._persist_faiss_index()
            new_ids = {doc.doc_id for doc in new_docs if doc.doc_id}
            self.doc_registry.add_ids(new_ids)
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
        self._add_documents_incremental(docs)

    def get_query_engine(self) -> RetrieverQueryEngine:
        if self.query_engine is None:
            self._load_or_build_index()

            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.settings.SIMILARITY_TOP_K,
            )

            node_postprocessors = []
            if config.settings.ENABLE_RERANK:
                if config.settings.RERANK_MODEL.startswith("cross-encoder"):
                    rerank = SentenceTransformerRerank(
                        model=config.settings.RERANK_MODEL,
                        top_n=config.settings.RERANK_TOP_N,
                    )
                else:
                    rerank = LLMRerank(llm=self.llm, top_n=config.settings.RERANK_TOP_N)
                node_postprocessors.append(rerank)

            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=node_postprocessors,
            )
        return self.query_engine

    def query(self, question: str) -> Dict[str, Any]:
        cache_key = self._cache_key(question)
        cached = self._get_cached(cache_key)
        if cached:
            rag_cache_hit_ratio.labels(hit="true").inc()
            rag_query_count.labels(success="true").inc()
            return cached

        rag_cache_hit_ratio.labels(hit="false").inc()
        query_engine = self.get_query_engine()
        with rag_query_latency.time():
            try:
                response = query_engine.query(question)
            except RateLimitError as exc:
                rag_query_count.labels(success="false").inc()
                raise APIThrottlingError("OpenAI rate limit") from exc
            except (APITimeoutError, APIConnectionError) as exc:
                rag_query_count.labels(success="false").inc()
                raise QueryTimeoutError("OpenAI request timeout") from exc
            except Exception as exc:
                rag_query_count.labels(success="false").inc()
                raise RAGError(f"Query failed: {exc}") from exc

        source_nodes = response.source_nodes or []
        sources = [
            {
                "text": node.node.text[:200],
                "score": node.score,
                "metadata": node.node.metadata,
            }
            for node in source_nodes
        ]
        result = {
            "answer": str(response),
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }
        self._set_cache(cache_key, result)
        rag_query_count.labels(success="true").inc()
        return result
