from __future__ import annotations

import atexit
import logging
from typing import Any, Callable, Dict, List, Literal, Optional

from llama_index.core import Document
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

import config
from rag.engine.embedding_factory import RAGEmbeddingFactory
from rag.engine.index_manager import DocumentRegistry, RAGIndexManager
from rag.engine.query_engine import RAGQueryEngine
from rag.line_profiler_instrument import start_profiler, stop_profiler

logger = logging.getLogger(__name__)


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
        self.embed_model = RAGEmbeddingFactory.create_embed_model()
        self._embed_dim: Optional[int] = None  # lazy — loaded on first use
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


    # ------------------------------------------------------------------ collaborators (lazy)
    def _embedding_factory(self):
        factory = getattr(self, "_embedding_factory_obj", None)
        if factory is None:
            factory = self._embedding_factory_obj = RAGEmbeddingFactory(self)
        return factory

    def _index_manager(self):
        manager = getattr(self, "_index_manager_obj", None)
        if manager is None:
            manager = self._index_manager_obj = RAGIndexManager(self)
        return manager

    def _query_engine(self):
        qe = getattr(self, "_query_engine_obj", None)
        if qe is None:
            qe = self._query_engine_obj = RAGQueryEngine(self)
        return qe

    def invalidate_runtime_state(self, *, clear_chroma_cache: bool = False) -> None:
        self.index = None
        self.query_engine = None
        if clear_chroma_cache and config.settings.VECTOR_STORE_TYPE == "chroma":
            self._clear_chroma_system_cache()

    def _ensure_db_context(self) -> None:
        current_persist_dir = config.get_rag_persist_dir()
        if current_persist_dir == self._active_persist_dir:
            return
        self._active_persist_dir = current_persist_dir
        self.invalidate_runtime_state(clear_chroma_cache=(config.settings.VECTOR_STORE_TYPE == "chroma"))
        self.doc_registry = DocumentRegistry(current_persist_dir)
        logger.info("Switched RAG context to %s", current_persist_dir)

    @property
    def embed_dim(self) -> int:
        return self._embedding_factory().embed_dim

    # ------------------------------------------------------------------ private wrappers (cross-boundary routing)
    def _init_cache(self):
        return self._query_engine()._init_cache()

    def _clear_chroma_system_cache(self) -> None:
        self._index_manager()._clear_chroma_system_cache()

    def _load_or_build_index(self) -> None:
        self._index_manager()._load_or_build_index()

    def _load_doc_tree_entries(self, *, ensure_keywords_current: bool = False):
        return self._index_manager()._load_doc_tree_entries(ensure_keywords_current=ensure_keywords_current)

    def _filter_doc_tree_search_rows(
        self, rows, *, compiled_regex, section, page_start, page_end, chunk,
    ):
        return self._index_manager()._filter_doc_tree_search_rows(
            rows,
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )

    def _sort_doc_tree_entries(self, entries):
        return self._index_manager()._sort_doc_tree_entries(entries)

    def _normalize_keyword_list(self, values):
        return self._index_manager()._normalize_keyword_list(values)

    def _resolve_keyword_limit(self, total_keywords, top_k, top_k_percent):
        return self._index_manager()._resolve_keyword_limit(total_keywords, top_k, top_k_percent)

    def _keyword_rank_percent(self, rank, total_keywords):
        return self._index_manager()._keyword_rank_percent(rank, total_keywords)

    # ------------------------------------------------------------------ public facade delegations
    def clear_query_cache(self) -> None:
        self._query_engine().clear_query_cache()

    def add_documents(self, docs: List[Document]) -> None:
        self._index_manager().add_documents(docs)

    def add_documents_from_paths(
        self,
        paths: List[str],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> int:
        return self._index_manager().add_documents_from_paths(
            paths,
            progress_callback=progress_callback,
        )

    def rebuild_index_from_paths(
        self,
        paths: List[str],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> int:
        return self._index_manager().rebuild_index_from_paths(
            paths,
            progress_callback=progress_callback,
        )

    def rebuild_index(self) -> int:
        return self._index_manager().rebuild_index()

    def get_query_engine(self) -> RetrieverQueryEngine:
        return self._query_engine().get_query_engine()

    def query(self, question: str, db_names: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._query_engine().query(question, db_names)

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
        return self._query_engine().regex_retrieve(
            regex=regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
            doc_name=doc_name,
            limit=limit,
        )

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
        return self._query_engine().vector_retrieve(
            query=query,
            section=section,
            page_start=page_start,
            page_end=page_end,
            regex=regex,
            chunk=chunk,
            doc_name=doc_name,
            limit=limit,
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._query_engine().list_documents()

    def keyword_search(
        self,
        keyword_regex: str,
        top_k: int = -1,
        top_k_percent: float = 0.5,
        return_top_k: int = -1,
        return_top_k_percent: float = -1.0,
        document_ranker: Literal["rank_percent", "rank"] = "rank_percent",
    ) -> Dict[str, Any]:
        return self._query_engine().keyword_search(
            keyword_regex=keyword_regex,
            top_k=top_k,
            top_k_percent=top_k_percent,
            return_top_k=return_top_k,
            return_top_k_percent=return_top_k_percent,
            document_ranker=document_ranker,
        )

    def get_document_catalog(self, doc_name: str) -> list[Dict[str, Any]]:
        return self._query_engine().get_document_catalog(doc_name)
