from __future__ import annotations

import logging
import math
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Set, cast

import faiss
import numpy as np
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

import config
from exceptions import RAGError
from rag.documents import (
    RAG_DB_Document,
    chunk_documents_from_rag_documents,
    load_chunk_documents_from_data_dir,
    load_chunk_documents_from_persist_dir,
    load_rag_documents_from_paths,
    load_rag_documents_from_persist_dir,
    stable_doc_id,
)
from rag.engine.constants import (
    DOC_TREE_CACHE_FILENAME,
    DOC_TREE_CACHE_VERSION,
    DOC_TREE_KEYWORD_VERSION,
    INDEX_METADATA_DROP_KEYS,
    INDEX_METADATA_MAX_VALUE_LENGTH,
    SUPPORTED_RAG_EXTENSIONS,
)
from rag.file_locking import (
    _json_safe_value,
    _read_json_file_locked,
    _write_json_file_locked,
)
from rag.line_profiler_instrument import profile_if_enabled
from rag.logprob_keyword_extractor import logprobs_extract as extract_document_keywords
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

def _emit_progress_callback(
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    stage: str,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(str(stage), {str(key): value for key, value in payload.items()})
    except Exception as exc:
        logger.debug("Ignoring RAG progress callback failure (%s): %s", stage, exc)


class DocumentRegistry:
    def __init__(self, persist_dir: str) -> None:
        self.path = os.path.join(persist_dir, "doc_registry.json")
        self.cache: Optional[Set[str]] = None
        self.last_load_time = 0.0
        self.cache_ttl = 300

    def _load_with_lock(self) -> Set[str]:
        now = time.time()
        if self.cache and (now - self.last_load_time) < self.cache_ttl:
            return self.cache
        data = _read_json_file_locked(self.path, [])
        self.cache = set(data)
        self.last_load_time = now
        return self.cache

    def _save_with_lock(self) -> None:
        _write_json_file_locked(self.path, sorted(self.cache or set()))

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


class RAGIndexManager:
    """Index construction / persistence / document-registry responsibilities."""

    def __init__(self, engine) -> None:
        self._engine = engine


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


    @staticmethod
    def _clear_chroma_system_cache() -> None:
        try:
            from chromadb.api.client import SharedSystemClient  # type: ignore

            SharedSystemClient.clear_system_cache()
        except Exception as exc:
            logger.debug("Failed to clear Chroma shared system cache: %s", exc)


    @staticmethod
    def _is_readonly_db_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "readonly database" in message
            or "read-only database" in message
            or "attempt to write a readonly database" in message
        )


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
            text = str(doc.text or "")
            if not text.strip():
                continue
            sanitized_docs.append(
                Document(
                    text=text,
                    metadata=self._sanitize_metadata_for_indexing(dict(doc.metadata or {})),
                    doc_id=doc.doc_id or stable_doc_id(doc),
                )
            )
        return sanitized_docs


    def _doc_tree_cache_path(self) -> str:
        return os.path.join(config.get_rag_persist_dir(), DOC_TREE_CACHE_FILENAME)


    @staticmethod
    def _default_doc_tree_cache() -> Dict[str, Any]:
        return {"version": DOC_TREE_CACHE_VERSION, "documents": []}


    def _load_doc_tree_cache(self) -> Dict[str, Any]:
        payload = _read_json_file_locked(
            self._doc_tree_cache_path(),
            self._default_doc_tree_cache(),
        )
        if not isinstance(payload, dict):
            return self._default_doc_tree_cache()
        documents = [
            dict(item)
            for item in list(payload.get("documents") or [])
            if isinstance(item, dict)
        ]
        return {
            "version": int(payload.get("version") or DOC_TREE_CACHE_VERSION),
            "documents": documents,
        }


    def _save_doc_tree_cache(self, payload: Dict[str, Any]) -> None:
        documents = [
            dict(item)
            for item in list((payload or {}).get("documents") or [])
            if isinstance(item, dict)
        ]
        _write_json_file_locked(
            self._doc_tree_cache_path(),
            _json_safe_value({"version": DOC_TREE_CACHE_VERSION, "documents": documents}),
        )


    @staticmethod
    def _sort_doc_tree_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            entries,
            key=lambda item: (
                str(item.get("title") or ""),
                str(item.get("doc_name") or ""),
            ),
        )


    @staticmethod
    def _build_doc_tree_search_rows(rag_doc: RAG_DB_Document) -> List[Dict[str, Any]]:
        rows = rag_doc.retrieve_by_regex(
            compiled_regex=None,
            section=None,
            page_start=None,
            page_end=None,
            chunk=None,
        )
        cached_rows: List[Dict[str, Any]] = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            metadata = dict(row.get("metadata") or {})
            doc_name = str(row.get("doc_name") or metadata.get("doc_name") or rag_doc.doc_name).strip()
            section_path = str(row.get("section_path") or metadata.get("section_path") or "").strip()
            section_id = str(metadata.get("section_id") or "").strip()
            parent_section_id = str(metadata.get("parent_section_id") or "").strip()
            page = RAG_DB_Document.coerce_page_number(row.get("page"))
            page_start = RAG_DB_Document.coerce_page_number(row.get("page_start"))
            page_end = RAG_DB_Document.coerce_page_number(row.get("page_end"))
            cached_rows.append(
                {
                    "text": str(row.get("text") or ""),
                    "doc_name": doc_name,
                    "section_path": section_path,
                    "section_id": section_id,
                    "parent_section_id": parent_section_id or None,
                    "page": page,
                    "page_start": page_start,
                    "page_end": page_end,
                    "metadata": {
                        "doc_name": doc_name,
                        "section_path": section_path,
                        "section_id": section_id,
                        "parent_section_id": parent_section_id or None,
                        "page": page,
                        "section_start_page": page_start,
                        "section_end_page": page_end,
                    },
                }
            )
        return cached_rows


    def _build_doc_tree_cache_entries(self, rag_docs: List[RAG_DB_Document]) -> List[Dict[str, Any]]:
        entries_by_doc_name: Dict[str, Dict[str, Any]] = {}
        for rag_doc in rag_docs:
            payload = rag_doc.list_payload()
            doc_name = str(payload.get("doc_name") or "").strip()
            if not doc_name:
                continue
            entries_by_doc_name[doc_name] = {
                "doc_name": doc_name,
                "title": str(payload.get("title") or "").strip(),
                "page_count": int(payload.get("page_count") or 0),
                "chunk_count": int(payload.get("chunk_count") or 0),
                "pagination_mode": str(payload.get("pagination_mode") or "").strip(),
                "catalog": list(rag_doc.catalog_payload() or []),
                "search_rows": self._build_doc_tree_search_rows(rag_doc),
                "tree": rag_doc.to_payload(),
                "keywords": [],
                "build_trace": rag_doc.get_build_trace(),
            }
        entries = self._sort_doc_tree_entries(list(entries_by_doc_name.values()))
        self._refresh_doc_tree_keywords(entries)
        return entries


    @staticmethod
    def _normalize_keyword_list(values: Any) -> List[str]:
        return [
            str(keyword).strip()
            for keyword in list(values or [])
            if str(keyword).strip()
        ]


    @staticmethod
    def _doc_tree_keywords_need_refresh(entry: Dict[str, Any]) -> bool:
        if int(entry.get("keyword_version") or 0) != DOC_TREE_KEYWORD_VERSION:
            return True
        return not isinstance(entry.get("keywords"), list)


    @staticmethod
    def _resolve_keyword_limit(total_keywords: int, top_k: int, top_k_percent: float) -> int:
        if total_keywords <= 0:
            return 0
        limits: List[int] = []
        top_k_value = int(top_k)
        if top_k_value > 0:
            limits.append(min(total_keywords, top_k_value))
        try:
            percent_value = float(top_k_percent)
        except (TypeError, ValueError):
            percent_value = -1.0
        if math.isfinite(percent_value) and 0.0 < percent_value <= 1.0:
            limits.append(max(1, min(total_keywords, int(math.ceil(total_keywords * percent_value)))))
        return min(limits) if limits else total_keywords


    @staticmethod
    def _keyword_rank_percent(rank: int, total_keywords: int) -> float:
        if total_keywords <= 0:
            return 1.0
        return float(max(1, rank)) / float(total_keywords)


    def _refresh_doc_tree_keywords(self, entries: List[Dict[str, Any]]) -> None:
        texts_by_doc_name: Dict[str, List[str]] = {}
        for entry in entries:
            doc_name = str(entry.get("doc_name") or "").strip()
            if not doc_name:
                continue
            title = str(entry.get("title") or "").strip()
            tree_payload = dict(entry.get("tree") or {}) if isinstance(entry.get("tree"), dict) else {}
            markdown_text = str(tree_payload.get("markdown_text") or "").strip()
            keyword_text_parts: List[str] = []
            if title:
                keyword_text_parts.append(title)
            if markdown_text:
                keyword_text_parts.append(markdown_text)
            if keyword_text_parts:
                texts_by_doc_name.setdefault(doc_name, []).append("\n".join(keyword_text_parts))
        keyword_map: Dict[str, List[str]] = {}
        if texts_by_doc_name:
            try:
                keyword_map = extract_document_keywords(texts_by_doc_name, top_k=-1)
            except (RuntimeError, Exception) as _kw_exc:
                logger.warning("logprobs keyword extraction failed (%s), falling back to TF-IDF", _kw_exc)
                try:
                    from rag.tfidf_keyword_extractor import tfidf_extract
                    keyword_map = tfidf_extract(texts_by_doc_name, top_k=12)
                except Exception:
                    keyword_map = {}
        for entry in entries:
            doc_name = str(entry.get("doc_name") or "").strip()
            entry["keywords"] = self._normalize_keyword_list(keyword_map.get(doc_name) or [])
            entry["keyword_version"] = DOC_TREE_KEYWORD_VERSION
            entry["keyword_algorithm"] = "logprobs_square_surprise_plus_adjusted_total_granularity"


    def _ensure_doc_tree_keywords_current(
        self,
        entries: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], bool]:
        normalized_entries = [dict(item) for item in entries if isinstance(item, dict)]
        if not normalized_entries:
            return [], False

        changed = False
        for entry in normalized_entries:
            normalized_keywords = self._normalize_keyword_list(entry.get("keywords") or [])
            if normalized_keywords != list(entry.get("keywords") or []):
                changed = True
            entry["keywords"] = normalized_keywords

        if any(self._doc_tree_keywords_need_refresh(entry) for entry in normalized_entries):
            self._refresh_doc_tree_keywords(normalized_entries)
            changed = True
        return normalized_entries, changed


    def _persist_doc_tree_cache_from_rag_docs(
        self,
        rag_docs: List[RAG_DB_Document],
        *,
        replace: bool,
    ) -> None:
        new_entries = self._build_doc_tree_cache_entries(rag_docs)
        if replace:
            self._save_doc_tree_cache({"version": DOC_TREE_CACHE_VERSION, "documents": new_entries})
            return

        existing_payload = self._load_doc_tree_cache()
        by_doc_name: Dict[str, Dict[str, Any]] = {}
        for item in list(existing_payload.get("documents") or []):
            if not isinstance(item, dict):
                continue
            doc_name = str(item.get("doc_name") or "").strip()
            if not doc_name:
                continue
            by_doc_name[doc_name] = dict(item)
        for item in new_entries:
            doc_name = str(item.get("doc_name") or "").strip()
            if not doc_name:
                continue
            by_doc_name[doc_name] = dict(item)
        merged_entries = self._sort_doc_tree_entries(list(by_doc_name.values()))
        self._refresh_doc_tree_keywords(merged_entries)
        self._save_doc_tree_cache({"version": DOC_TREE_CACHE_VERSION, "documents": merged_entries})


    def _load_doc_tree_entries(self, *, ensure_keywords_current: bool = False) -> List[Dict[str, Any]]:
        payload = self._load_doc_tree_cache()
        entries = [
            dict(item)
            for item in list(payload.get("documents") or [])
            if isinstance(item, dict)
        ]
        if not ensure_keywords_current:
            return entries
        ensured_entries, changed = self._ensure_doc_tree_keywords_current(entries)
        if changed:
            self._save_doc_tree_cache({"version": DOC_TREE_CACHE_VERSION, "documents": ensured_entries})
        return ensured_entries


    @staticmethod
    def _filter_doc_tree_search_rows(
        rows: List[Dict[str, Any]],
        *,
        compiled_regex: Optional[re.Pattern[str]],
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        chunk: Optional[str],
    ) -> List[Dict[str, Any]]:
        section_norm = (section or "").strip().lower()
        chunk_norm = (chunk or "").strip().lower()
        page_filtered = page_start is not None or page_end is not None
        filtered: List[Dict[str, Any]] = []
        for raw_row in list(rows or []):
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            metadata = dict(row.get("metadata") or {})
            section_path = str(row.get("section_path") or metadata.get("section_path") or "").strip()
            if section_norm and section_norm not in section_path.lower():
                continue
            if chunk_norm and chunk_norm not in text.lower():
                continue
            if compiled_regex and not compiled_regex.search(text):
                continue

            candidate_start = RAG_DB_Document.coerce_page_number(row.get("page_start"))
            candidate_end = RAG_DB_Document.coerce_page_number(row.get("page_end"))
            node_page = RAG_DB_Document.coerce_page_number(row.get("page")) or candidate_start or candidate_end

            if page_filtered:
                if candidate_start is None or candidate_end is None:
                    continue
                if page_start is not None and candidate_end < page_start:
                    continue
                if page_end is not None and candidate_start > page_end:
                    continue

            doc_name = str(row.get("doc_name") or metadata.get("doc_name") or "").strip()
            section_id = str(row.get("section_id") or metadata.get("section_id") or "").strip()
            parent_section_id = str(
                row.get("parent_section_id") or metadata.get("parent_section_id") or ""
            ).strip()
            normalized_metadata = dict(metadata)
            normalized_metadata.setdefault("doc_name", doc_name)
            normalized_metadata.setdefault("section_path", section_path)
            normalized_metadata.setdefault("section_id", section_id)
            if parent_section_id:
                normalized_metadata.setdefault("parent_section_id", parent_section_id)
            normalized_metadata.setdefault("page", node_page)
            normalized_metadata.setdefault("section_start_page", candidate_start)
            normalized_metadata.setdefault("section_end_page", candidate_end)
            filtered.append(
                {
                    "score": 0.0,
                    "text": text[:1400],
                    "doc_name": doc_name,
                    "section_path": section_path or None,
                    "page": node_page,
                    "page_start": candidate_start,
                    "page_end": candidate_end,
                    "section_id": section_id,
                    "parent_section_id": parent_section_id or None,
                    "metadata": normalized_metadata,
                }
            )
        return filtered


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
                [self._engine.embed_model.get_text_embedding(doc.text) for doc in sample_docs],
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
        rag_docs = load_rag_documents_from_persist_dir(
            config.get_rag_persist_dir(),
            SUPPORTED_RAG_EXTENSIONS,
        )
        if not rag_docs:
            self._engine.index = None
            self._engine.query_engine = None
            self._save_doc_tree_cache(self._default_doc_tree_cache())
            self._save_doc_registry(set())
            return False
        docs = chunk_documents_from_rag_documents(rag_docs)
        if not docs:
            self._engine.index = None
            self._engine.query_engine = None
            self._persist_doc_tree_cache_from_rag_docs(rag_docs, replace=True)
            self._save_doc_registry(set())
            return False
        self._build_index_from_docs(docs)
        self._persist_doc_tree_cache_from_rag_docs(rag_docs, replace=True)
        return True


    @profile_if_enabled
    def _build_index_from_docs(self, docs: List[Document]) -> None:
        docs = self._sanitize_documents_for_indexing(docs)
        if not docs:
            raise ValueError("Document tree chunking produced no indexable nodes")

        persist_dir = config.get_rag_persist_dir()

        def _build_once() -> None:
            vector_store = get_vector_store(
                store_type=config.settings.VECTOR_STORE_TYPE,
                persist_dir=persist_dir,
                embed_dim=self._engine.embed_dim,
                embed_model=self._engine.embed_model,
                index_builder=self._build_faiss_index_with_training,
            )
            if (
                getattr(vector_store, "fallback", False)
                and config.settings.VECTOR_STORE_FALLBACK_WARNING
            ):
                logger.warning("Vector store fallback active; performance may be degraded")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._engine.index = VectorStoreIndex(
                docs,
                storage_context=storage_context,
                embed_model=self._engine.embed_model,
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
                self._engine.invalidate_runtime_state(clear_chroma_cache=True)
                self._reset_persisted_index_artifacts(persist_dir)
                _build_once()
            else:
                raise
        if self._engine.index is None:
            raise RAGError("RAG index build failed")
        self._engine.index.storage_context.persist(persist_dir=config.get_rag_persist_dir())
        self._persist_faiss_index()
        self._save_doc_registry({doc.doc_id for doc in docs})
        self._engine.clear_query_cache()
        logger.info("Built and persisted index")


    def _persist_faiss_index(self) -> None:
        if not self._engine.index:
            return
        vector_store = self._engine.index.storage_context.vector_store
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
        self._engine._ensure_db_context()
        persist_dir = config.get_rag_persist_dir()
        os.makedirs(persist_dir, exist_ok=True)
        selected_db_mode = bool((config.settings.RAG_DB_NAME or "").strip())

        try:
            vector_store = get_vector_store(
                store_type=config.settings.VECTOR_STORE_TYPE,
                persist_dir=persist_dir,
                embed_dim=self._engine.embed_dim,
                embed_model=self._engine.embed_model,
                index_builder=self._build_faiss_index_with_training,
            )
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=vector_store,
            )
            self._engine.index = load_index_from_storage(
                storage_context,
                embed_model=self._engine.embed_model,
            )
            logger.info("Loaded index from storage")
        except FileNotFoundError:
            logger.info("No persisted index found, building a new one")
            self._engine.invalidate_runtime_state(clear_chroma_cache=(config.settings.VECTOR_STORE_TYPE == "chroma"))
            if selected_db_mode:
                built = self._build_index_for_selected_db()
                if not built:
                    logger.info("Selected DB has no docs/index: %s", config.settings.RAG_DB_NAME)
            else:
                self._build_index()
        except Exception as exc:
            logger.error("Failed to load index, rebuilding: %s", exc)
            if config.settings.VECTOR_STORE_TYPE == "chroma" and self._is_readonly_db_error(exc):
                self._engine.invalidate_runtime_state(clear_chroma_cache=True)
                self._reset_persisted_index_artifacts(persist_dir)
            if selected_db_mode:
                built = self._build_index_for_selected_db()
                if not built:
                    logger.info("Selected DB has no docs/index after rebuild: %s", config.settings.RAG_DB_NAME)
            else:
                self._build_index()


    @profile_if_enabled
    def _add_documents_incremental(self, new_docs: List[Document]) -> None:
        self._engine._ensure_db_context()
        if self._engine.index is None:
            self._load_or_build_index()
        if self._engine.index is None:
            raise RAGError("RAG index is unavailable")

        new_docs = self._sanitize_documents_for_indexing(new_docs)
        if not new_docs:
            logger.info("No indexable document-tree chunks to add")
            return

        existing_ids = self._get_existing_doc_ids()
        inserted = 0
        inserted_ids: Set[str] = set()
        for doc in new_docs:
            doc_id = doc.doc_id or stable_doc_id(doc)
            if doc_id in existing_ids:
                continue
            self._engine.index.insert_nodes([doc])
            inserted += 1
            inserted_ids.add(doc_id)
        if inserted:
            self._engine.index.storage_context.persist(persist_dir=config.get_rag_persist_dir())
            self._persist_faiss_index()
            self._engine.doc_registry.add_ids(inserted_ids)
            self._engine.clear_query_cache()
        logger.info("Incrementally added %s documents", inserted)


    def _get_existing_doc_ids(self) -> Set[str]:
        if self._engine.index is None:
            return set()
        vector_store = self._engine.index.storage_context.vector_store
        if config.settings.VECTOR_STORE_TYPE == "chroma":
            collection = getattr(vector_store, "_collection", None)
            if collection:
                data = collection.get(include=["metadatas"])
                return {
                    meta.get("doc_id")
                    for meta in data.get("metadatas", [])
                    if meta and meta.get("doc_id")
                }
        return self._engine.doc_registry.get_existing_ids()


    def _save_doc_registry(self, doc_ids: Set[str]) -> None:
        try:
            self._engine.doc_registry.set_all(doc_ids)
        except Exception as exc:
            logger.warning("Failed to write doc registry: %s", exc)


    def add_documents(self, docs: List[Document]) -> None:
        self._engine._ensure_db_context()
        self._add_documents_incremental(docs)


    def add_documents_from_paths(
        self,
        paths: List[str],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> int:
        self._engine._ensure_db_context()
        _emit_progress_callback(progress_callback, "engine_started", mode="add", path_count=len(paths))
        rag_docs = load_rag_documents_from_paths(
            paths,
            SUPPORTED_RAG_EXTENSIONS,
            progress_callback=progress_callback,
        )
        if not rag_docs:
            _emit_progress_callback(progress_callback, "engine_completed", mode="add", added=0, doc_count=0)
            return 0
        docs = chunk_documents_from_rag_documents(rag_docs)
        if docs:
            _emit_progress_callback(
                progress_callback,
                "index_started",
                mode="add",
                doc_count=len(rag_docs),
                chunk_count=len(docs),
            )
            self._add_documents_incremental(docs)
            _emit_progress_callback(
                progress_callback,
                "index_completed",
                mode="add",
                doc_count=len(rag_docs),
                chunk_count=len(docs),
            )
        _emit_progress_callback(progress_callback, "persist_started", mode="add", doc_count=len(rag_docs))
        self._persist_doc_tree_cache_from_rag_docs(rag_docs, replace=False)
        _emit_progress_callback(
            progress_callback,
            "persist_completed",
            mode="add",
            added=len(docs),
            doc_count=len(rag_docs),
        )
        return len(docs)


    def rebuild_index_from_paths(
        self,
        paths: List[str],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> int:
        self._engine._ensure_db_context()
        self._engine.index = None
        self._engine.query_engine = None
        _emit_progress_callback(progress_callback, "engine_started", mode="rebuild", path_count=len(paths))
        if not paths:
            self._save_doc_tree_cache(self._default_doc_tree_cache())
            self._save_doc_registry(set())
            _emit_progress_callback(progress_callback, "engine_completed", mode="rebuild", added=0, doc_count=0)
            return 0
        rag_docs = load_rag_documents_from_paths(
            paths,
            SUPPORTED_RAG_EXTENSIONS,
            progress_callback=progress_callback,
        )
        docs = chunk_documents_from_rag_documents(rag_docs)
        if not docs:
            _emit_progress_callback(progress_callback, "persist_started", mode="rebuild", doc_count=len(rag_docs))
            self._persist_doc_tree_cache_from_rag_docs(rag_docs, replace=True)
            self._save_doc_registry(set())
            _emit_progress_callback(
                progress_callback,
                "persist_completed",
                mode="rebuild",
                added=0,
                doc_count=len(rag_docs),
            )
            return 0
        _emit_progress_callback(
            progress_callback,
            "index_started",
            mode="rebuild",
            doc_count=len(rag_docs),
            chunk_count=len(docs),
        )
        self._build_index_from_docs(docs)
        _emit_progress_callback(
            progress_callback,
            "index_completed",
            mode="rebuild",
            doc_count=len(rag_docs),
            chunk_count=len(docs),
        )
        _emit_progress_callback(progress_callback, "persist_started", mode="rebuild", doc_count=len(rag_docs))
        self._persist_doc_tree_cache_from_rag_docs(rag_docs, replace=True)
        _emit_progress_callback(
            progress_callback,
            "persist_completed",
            mode="rebuild",
            added=len(docs),
            doc_count=len(rag_docs),
        )
        return len(docs)


    def rebuild_index(self) -> int:
        self._engine._ensure_db_context()
        self._engine.index = None
        self._engine.query_engine = None
        docs = self._load_all_documents()
        if not docs:
            raise ValueError(f"No documents found in {config.settings.DATA_DIR}")
        self._build_index_from_docs(docs)
        return len(docs)
