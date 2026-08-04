from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set, cast

from cachetools import TTLCache
from huggingface_hub import snapshot_download
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from openai import APIConnectionError, APITimeoutError, RateLimitError, OpenAI as OpenAIClient
from redis import Redis
from redis.exceptions import RedisError

import config
from exceptions import APIThrottlingError, QueryTimeoutError, RAGError
from monitoring import rag_cache_hit_ratio, rag_query_count, rag_query_latency
from rag.documents import RAG_DB_Document
from rag.engine.constants import SUPPORTED_RAG_EXTENSIONS
from rag.line_profiler_instrument import profile_if_enabled

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """Query / retrieval / rerank / cache responsibilities."""

    def __init__(self, engine) -> None:
        self._engine = engine


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
            self._engine._rerank_processor is not None
            and self._engine._rerank_model_key == model_name
            and self._engine._rerank_device_key == device_pref
        ):
            return self._engine._rerank_processor
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
            self._engine._rerank_processor = self._build_sentence_reranker(
                resolved_model,
                device=target_device,
            )
        except Exception as exc:
            if device_pref == "cpu":
                logger.warning(
                    "Reranker device=cpu direct init failed, retrying CPU-safe mode: %s",
                    exc,
                )
                self._engine._rerank_processor = self._build_sentence_reranker_cpu_safe(resolved_model)
            elif self._is_cuda_oom_error(exc):
                logger.warning(
                    "CUDA OOM while loading reranker (%s), fallback to CPU",
                    model_name,
                )
                try:
                    self._engine._rerank_processor = self._build_sentence_reranker(
                        resolved_model,
                        device="cpu",
                    )
                except Exception:
                    self._engine._rerank_processor = self._build_sentence_reranker_cpu_safe(resolved_model)
            else:
                raise
        self._engine._rerank_model_key = model_name
        self._engine._rerank_device_key = device_pref
        return self._engine._rerank_processor


    def _init_cache(self):
        if config.settings.CACHE_TYPE == "redis" and config.settings.REDIS_URL:
            try:
                redis_client = Redis.from_url(
                    config.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=float(getattr(config.settings, "REDIS_CONNECT_TIMEOUT", 1.0)),
                    socket_timeout=float(getattr(config.settings, "REDIS_SOCKET_TIMEOUT", 2.0)),
                    retry_on_timeout=False,
                    health_check_interval=30,
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
        if isinstance(self._engine.cache, Redis):
            cached = self._engine.cache.get(key)
            if cached:
                return json.loads(cast(str, cached))
            return None
        return self._engine.cache.get(key)


    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        if isinstance(self._engine.cache, Redis):
            self._engine.cache.setex(key, config.settings.CACHE_TTL, json.dumps(value))
        else:
            self._engine.cache[key] = value


    def _clear_cache(self) -> None:
        if isinstance(self._engine.cache, Redis):
            try:
                for key in self._engine.cache.scan_iter(match="rag:v2:*"):
                    self._engine.cache.delete(key)
            except Exception as exc:
                logger.warning("Failed to clear redis RAG cache: %s", exc)
            return
        try:
            self._engine.cache.clear()
        except Exception as exc:
            logger.warning("Failed to clear in-memory RAG cache: %s", exc)


    def clear_query_cache(self) -> None:
        self._clear_cache()


    def get_query_engine(self) -> RetrieverQueryEngine:
        self._engine._ensure_db_context()
        if self._engine.query_engine is None:
            self._engine._load_or_build_index()
            if self._engine.index is None:
                raise RAGError("RAG index is unavailable")

            retriever = VectorIndexRetriever(
                index=cast(VectorStoreIndex, self._engine.index),
                similarity_top_k=config.settings.SIMILARITY_TOP_K,
            )

            node_postprocessors = []
            if config.settings.ENABLE_RERANK:
                if config.settings.RERANK_MODEL.startswith("cross-encoder"):
                    rerank = self._get_sentence_reranker()
                else:
                    rerank = LLMRerank(llm=self._engine.llm, top_n=config.settings.RERANK_TOP_N)
                node_postprocessors.append(rerank)

            try:
                self._engine.query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    llm=self._engine.llm,
                    node_postprocessors=node_postprocessors,
                )
            except Exception as exc:
                logger.error("RetrieverQueryEngine initialization failed: %s", exc)
                raise RAGError(f"Query engine init failed: {exc}") from exc
        return self._engine.query_engine


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
                    self._engine._ensure_db_context()
                    self._engine._load_or_build_index()
                    if self._engine.index is None:
                        raise RAGError("RAG index is unavailable")
                    retrieval_top_k = min(
                        max(config.settings.SIMILARITY_TOP_K * 4, config.settings.SIMILARITY_TOP_K),
                        40,
                    )
                    retriever = VectorIndexRetriever(
                        index=cast(VectorStoreIndex, self._engine.index),
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
                                llm=self._engine.llm,
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


    @staticmethod
    def _resolve_rag_doc_name(rag_doc: RAG_DB_Document) -> str:
        direct_name = getattr(rag_doc, "doc_name", None)
        if isinstance(direct_name, str) and direct_name.strip():
            return direct_name.strip()
        try:
            payload = rag_doc.list_payload()
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        payload_name = payload.get("doc_name")
        if not isinstance(payload_name, str):
            return ""
        return payload_name.strip()


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
        self._engine._ensure_db_context()
        limit = max(1, min(int(limit), 50))
        cached_entries = self._engine._load_doc_tree_entries()
        if not cached_entries:
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
            self._engine._load_or_build_index()
            if self._engine.index is None:
                raise RAGError("RAG index is unavailable")
            retriever = VectorIndexRetriever(
                index=cast(VectorStoreIndex, self._engine.index),
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
                str(entry.get("doc_name") or "").strip()
                for entry in cached_entries
                if str(entry.get("doc_name") or "").strip()
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

        scoped_entries = [
            entry
            for entry in cached_entries
            if matched_doc_names is None or str(entry.get("doc_name") or "").strip() in matched_doc_names
        ]

        results: List[Dict[str, Any]] = []
        for entry in scoped_entries:
            candidate_doc_name = str(entry.get("doc_name") or "").strip()
            rows = self._engine._filter_doc_tree_search_rows(
                list(entry.get("search_rows") or []),
                compiled_regex=compiled,
                section=section,
                page_start=page_start,
                page_end=page_end,
                chunk=chunk,
            )
            if use_vector:
                section_scores = vector_section_scores.get(candidate_doc_name) or {}
                if not section_scores:
                    continue
                query_norm = (query_text or "").strip().lower()
                for row in rows:
                    metadata = dict(row.get("metadata") or {})
                    section_id = str(row.get("section_id") or metadata.get("section_id") or "").strip()
                    parent_section_id = str(
                        row.get("parent_section_id") or metadata.get("parent_section_id") or ""
                    ).strip()
                    section_path = str(row.get("section_path") or "")
                    text = str(row.get("text") or "")
                    score = float(section_scores.get(section_id, 0.0))
                    if parent_section_id:
                        score = max(score, float(section_scores.get(parent_section_id, 0.0)))
                    if query_norm and query_norm in text.lower():
                        score += 0.2
                    if section_path:
                        score += float(section_scores.get(section_path, 0.0))

                    enriched_row = dict(row)
                    enriched_row["score"] = score
                    results.append(enriched_row)
                continue
            results.extend(rows)

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
        self._engine._ensure_db_context()
        entries = self._engine._load_doc_tree_entries(ensure_keywords_current=True)
        if not entries:
            docs_dir = os.path.join(config.get_rag_persist_dir(), "docs")
            if not os.path.isdir(docs_dir):
                return []
            fallback: List[Dict[str, Any]] = []
            for file_name in sorted(os.listdir(docs_dir)):
                file_path = os.path.join(docs_dir, file_name)
                if not os.path.isfile(file_path):
                    continue
                if os.path.splitext(file_name)[1].lower() not in SUPPORTED_RAG_EXTENSIONS:
                    continue
                fallback.append(
                    {
                        "doc_name": file_name,
                        "title": file_name,
                        "page_count": 0,
                        "chunk_count": 0,
                        "pagination_mode": "",
                        "catalog": [],
                        "keywords": [],
                    }
                )
            return fallback

        summaries: List[Dict[str, Any]] = []
        keyword_limit = max(1, int(getattr(config.settings, "DOC_TREE_SUMMARY_KEYWORDS_LIMIT", 50) or 50))
        for entry in entries:
            summaries.append(
                {
                    "doc_name": str(entry.get("doc_name") or "").strip(),
                    "title": str(entry.get("title") or "").strip(),
                    "page_count": int(entry.get("page_count") or 0),
                    "chunk_count": int(entry.get("chunk_count") or 0),
                    "pagination_mode": str(entry.get("pagination_mode") or "").strip(),
                    "catalog": list(entry.get("catalog") or []),
                    "keywords": [
                        str(keyword).strip()
                        for keyword in list(entry.get("keywords") or [])
                        if str(keyword).strip()
                    ][:keyword_limit],
                }
            )
        return self._engine._sort_doc_tree_entries(summaries)


    def keyword_search(
        self,
        keyword_regex: str,
        top_k: int = -1,
        top_k_percent: float = 0.5,
        return_top_k: int = -1,
        return_top_k_percent: float = -1.0,
        document_ranker: Literal["rank_percent", "rank"] = "rank_percent",
    ) -> Dict[str, Any]:
        self._engine._ensure_db_context()
        pattern_text = str(keyword_regex or "").strip()
        if not pattern_text:
            raise ValueError("keyword_regex is required")
        if document_ranker not in {"rank_percent", "rank"}:
            raise ValueError("document_ranker must be 'rank_percent' or 'rank'")

        compiled_regex = re.compile(pattern_text, re.IGNORECASE)
        entries = self._engine._load_doc_tree_entries(ensure_keywords_current=True)
        documents: List[Dict[str, Any]] = []
        return_top_k_value = int(return_top_k)
        try:
            return_top_k_percent_value = float(return_top_k_percent)
        except (TypeError, ValueError):
            return_top_k_percent_value = -1.0

        for entry in entries:
            keywords = self._engine._normalize_keyword_list(entry.get("keywords") or [])
            total_keywords = len(keywords)
            if total_keywords <= 0:
                continue

            kept_keyword_count = self._engine._resolve_keyword_limit(total_keywords, top_k, top_k_percent)
            candidate_keywords = keywords[:kept_keyword_count]
            matches: List[Dict[str, Any]] = []
            for rank, keyword in enumerate(candidate_keywords, start=1):
                if compiled_regex.search(keyword) is None:
                    continue
                rank_percent = self._engine._keyword_rank_percent(rank, total_keywords)
                matches.append(
                    {
                        "keyword": keyword,
                        "rank": rank,
                        "rank_percent": rank_percent,
                    }
                )

            if not matches:
                continue

            best_rank = min(int(item.get("rank") or 0) for item in matches)
            best_rank_percent = min(float(item.get("rank_percent") or 1.0) for item in matches)
            if return_top_k_value > 0 and best_rank > return_top_k_value:
                continue
            if math.isfinite(return_top_k_percent_value) and return_top_k_percent_value > 0.0 and best_rank_percent > return_top_k_percent_value:
                continue

            documents.append(
                {
                    "doc_name": str(entry.get("doc_name") or "").strip(),
                    "title": str(entry.get("title") or "").strip(),
                    "page_count": int(entry.get("page_count") or 0),
                    "chunk_count": int(entry.get("chunk_count") or 0),
                    "pagination_mode": str(entry.get("pagination_mode") or "").strip(),
                    "total_keyword_count": total_keywords,
                    "kept_keyword_count": kept_keyword_count,
                    "match_count": len(matches),
                    "best_rank": best_rank,
                    "best_rank_percent": best_rank_percent,
                    "matched_keywords": matches,
                }
            )

        documents.sort(
            key=lambda item: (
                float(item.get("best_rank_percent") or 1.0) if document_ranker == "rank_percent" else float(item.get("best_rank") or 10**9),
                float(item.get("best_rank") or 10**9),
                str(item.get("title") or item.get("doc_name") or ""),
                str(item.get("doc_name") or ""),
            )
        )

        return {
            "keyword_regex": pattern_text,
            "document_ranker": document_ranker,
            "filters": {
                "top_k": int(top_k),
                "top_k_percent": float(top_k_percent),
                "return_top_k": return_top_k_value,
                "return_top_k_percent": return_top_k_percent_value,
            },
            "count": len(documents),
            "documents": documents,
        }


    def get_document_catalog(self, doc_name: str) -> list[Dict[str, Any]]:
        """返回指定文档的目录：章节路径及起始页码。"""
        self._engine._ensure_db_context()
        entries = self._engine._load_doc_tree_entries()
        if not entries:
            return []
        available_doc_names = {
            str(entry.get("doc_name") or "").strip()
            for entry in entries
            if str(entry.get("doc_name") or "").strip()
        }
        matched_doc_names = RAG_DB_Document.resolve_doc_name_matches(
            doc_name,
            available_doc_names,
            data_dir=str(config.settings.DATA_DIR or ""),
        )
        if not matched_doc_names:
            return []

        catalog: List[Dict[str, Any]] = []
        for entry in entries:
            candidate_doc_name = str(entry.get("doc_name") or "").strip()
            if candidate_doc_name not in matched_doc_names:
                continue
            for item in list(entry.get("catalog") or []):
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                page = int(row.get("page") or 0)
                end_page = max(page, int(row.get("end_page") or page))
                row["title"] = title
                row["page"] = page
                row["end_page"] = end_page
                row["doc_name"] = candidate_doc_name
                row["category"] = str(row.get("category") or "").strip()
                row["level"] = max(1, int(row.get("level") or 1))
                row["parent_title"] = str(row.get("parent_title") or "").strip() or None
                catalog.append(row)

        sorted_pages: List[Dict[str, Any]] = sorted(
            catalog,
            key=lambda item: (
                str(item["doc_name"]),
                int(item["page"]),
                int(item["level"]),
                -int(item["end_page"]),
                str(item["title"]),
            ),
        )

        return sorted_pages
