from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

import config
from rag.documents import RAG_DB_Document
from rag.engine import RAGEngine
from tool_usage import (
    end_current_tool_call,
    get_current_session_id,
    start_current_tool_call,
)

from ._common import (
    InputSugarTool,
    _bi,
    _coerce_float,
    _EMPTY,
    _prune_empty_fields,
    _t,
    _to_json_safe,
)

logger = logging.getLogger(__name__)

rag_engine = RAGEngine()
_LAST_SEARCH_STATES: dict[str, dict[str, Any]] = {}
_HIDDEN_LINK_STORE: dict[str, list[str]] = {}
_SUGAR_URL_MARKS: dict[str, int] = {}

def _paginate_results(results: list[dict[str, Any]], page: int, page_size: int) -> dict[str, Any]:
    total_results = len(results)
    page_size = max(1, min(int(page_size), 50))
    total_pages = max(1, (total_results + page_size - 1) // page_size) if total_results else 1
    current_page = max(1, min(int(page), total_pages))
    start = (current_page - 1) * page_size
    end = start + page_size

    page_items: list[dict[str, Any]] = []
    for idx, row in enumerate(results[start:end], start=start + 1):
        item = dict(row)
        item["rank"] = idx
        page_items.append(item)

    return {
        "page": current_page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_results": total_results,
        "has_prev": current_page > 1,
        "has_next": current_page < total_pages,
        "results": page_items,
    }


def _build_search_page_payload(state: dict[str, Any], page: int) -> dict[str, Any]:
    paged = _paginate_results(state.get("results") or [], page, int(state.get("page_size") or 10))
    state["current_page"] = paged["page"]
    return {
        "search_type": state.get("search_type"),
        "query": state.get("query") or "",
        "filters": state.get("filters") or {},
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total_pages": paged["total_pages"],
        "total_results": paged["total_results"],
        "has_prev": paged["has_prev"],
        "has_next": paged["has_next"],
        "results": paged["results"],
        "timestamp": state.get("timestamp"),
    }


def _save_last_search(
    search_type: str,
    query: str,
    filters: dict[str, Any],
    sorted_results: list[dict[str, Any]],
    page_size: int,
    timestamp: str,
) -> dict[str, Any]:
    session_id = get_current_session_id() or "__global__"
    state: dict[str, Any] = {
        "search_type": search_type,
        "query": query,
        "filters": filters,
        "results": sorted_results,
        "page_size": max(1, min(int(page_size), 50)),
        "current_page": 1,
        "timestamp": timestamp,
    }
    _LAST_SEARCH_STATES[session_id] = state
    return _build_search_page_payload(state, 1)


def _resolve_paging_target(command: str, current_page: int, total_pages: int) -> int:
    cmd = (command or "").strip().lower()
    if not cmd:
        raise ValueError(_t("分页命令不能为空。", "Paging command cannot be empty."))
    if cmd == "first":
        return 1
    if cmd == "last":
        return total_pages
    if re.fullmatch(r"[+-]\d+", cmd):
        return max(1, min(total_pages, current_page + int(cmd)))
    if re.fullmatch(r"\d+", cmd):
        return max(1, min(total_pages, int(cmd)))
    raise ValueError(
        _t("无效分页命令，支持：+x、-x、x、first、last。", "Invalid paging command; use +x, -x, x, first, or last.")
    )


def _selected_rag_db_names() -> list[str]:
    names: list[str] = []
    for item in config.settings.RAG_DB_NAMES:
        value = (item or "").strip()
        if value and value not in names:
            names.append(value)
    fallback = (config.settings.RAG_DB_NAME or "").strip()
    if not names and fallback:
        names.append(fallback)
    return names


def _catalog_text_items(value: Any) -> list[str]:
    items: list[str] = []
    for item in list(value or []):
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            if title:
                items.append(title)
            continue
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _require_selected_rag_db() -> str | None:
    if not config.settings.ENABLE_RAG:
        return _t("RAG 功能已禁用。", "RAG is disabled.")
    if _selected_rag_db_names():
        return None
    return _t(
        "当前未选择 RAG 数据库，请先在前端选择数据库后再使用 RAG 工具。",
        "No RAG database is selected. Please select one in the UI before using RAG tools.",
    )

class RAGDocListInput(BaseModel):
    pass

class RAGDocCatalogInput(BaseModel):
    doc_name: str = Field(description=_bi("文档名（优先文件名，如 'foo.pdf'；兼容相对路径）", "Document name (prefer file name like 'foo.pdf'; relative path also supported)"))

class RAGKeywordSearchInput(BaseModel):
    keyword_regex: str = Field(description=_bi("关键词正则表达式", "Keyword regex pattern"))
    top_k: int = Field(default=-1, ge=-1, description=_bi("每个文档只保留前 top_k 个关键词；-1 表示不按数量截断", "Keep only the top_k keywords per document; -1 disables count truncation"))
    top_k_percent: float = Field(default=0.5, ge=-1.0, le=1.0, description=_bi("每个文档只保留前 top_k_percent 的关键词；-1 表示不按百分比截断", "Keep only the top_k_percent keywords per document; -1 disables percentage truncation"))
    return_top_k: int = Field(default=-1, ge=-1, description=_bi("仅返回最佳匹配关键词排名在前 top_k 内的文档；-1 表示不过滤", "Return only documents whose best matched keyword rank is within top_k; -1 disables the filter"))
    return_top_k_percent: float = Field(default=-1.0, ge=-1.0, le=1.0, description=_bi("仅返回最佳匹配关键词排名百分比在前 top_k_percent 内的文档；-1 表示不过滤", "Return only documents whose best matched keyword rank percent is within top_k_percent; -1 disables the filter"))
    document_ranker: Literal["rank_percent", "rank"] = Field(default="rank_percent", description=_bi("文档排序方式：按最佳关键词排名百分比或绝对排名升序排序", "Document ordering: ascending best keyword rank percent or absolute rank"))

class RAGRegexSearchInput(BaseModel):
    regex: str = Field(description=_bi("正则表达式", "Regular expression"))
    doc_name: str = Field(default="", description=_bi("限定文档名称（优先文件名；空表示所有文档）", "Restrict document name (prefer file name; empty means all documents)"))
    page_start: int | None = Field(default=None, description=_bi("起始页码（包含）", "Start page (inclusive)"))
    page_end: int | None = Field(default=None, description=_bi("结束页码（包含）", "End page (inclusive)"))
    limit: int = Field(default=10, ge=1, le=50, description=_bi("最大返回结果数", "Maximum number of results"))
    capture_group_weights: list[float] = Field(
        default_factory=list,
        description=_bi(
            "可选：为 regex 中捕获组按顺序设置加分权重（建议配合可选捕获组 (..)? 使用）。例如 regex='(foo)?bar(baz)?' 且 capture_group_weights=[0.4,0.2]，命中第1/2组时分别加分。",
            "Optional: score boosts for regex capture groups by order (recommended with optional groups like (..)?). Example: regex='(foo)?bar(baz)?' and capture_group_weights=[0.4,0.2], matched group 1/2 adds corresponding score.",
        ),
    )

class RAGVectorSearchInput(BaseModel):
    query: str = Field(description=_bi("查询文本", "Query text"))
    doc_name: str = Field(default="", description=_bi("限定文档名称（优先文件名；空表示所有文档）", "Restrict document name (prefer file name; empty means all documents)"))
    page_start: int | None = Field(default=None, description=_bi("起始页码（包含）", "Start page (inclusive)"))
    page_end: int | None = Field(default=None, description=_bi("结束页码（包含）", "End page (inclusive)"))
    limit: int = Field(default=10, ge=1, le=50, description=_bi("最大返回结果数", "Maximum number of results"))


class RAGLastSearchPagingInput(BaseModel):
    page: str = Field(
        description=_bi(
            "分页命令：'+x'（后翻x页）、'-x'（前翻x页）、'x'（跳到第x页）、'first'、'last'",
            "Paging command: '+x' (forward x pages), '-x' (back x pages), 'x' (go to page x), 'first', 'last'",
        )
    )


class RAGGetPagesInput(BaseModel):
    doc_name: str = Field(
        description=_bi(
            "文档名（优先文件名，如 'foo.pdf'；兼容相对路径）",
            "Document name (prefer file name like 'foo.pdf'; relative path also supported)",
        )
    )
    page_start: int = Field(default=1, ge=1, description=_bi("起始页码（包含）", "Start page (inclusive)"))
    page_end: int | None = Field(default=None, ge=1, description=_bi("结束页码（包含）", "End page (inclusive)"))
    max_chunks: int = Field(
        default=80,
        ge=1,
        le=300,
        description=_bi("最大读取块数（用于控制上下文长度）", "Maximum chunks to read (to control context length)"),
    )
    max_chars: int = Field(
        default=18000,
        ge=1000,
        le=120000,
        description=_bi("最大输出字符数（用于控制上下文长度）", "Maximum output characters (to control context length)"),
    )

class RAGDocListTool(InputSugarTool):
    name: str = "rag_doc_list"
    description: str = _bi(
        "列举当前RAG数据库中的所有文档标题、估计页数及基于 logprobs 惊喜度融合排序的关键词。",
        "List all documents in the current RAG database with estimated page counts and logprob-based fused surprise keywords.",
    )
    args_schema: Any = RAGDocListInput

    async def _arun(self) -> str:
        call_id = start_current_tool_call(self.name, {})
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            docs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, rag_engine.list_documents),
                timeout=config.settings.RAG_TOOL_TIMEOUT,
            )
            normalized_docs: list[dict[str, Any]] = []
            for item in list(docs or []):
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["catalog"] = _catalog_text_items(row.get("catalog"))
                normalized_docs.append(row)

            payload = _to_json_safe(normalized_docs)
            pruned = _prune_empty_fields(payload)
            if pruned is _EMPTY:
                pruned = []
            output_text = json.dumps(pruned, ensure_ascii=False)
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("列举文档超时。", "Listing documents timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_doc_list failed")
            output_text = (
                _t(f"列举文档失败: {str(exc)[:200]}", f"Failed to list documents: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("列举文档失败。", "Failed to list documents.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self) -> str:
        raise NotImplementedError("Use async call")

class RAGDocCatalogTool(InputSugarTool):
    name: str = "rag_doc_catalog"
    description: str = _bi(
        "返回指定RAG文档的目录结构（章节路径及页码）。⚠️ 页面范围可能不准确，目录及页码均为程序自动提取，可能存在缺失或错误，请谨慎使用并勿过度依赖页码准确性。", 
        "Return catalog structure of a specified RAG document (section path and page number).⚠️ Page ranges may be inaccurate; both catalog and page numbers are automatically extracted and may contain omissions or errors. Use with caution and do not overly rely on page number accuracy.")
    args_schema: Any = RAGDocCatalogInput

    async def _arun(self, doc_name: str) -> str:
        call_id = start_current_tool_call(self.name, {"doc_name": doc_name})
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            catalog = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, rag_engine.get_document_catalog, doc_name),
                timeout=config.settings.RAG_TOOL_TIMEOUT,
            )
            output_text = _to_json_safe(catalog)
            if not isinstance(output_text, str):
                output_text = "output format error"
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("获取目录超时。", "Fetching catalog timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_doc_catalog failed")
            output_text = (
                _t(f"获取目录失败: {str(exc)[:200]}", f"Failed to fetch catalog: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("获取目录失败。", "Failed to fetch catalog.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, doc_name: str) -> str:
        raise NotImplementedError("Use async call")

class RAGKeywordSearchTool(InputSugarTool):
    name: str = "rag_keyword_search"
    description: str = _bi(
        "在当前 RAG 数据库所有文档的完整关键词列表中做正则匹配，可按每文档关键词 top_k / top_k_percent 过滤，并按最佳关键词排名或排名百分比对文档排序。⚠️注意，关键词更适合用于读取而非检索，请尽可能扩大检索范围，否则极容易返回空结果；同时关键词列表是基于 logprobs 融合的惊喜度排序，可能与直觉不完全一致，请谨慎使用并调整参数以获得更合理的结果。",
        "Regex-match against the full keyword lists of all documents in the current RAG database, with per-document top_k / top_k_percent filtering and document ordering by best keyword rank or rank percent.⚠️ Note that keywords are more suitable for reading than retrieval; please expand the search scope as much as possible, or you may easily get empty results. Also, the keyword list is ordered by logprob-based fused surprise, which may not fully align with intuition. Use with caution and adjust parameters for more reasonable results.",
    )
    args_schema: Any = RAGKeywordSearchInput

    async def _arun(
        self,
        keyword_regex: str,
        top_k: int = -1,
        top_k_percent: float = 0.5,
        return_top_k: int = -1,
        return_top_k_percent: float = -1.0,
        document_ranker: Literal["rank_percent", "rank"] = "rank_percent",
    ) -> str:
        call_id = start_current_tool_call(
            self.name,
            {
                "keyword_regex": keyword_regex,
                "top_k": top_k,
                "top_k_percent": top_k_percent,
                "return_top_k": return_top_k,
                "return_top_k_percent": return_top_k_percent,
                "document_ranker": document_ranker,
            },
        )
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    rag_engine.keyword_search,
                    keyword_regex,
                    top_k,
                    top_k_percent,
                    return_top_k,
                    return_top_k_percent,
                    document_ranker,
                ),
                timeout=config.settings.RAG_TOOL_TIMEOUT,
            )
            payload = _to_json_safe(result)
            pruned = _prune_empty_fields(payload)
            if pruned is _EMPTY:
                pruned = {}
            output_text = json.dumps(pruned, ensure_ascii=False)
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("关键词检索超时。", "Keyword search timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_keyword_search failed")
            output_text = (
                _t(f"关键词检索失败: {str(exc)[:200]}", f"Keyword search failed: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("关键词检索失败。", "Keyword search failed.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")

class RAGRegexSearchTool(InputSugarTool):
    name: str = "rag_regex_search"
    description: str = _bi(
        "根据正则表达式检索文档片段，支持指定文档和页面范围，请确保正则表达式正确。可选动态权重：在 regex 使用 (..)? 等捕获组，并通过 capture_group_weights 按组顺序传入加分权重；命中对应组即加分。",
        "Retrieve document chunks by regex with optional document/page filters, please ensure the regex is correct. Optional dynamic weighting: define capture groups (e.g., (..)? ) in regex and pass score boosts via capture_group_weights in group order; matched groups add score.",
    )
    args_schema: Any = RAGRegexSearchInput

    async def _arun(self, regex: str, doc_name: str = "", page_start: int | None = None,
                    page_end: int | None = None, limit: int = 5,
                    capture_group_weights: list[float] | None = None) -> str:
        call_id = start_current_tool_call(self.name, {
            "regex": regex, "doc_name": doc_name,
            "page_start": page_start, "page_end": page_end, "limit": limit,
            "capture_group_weights": capture_group_weights or [],
        })
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            page_size = max(1, min(int(limit), 50))
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    rag_engine.regex_retrieve,
                    regex,
                    None,          # section
                    page_start,
                    page_end,
                    None,          # chunk
                    doc_name or None,
                    50,
                ),
                timeout=config.settings.RAG_REGEX_RETRIEVE_TIMEOUT,
            )

            safe_result = _to_json_safe(result)
            if not isinstance(safe_result, dict):
                safe_result = {"count": 0, "results": []}

            rows = safe_result.get("results") or []
            normalized_rows: list[dict[str, Any]] = []
            compiled = None
            capture_group_weights = capture_group_weights or []
            if regex.strip() and capture_group_weights:
                try:
                    compiled = re.compile(regex, re.IGNORECASE | re.MULTILINE)
                except re.error:
                    compiled = None

            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                base_score = _coerce_float(item.get("score"), 0.0)
                bonus = 0.0
                matched_groups: list[int] = []
                if compiled is not None:
                    text = str(item.get("text") or "")
                    matched = compiled.search(text)
                    if matched is not None:
                        for idx, weight in enumerate(capture_group_weights, start=1):
                            if idx > matched.re.groups:
                                break
                            group_text = matched.group(idx)
                            if group_text is None or group_text == "":
                                continue
                            bonus += _coerce_float(weight, 0.0)
                            matched_groups.append(idx)
                item["base_score"] = base_score
                item["weight_bonus"] = bonus
                item["score"] = base_score + bonus
                if matched_groups:
                    item["matched_capture_groups"] = matched_groups
                normalized_rows.append(item)

            normalized_rows.sort(key=lambda row: _coerce_float(row.get("score"), 0.0), reverse=True)

            page_payload = _save_last_search(
                search_type=self.name,
                query=str(safe_result.get("query") or ""),
                filters=dict(safe_result.get("filters") or {}),
                sorted_results=normalized_rows,
                page_size=page_size,
                timestamp=str(safe_result.get("timestamp") or ""),
            )

            pruned = _prune_empty_fields(page_payload)
            if pruned is _EMPTY:
                pruned = {}
            output_text = json.dumps(pruned, ensure_ascii=False)
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("正则检索超时。", "Regex retrieval timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_regex_search failed")
            output_text = (
                _t(f"正则检索失败: {str(exc)[:200]}", f"Regex retrieval failed: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("正则检索失败。", "Regex retrieval failed.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")

class RAGVectorSearchTool(InputSugarTool):
    name: str = "rag_vector_search"
    description: str = _bi(
        "根据查询文本的向量相似度检索文档片段（结果不准确），支持指定文档和页面范围，建议谨慎使用。", 
        "Retrieve document chunks by vector similarity to query text (results may be imperfect), with optional document and page range filters.")
    args_schema: Any = RAGVectorSearchInput

    async def _arun(self, query: str, doc_name: str = "", page_start: int | None = None,
                    page_end: int | None = None, limit: int = 5) -> str:
        call_id = start_current_tool_call(self.name, {
            "query": query, "doc_name": doc_name,
            "page_start": page_start, "page_end": page_end, "limit": limit
        })
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            page_size = max(1, min(int(limit), 50))
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    rag_engine.vector_retrieve,
                    query,
                    None,          # section
                    page_start,
                    page_end,
                    None,          # regex
                    None,          # chunk
                    doc_name or None,
                    50,
                ),
                timeout=config.settings.RAG_VECTOR_RETRIEVE_TIMEOUT,
            )

            safe_result = _to_json_safe(result)
            if not isinstance(safe_result, dict):
                safe_result = {"count": 0, "results": []}

            rows = safe_result.get("results") or []
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item["score"] = _coerce_float(item.get("score"), 0.0)
                normalized_rows.append(item)

            normalized_rows.sort(key=lambda row: _coerce_float(row.get("score"), 0.0), reverse=True)

            page_payload = _save_last_search(
                search_type=self.name,
                query=str(safe_result.get("query") or ""),
                filters=dict(safe_result.get("filters") or {}),
                sorted_results=normalized_rows,
                page_size=page_size,
                timestamp=str(safe_result.get("timestamp") or ""),
            )

            pruned = _prune_empty_fields(page_payload)
            if pruned is _EMPTY:
                pruned = {}
            output_text = json.dumps(pruned, ensure_ascii=False)
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("向量检索超时。", "Vector retrieval timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_vector_search failed")
            output_text = (
                _t(f"向量检索失败: {str(exc)[:200]}", f"Vector retrieval failed: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("向量检索失败。", "Vector retrieval failed.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class RAGLastSearchPagingTool(InputSugarTool):
    name: str = "rag_last_search_paging"
    description: str = _bi(
        "对最后一次 rag_regex_search / rag_vector_search 结果翻页：支持 +x、-x、x、first、last。",
        "Paginate the latest rag_regex_search / rag_vector_search results: supports +x, -x, x, first, last.",
    )
    args_schema: Any = RAGLastSearchPagingInput

    async def _arun(self, page: str) -> str:
        call_id = start_current_tool_call(self.name, {"page": page})
        output_text = ""
        try:
            session_id = get_current_session_id() or "__global__"
            state = _LAST_SEARCH_STATES.get(session_id)
            if state is None:
                output_text = _t(
                    "当前没有可翻页的搜索结果，请先执行 rag_regex_search 或 rag_vector_search。",
                    "No searchable history for paging yet. Run rag_regex_search or rag_vector_search first.",
                )
                return output_text

            current_page = int(state.get("current_page") or 1)
            total_pages = _paginate_results(
                state.get("results") or [],
                current_page,
                int(state.get("page_size") or 10),
            )["total_pages"]
            target_page = _resolve_paging_target(page, current_page, total_pages)
            payload = _build_search_page_payload(state, target_page)

            pruned = _prune_empty_fields(payload)
            if pruned is _EMPTY:
                pruned = {}
            output_text = json.dumps(pruned, ensure_ascii=False)
            return output_text
        except ValueError as exc:
            output_text = str(exc)
            return output_text
        except Exception as exc:
            logger.exception("rag_last_search_paging failed")
            output_text = (
                _t(f"翻页失败: {str(exc)[:200]}", f"Paging failed: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("翻页失败。", "Paging failed.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class RAGGetPagesTool(InputSugarTool):
    name: str = "rag_get_pages"
    description: str = _bi(
        "直接按页获取指定文档内容，并整理为 Markdown 供 LLM 使用。⚠️ 请勿一次性获取过长范围或过大内容，否则会显著占用上下文长度并影响后续推理。",
        "Fetch page-range content from a document directly and format it as Markdown for LLM use. ⚠️ Avoid requesting overly long ranges or huge content at once, otherwise context length can be heavily consumed and degrade later reasoning.",
    )
    args_schema: Any = RAGGetPagesInput

    async def _arun(
        self,
        doc_name: str,
        page_start: int = 1,
        page_end: int | None = None,
        max_chunks: int = 80,
        max_chars: int = 18000,
    ) -> str:
        call_id = start_current_tool_call(
            self.name,
            {
                "doc_name": doc_name,
                "page_start": page_start,
                "page_end": page_end,
                "max_chunks": max_chunks,
                "max_chars": max_chars,
            },
        )
        output_text = ""
        try:
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
            page_start = max(1, int(page_start))
            if page_end is not None:
                page_end = max(page_start, int(page_end))

            if not (doc_name or "").strip():
                output_text = _t("doc_name 不能为空。", "doc_name cannot be empty.")
                return output_text

            fetch_limit = max(1, min(int(max_chunks), 300))
            max_chars = max(1000, min(int(max_chars), 120000))

            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    rag_engine.regex_retrieve,
                    None,                # regex
                    None,                # section
                    page_start,
                    page_end,
                    None,                # chunk
                    doc_name,
                    fetch_limit,
                ),
                timeout=config.settings.RAG_REGEX_RETRIEVE_TIMEOUT,
            )

            safe_result = _to_json_safe(result)
            if not isinstance(safe_result, dict):
                safe_result = {"count": 0, "results": []}

            rows = safe_result.get("results") or []
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                p_start = RAG_DB_Document.coerce_page_number(item.get("page_start"))
                p_end = RAG_DB_Document.coerce_page_number(item.get("page_end"))
                p = RAG_DB_Document.coerce_page_number(item.get("page"))
                if p_start is None:
                    p_start = p if p is not None else page_start
                if p_end is None:
                    p_end = p_start
                if p_start is None:
                    p_start = page_start
                if p_end is None:
                    p_end = p_start

                section_path = str(item.get("section_path") or "").strip()
                text = str(item.get("text") or "").strip()
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                chunk_part = RAG_DB_Document.coerce_page_number(metadata.get("chunk_part")) if metadata else None

                if not text:
                    continue

                normalized_rows.append(
                    {
                        "page_start": p_start,
                        "page_end": max(p_start, p_end),
                        "page": p if p is not None else p_start,
                        "section_path": section_path,
                        "chunk_part": chunk_part or 1,
                        "text": text,
                    }
                )

            if not normalized_rows:
                output_text = _t("未获取到页面内容。", "No page content was retrieved.")
                return output_text

            normalized_rows.sort(
                key=lambda item: (
                    int(item.get("page_start") or 0),
                    int(item.get("page_end") or 0),
                    str(item.get("section_path") or ""),
                    int(item.get("chunk_part") or 1),
                )
            )

            requested_end = page_end if page_end is not None else "?"
            header = _t(
                f"# 文档页面内容（Markdown）\n\n- 文档: {doc_name}\n- 请求页范围: {page_start} - {requested_end}\n- 说明: 已按 chunk 聚合为页面视图；若一次读取过长，会占用大量上下文。\n",
                f"# Document Page Content (Markdown)\n\n- Document: {doc_name}\n- Requested page range: {page_start} - {requested_end}\n- Note: Grouped by chunks into a page-oriented view; requesting too much at once can consume large context.\n",
            )

            lines: list[str] = [header]
            seen = set()
            total_chars = len(header)
            truncated = False

            for row in normalized_rows:
                key = (
                    int(row.get("page_start") or 0),
                    int(row.get("page_end") or 0),
                    str(row.get("section_path") or ""),
                    int(row.get("chunk_part") or 1),
                    str(row.get("text") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)

                p_start = int(row.get("page_start") or 0)
                p_end = int(row.get("page_end") or p_start)
                section_path = str(row.get("section_path") or "").strip() or _t("未命名章节", "Untitled Section")
                text = str(row.get("text") or "").strip()

                block = (
                    f"\n## Page {p_start}" if p_start == p_end else f"\n## Pages {p_start}-{p_end}"
                )
                block += f"\n\n### {section_path}\n\n{text}\n"

                if total_chars + len(block) > max_chars:
                    truncated = True
                    break

                lines.append(block)
                total_chars += len(block)

            if truncated:
                lines.append(
                    _t(
                        "\n---\n⚠️ 内容已截断。请缩小页范围或降低 max_chunks 后再次获取。\n",
                        "\n---\n⚠️ Content truncated. Narrow page range or lower max_chunks and fetch again.\n",
                    )
                )

            output_text = "".join(lines).strip()
            return output_text
        except asyncio.TimeoutError:
            output_text = _t("页面内容获取超时。", "Page content retrieval timed out.")
            return output_text
        except Exception as exc:
            logger.exception("rag_get_pages failed")
            output_text = (
                _t(f"获取页面内容失败: {str(exc)[:200]}", f"Failed to get page content: {str(exc)[:200]}")
                if config.settings.ENV != "prod"
                else _t("获取页面内容失败。", "Failed to get page content.")
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")
    
