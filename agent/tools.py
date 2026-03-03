from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
from agent.skills import AgentSkill, get_agent_skills
from exceptions import MCPFatalError
from mcp_client.client import get_mcp_client
from rag.documents import RAG_DB_Document
from rag.engine import RAGEngine
from tool_usage import start_current_tool_call, end_current_tool_call

import subprocess
import re
import xml.etree.ElementTree as ET
from html import unescape
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import os
from pathlib import Path
import sympy as sp
from locale_context import get_current_language

logger = logging.getLogger(__name__)

rag_engine = RAGEngine()
_EMPTY = object()
_LAST_SEARCH_STATE: dict[str, Any] | None = None


def _lang() -> str:
    language = (get_current_language() or "zh").strip().lower()
    if language.startswith("en"):
        return "en"
    return "zh"


def _t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def _bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


def _workspace_root() -> Path:
    return Path(os.getcwd()).resolve()


def _safe_path(path: str) -> Path:
    root = _workspace_root()
    target = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(_t("路径超出工作区根目录", "Path escapes workspace root"))
    return target


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_json_safe(item())
        except Exception:
            pass
    return str(value)


def _prune_empty_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _prune_empty_fields(item)
            if normalized is _EMPTY:
                continue
            cleaned[str(key)] = normalized
        return cleaned if cleaned else _EMPTY
    if isinstance(value, list):
        cleaned_list = [item for item in (_prune_empty_fields(item) for item in value) if item is not _EMPTY]
        return cleaned_list if cleaned_list else _EMPTY
    if isinstance(value, str):
        return _EMPTY if value == "" else value
    if value is None:
        return _EMPTY
    return value


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


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
    global _LAST_SEARCH_STATE
    _LAST_SEARCH_STATE = {
        "search_type": search_type,
        "query": query,
        "filters": filters,
        "results": sorted_results,
        "page_size": max(1, min(int(page_size), 50)),
        "current_page": 1,
        "timestamp": timestamp,
    }
    return _build_search_page_payload(_LAST_SEARCH_STATE, 1)


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


class RAGDocListInput(BaseModel):
    pass

class RAGDocCatalogInput(BaseModel):
    doc_name: str = Field(description=_bi("文档名（优先文件名，如 'foo.pdf'；兼容相对路径）", "Document name (prefer file name like 'foo.pdf'; relative path also supported)"))

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


class MathComputeInput(BaseModel):
    mode: Literal["eval", "simplify", "function", "solve", "matrix"] = Field(
        description=_bi("计算模式：eval/simplify/function/solve/matrix", "Computation mode: eval/simplify/function/solve/matrix")
    )
    expression: str = Field(default="", description=_bi("表达式（eval/simplify/function/matrix 时可用）", "Expression (used in eval/simplify/function/matrix modes)"))
    variables: dict[str, float] = Field(default_factory=dict, description=_bi("变量赋值，如 {\"x\": 2}", "Variable assignments, e.g. {\"x\": 2}"))
    equation: str = Field(default="", description=_bi("方程字符串，如 'x**2-4=0'（solve 模式）", "Equation string, e.g. 'x**2-4=0' (solve mode)"))
    symbol: str = Field(default="x", description=_bi("求解变量名（solve 模式）", "Variable name to solve for (solve mode)"))
    matrix_a: list[list[float]] = Field(default_factory=list, description=_bi("矩阵 A（matrix 模式）", "Matrix A (matrix mode)"))
    matrix_b: list[list[float]] = Field(default_factory=list, description=_bi("矩阵 B（matrix add/sub/mul 模式）", "Matrix B (matrix add/sub/mul modes)"))
    matrix_op: Literal["add", "sub", "mul", "det", "inv", "transpose", "rank"] = Field(
        default="mul", description=_bi("矩阵运算类型", "Matrix operation type")
    )


class MathComputeTool(BaseTool):
    name: str = "math_compute"
    description: str = _bi("数学计算工具：表达式计算/表达式化简/函数计算/方程求解/矩阵运算。", "Math tool: expression eval/simplify, function eval, equation solving, and matrix operations.")
    args_schema: Any = MathComputeInput

    @staticmethod
    def _to_number_if_possible(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value
        if getattr(value, "is_real", False) and getattr(value, "is_number", False):
            as_float = float(value)
            if as_float.is_integer():
                return int(as_float)
            return as_float
        return str(value)

    async def _arun(
        self,
        mode: str,
        expression: str = "",
        variables: dict[str, float] | None = None,
        equation: str = "",
        symbol: str = "x",
        matrix_a: list[list[float]] | None = None,
        matrix_b: list[list[float]] | None = None,
        matrix_op: str = "mul",
    ) -> str:
        call_payload = {
            "mode": mode,
            "expression": expression,
            "variables": variables or {},
            "equation": equation,
            "symbol": symbol,
            "matrix_op": matrix_op,
        }
        call_id = start_current_tool_call(self.name, call_payload)
        output_text = ""
        try:
            mode = (mode or "").strip().lower()
            variables = variables or {}
            local_vars = {name: sp.Float(value) for name, value in variables.items()}

            if mode == "eval":
                if not expression.strip():
                    return _t("表达式不能为空。", "Expression cannot be empty.")
                expr = sp.sympify(expression)
                result = expr.evalf(subs=local_vars)
                payload = {
                    "mode": mode,
                    "result": self._to_number_if_possible(result),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "simplify":
                if not expression.strip():
                    return _t("表达式不能为空。", "Expression cannot be empty.")
                expr = sp.sympify(expression)
                simplified = sp.simplify(expr)
                payload = {
                    "mode": mode,
                    "result": str(simplified),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "function":
                if not expression.strip():
                    return _t("函数表达式不能为空。", "Function expression cannot be empty.")
                expr = sp.sympify(expression)
                result = expr.evalf(subs=local_vars)
                payload = {
                    "mode": mode,
                    "result": self._to_number_if_possible(result),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "solve":
                source = (equation or expression or "").strip()
                if not source:
                    return _t("方程不能为空。", "Equation cannot be empty.")
                var = sp.Symbol((symbol or "x").strip() or "x")
                if "=" in source:
                    left, right = source.split("=", 1)
                    eq = sp.Eq(sp.sympify(left), sp.sympify(right))
                    solutions = sp.solve(eq, var)
                else:
                    solutions = sp.solve(sp.sympify(source), var)
                payload = {
                    "mode": mode,
                    "symbol": str(var),
                    "solutions": [self._to_number_if_possible(item) for item in solutions],
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "matrix":
                matrix_a = matrix_a or []
                matrix_b = matrix_b or []
                if not matrix_a:
                    return _t("矩阵模式需要 matrix_a 参数。", "Matrix mode requires matrix_a.")
                mat_a = sp.Matrix(matrix_a)
                op = (matrix_op or "mul").strip().lower()
                result: Any
                if op == "add":
                    if not matrix_b:
                        return _t("矩阵加法需要 matrix_b 参数。", "Matrix add requires matrix_b.")
                    result = mat_a + sp.Matrix(matrix_b)
                elif op == "sub":
                    if not matrix_b:
                        return _t("矩阵减法需要 matrix_b 参数。", "Matrix subtraction requires matrix_b.")
                    result = mat_a - sp.Matrix(matrix_b)
                elif op == "mul":
                    if not matrix_b:
                        return _t("矩阵乘法需要 matrix_b 参数。", "Matrix multiplication requires matrix_b.")
                    result = mat_a * sp.Matrix(matrix_b)
                elif op == "det":
                    result = mat_a.det()
                elif op == "inv":
                    result = mat_a.inv()
                elif op == "transpose":
                    result = mat_a.T
                elif op == "rank":
                    result = mat_a.rank()
                else:
                    return _t("不支持的矩阵运算类型。", "Unsupported matrix operation.")

                if isinstance(result, sp.MatrixBase):
                    matrix_payload = [
                        [self._to_number_if_possible(cell) for cell in row]
                        for row in result.tolist()
                    ]
                    payload = {
                        "mode": mode,
                        "matrix_op": op,
                        "result": matrix_payload,
                    }
                else:
                    payload = {
                        "mode": mode,
                        "matrix_op": op,
                        "result": self._to_number_if_possible(result),
                    }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            output_text = _t("不支持的计算模式。", "Unsupported computation mode.")
            return output_text
        except Exception as exc:
            logger.exception("math_compute failed")
            if config.settings.ENV == "prod":
                output_text = _t("数学计算失败。", "Math computation failed.")
            else:
                output_text = _t(
                    f"数学计算失败: {type(exc).__name__}: {str(exc)[:160]}",
                    f"Math computation failed: {type(exc).__name__}: {str(exc)[:160]}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class RAGDocListTool(BaseTool):
    name: str = "rag_doc_list"
    description: str = _bi("列举当前RAG数据库中的所有文档标题及其估计页数。", "List all documents in the current RAG database with estimated page counts.")
    args_schema: Any = RAGDocListInput

    async def _arun(self) -> str:
        call_id = start_current_tool_call(self.name, {})
        output_text = ""
        try:
            docs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, rag_engine.list_documents),
                timeout=config.settings.RAG_TOOL_TIMEOUT,
            )
            payload = _to_json_safe(docs)
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

class RAGDocCatalogTool(BaseTool):
    name: str = "rag_doc_catalog"
    description: str = _bi("返回指定RAG文档的目录结构（章节路径及页码）。", "Return catalog structure of a specified RAG document (section path and page number).")
    args_schema: Any = RAGDocCatalogInput

    async def _arun(self, doc_name: str) -> str:
        call_id = start_current_tool_call(self.name, {"doc_name": doc_name})
        output_text = ""
        try:
            catalog = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, rag_engine.get_document_catalog, doc_name),
                timeout=config.settings.RAG_TOOL_TIMEOUT,
            )
            payload = _to_json_safe(catalog)
            pruned = _prune_empty_fields(payload)
            if pruned is _EMPTY:
                pruned = []
            output_text = json.dumps(pruned, ensure_ascii=False)
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

class RAGRegexSearchTool(BaseTool):
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

class RAGVectorSearchTool(BaseTool):
    name: str = "rag_vector_search"
    description: str = _bi("根据查询文本的向量相似度检索文档片段（结果不准确），支持指定文档和页面范围，建议谨慎使用。", "Retrieve document chunks by vector similarity to query text (results may be imperfect), with optional document and page range filters.")
    args_schema: Any = RAGVectorSearchInput

    async def _arun(self, query: str, doc_name: str = "", page_start: int | None = None,
                    page_end: int | None = None, limit: int = 5) -> str:
        call_id = start_current_tool_call(self.name, {
            "query": query, "doc_name": doc_name,
            "page_start": page_start, "page_end": page_end, "limit": limit
        })
        output_text = ""
        try:
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


class RAGLastSearchPagingTool(BaseTool):
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
            state = _LAST_SEARCH_STATE
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


class RAGGetPagesTool(BaseTool):
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
    

class FetchWebpageInput(BaseModel):
    url: str = Field(description=_bi("要抓取的网页 URL", "Webpage URL to fetch"))


class FetchWebpageTool(BaseTool):
    name: str = "fetch_webpage"
    description: str = _bi("抓取指定网页 URL 的内容。", "Fetch content from a specified webpage URL.")
    args_schema: Any = FetchWebpageInput

    @retry(
        stop=stop_after_attempt(config.settings.MCP_RETRY_TIMES),
        wait=wait_exponential(multiplier=config.settings.MCP_RETRY_DELAY),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
    )
    async def _arun(self, url: str) -> str:
        
        call_id = start_current_tool_call(self.name, {"url": url})
        output_text = ""
        client = get_mcp_client()
        try:
            output_text = await client.fetch(url)
            return output_text
        except MCPFatalError:
            output_text = _t("MCP 服务不可用。", "MCP service unavailable.")
            return output_text
        except Exception as exc:
            logger.exception("fetch_webpage failed: %s", url)
            if config.settings.ENV == "prod":
                output_text = _t("网页抓取失败。", "Webpage fetch failed.")
                return output_text
            output_text = _t(
                f"网页抓取失败: {type(exc).__name__}",
                f"Webpage fetch failed: {type(exc).__name__}",
            )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async call")


class SkillBridgeTool(BaseTool):
    skill: AgentSkill
    name: str = ""
    description: str = ""
    args_schema: Any = None

    def __init__(self, skill: AgentSkill, **kwargs: Any):
        super().__init__(
            name=skill.name,
            description=skill.description,
            args_schema=skill.args_schema,
            skill=skill,
            **kwargs,
        )

    async def _arun(self, **kwargs: Any) -> str:
        
        call_id = start_current_tool_call(self.name, kwargs)
        output_text = ""
        try:
            output_text = await self.skill.run(**kwargs)
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async call")


class ShellInput(BaseModel):
    command: str = Field(description=_bi("要执行的 Shell 命令", "Shell command to execute"))
    timeout: int = Field(default=20, ge=1, le=120, description=_bi("执行超时时间（秒）", "Execution timeout in seconds"))


class ShellTool(BaseTool):
    name: str = "skill_shell"
    description: str = _bi("在工作区执行 Shell 命令，并返回标准输出/标准错误。", "Execute shell commands in the workspace and return stdout/stderr.")
    args_schema: Any = ShellInput

    async def _arun(self, command: str, timeout: int = 20) -> str:
        call_id = start_current_tool_call(self.name, {"command": command, "timeout": timeout})
        output_text = ""
        try:
            if not config.settings.ENABLE_SHELL_SKILL:
                output_text = _t("Shell 工具已被配置禁用。", "Shell tool is disabled by configuration.")
                return output_text

            command = command.strip()
            if not command:
                output_text = _t("缺少命令参数。", "Missing command argument.")
                return output_text

            blocked_tokens = [
                "rm -rf /", "shutdown", "reboot", "mkfs", "dd if=", ":(){:|:&};:"
            ]
            lower_command = command.lower()
            if any(token in lower_command for token in blocked_tokens):
                output_text = _t("命令被安全策略拒绝。", "Command rejected by safety policy.")
                return output_text

            def _run_command():
                return subprocess.run(
                    command,
                    shell=True,
                    cwd=str(_workspace_root()),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

            proc = await asyncio.to_thread(_run_command)
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            payload = {
                "exit_code": proc.returncode,
                "stdout": stdout[: config.settings.SKILL_OUTPUT_MAX_CHARS],
                "stderr": stderr[: config.settings.SKILL_OUTPUT_MAX_CHARS],
            }
            output_text = json.dumps(payload, ensure_ascii=False)
            return output_text
        except subprocess.TimeoutExpired:
            output_text = _t(f"命令执行超时（{timeout}秒）。", f"Command timed out ({timeout}s).")
            return output_text
        except Exception as exc:
            logger.exception("shell tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("Shell 工具执行失败。", "Shell tool execution failed.")
            else:
                output_text = _t(
                    f"Shell 工具执行失败: {type(exc).__name__}",
                    f"Shell tool execution failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class FileIOInput(BaseModel):
    action: str = Field(description=_bi("操作类型：read、write、append、list、mkdir", "Action type: read, write, append, list, mkdir"))
    path: str = Field(description=_bi("相对于工作区根目录的路径", "Path relative to workspace root"))
    content: str = Field(default="", description=_bi("write/append 时要写入的内容", "Content to write/append"))


class FileIOTool(BaseTool):
    name: str = "skill_file_io"
    description: str = _bi("在工作区根目录下读/写/追加文件，并列出/创建目录。", "Read/write/append files and list/create directories under workspace root.")
    args_schema: Any = FileIOInput

    async def _arun(self, action: str, path: str, content: str = "") -> str:
        call_id = start_current_tool_call(self.name, {"action": action, "path": path, "content": content})
        output_text = ""
        try:
            if not config.settings.ENABLE_FILE_IO_SKILL:
                output_text = _t("文件 IO 工具已被配置禁用。", "File IO tool is disabled by configuration.")
                return output_text

            action = action.strip().lower()
            path = path.strip()
            if not action or not path:
                output_text = _t("缺少操作类型或路径参数。", "Missing action or path argument.")
                return output_text

            target = _safe_path(path)

            if action == "read":
                if not target.exists() or not target.is_file():
                    output_text = _t("文件不存在。", "File does not exist.")
                    return output_text
                if target.stat().st_size > config.settings.SKILL_MAX_FILE_BYTES:
                    output_text = _t("文件过大，无法通过工具读取。", "File is too large to read via tool.")
                    return output_text
                text = await asyncio.to_thread(target.read_text, "utf-8")
                output_text = text[: config.settings.SKILL_OUTPUT_MAX_CHARS]
                return output_text

            if action == "write":
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_text, content, "utf-8")
                output_text = _t(
                    f"已写入 {len(content)} 个字符到 {target.relative_to(_workspace_root())}",
                    f"Wrote {len(content)} characters to {target.relative_to(_workspace_root())}",
                )
                return output_text

            if action == "append":
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(content)
                output_text = _t(
                    f"已追加 {len(content)} 个字符到 {target.relative_to(_workspace_root())}",
                    f"Appended {len(content)} characters to {target.relative_to(_workspace_root())}",
                )
                return output_text

            if action == "list":
                if target.exists() and target.is_file():
                    output_text = target.name
                    return output_text
                if not target.exists():
                    output_text = _t("路径不存在。", "Path does not exist.")
                    return output_text
                names = sorted(item.name + ("/" if item.is_dir() else "") for item in target.iterdir())
                output_text = "\n".join(names[:200])
                return output_text

            if action == "mkdir":
                await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
                output_text = _t(
                    f"目录已就绪：{target.relative_to(_workspace_root())}",
                    f"Directory is ready: {target.relative_to(_workspace_root())}",
                )
                return output_text

            output_text = _t("不支持的操作类型。", "Unsupported action type.")
            return output_text
        except ValueError as exc:
            output_text = str(exc)
            return output_text
        except Exception as exc:
            logger.exception("file io tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("文件 IO 工具执行失败。", "File IO tool execution failed.")
            else:
                output_text = _t(
                    f"文件 IO 工具执行失败: {type(exc).__name__}",
                    f"File IO tool execution failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class BingSearchInput(BaseModel):
    query: str = Field(description=_bi("搜索关键词", "Search keywords"))
    top_k: int = Field(default=5, ge=1, le=10, description=_bi("返回结果数量", "Number of results to return"))


class AgentIdeaInput(BaseModel):
    content: str = Field(description=_bi("要记录的内容", "Content to record"))

class BingSearchTool(BaseTool):
    name: str = "skill_bing_search"
    description: str = _bi("通过抓取 Bing 搜索结果页进行联网搜索，并返回前若干条结果。", "Run web search by parsing Bing results pages and return top entries.")
    args_schema: Any = BingSearchInput

    async def _arun(self, query: str, top_k: int = 5) -> str:
        call_id = start_current_tool_call(self.name, {"query": query, "top_k": top_k})
        output_text = ""
        try:
            if not config.settings.ENABLE_BING_SEARCH_SKILL:
                output_text = _t("Bing 搜索工具已被配置禁用。", "Bing search tool is disabled by configuration.")
                return output_text

            query = query.strip()
            if not query:
                output_text = _t("缺少查询关键词。", "Missing search query.")
                return output_text

            top_k = max(1, min(top_k, 10))
            set_lang = "zh-Hans" if _lang() == "zh" else "en-US"
            # exactly urls 
            # "https://cn.bing.com/search?q=%E5%A4%A9%E6%B2%B3%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F+mn0%E8%8A%82%E7%82%B9+%E8%8E%B7%E5%8F%96%E6%9C%BA%E6%9F%9CR1P3%E8%8A%82%E7%82%B9%E5%8A%A0%E7%94%B5%E4%BF%A1%E6%81%AF+%E5%91%BD%E4%BB%A4&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIGCAEQABhAMgcIAhDrBxhA0gEHNjM2ajBqMagCALACAA&FORM=ANAB01&PC=U531"
            # "https://cn.bing.com/search?q=%E5%A4%A9%E6%B2%B3%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F+mn0%E8%8A%82%E7%82%B9+%E8%8E%B7%E5%8F%96%E6%9C%BA%E6%9F%9CR1P3%E8%8A%82%E7%82%B9%E5%8A%A0%E7%94%B5%E4%BF%A1%E6%81%AF+%E5%91%BD%E4%BB%A4&form=QBLH&sp=-1&lq=0&pq=%E5%A4%A9%E6%B2%B3%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F+mn0%E8%8A%82%E7%82%B9+%E8%8E%B7%E5%8F%96%E6%9C%BA%E6%9F%9Cr1p3%E8%8A%82%E7%82%B9%E5%8A%A0%E7%94%B5%E4%BF%A1%E6%81%AF+%E5%91%BD%E4%BB%A4&sc=0-34&qs=n&sk=&cvid=A992994A8F3245E6849CE4E4D8A61787"

            url = f"https://www.bing.com/search?q={quote_plus(query)}&count={top_k}&setlang={set_lang}"
            headers = {"User-Agent": config.settings.BING_SEARCH_USER_AGENT}

            def _fetch(target_url: str) -> str:
                request = Request(url=target_url, headers=headers)
                with urlopen(request, timeout=config.settings.BING_SEARCH_TIMEOUT) as response:
                    data = response.read()
                return data.decode("utf-8", errors="ignore")

            def _strip_html(text: str) -> str:
                return unescape(re.sub(r"<.*?>", "", text or "")).strip()

            def _parse_html(html: str) -> list[dict[str, str]]:
                items = []
                blocks = re.findall(r'<li[^>]*class="[^\"]*b_algo[^\"]*"[^>]*>(.*?)</li>', html, re.S)
                for block in blocks:
                    link_match = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
                    if not link_match:
                        continue
                    link = unescape((link_match.group(1) or "").strip())
                    title = _strip_html(link_match.group(2) or "")
                    snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
                    snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
                    if not link:
                        continue
                    items.append({
                        "title": title or link,
                        "link": link,
                        "snippet": snippet,
                    })
                    if len(items) >= top_k:
                        break
                return items

            def _parse_generic_links(html: str) -> list[dict[str, str]]:
                found = []
                seen = set()
                for match in re.findall(r'<a[^>]+href="([^\"]+)"[^>]*>(.*?)</a>', html, re.S):
                    href, inner = match
                    href = unescape((href or "").strip())
                    text = _strip_html(inner)
                    if not href or not text:
                        continue
                    if not href.startswith("http"):
                        continue
                    if len(text) < 8:
                        continue
                    if href in seen:
                        continue
                    if "bing.com/search" in href:
                        continue
                    seen.add(href)
                    found.append({"title": text, "link": href, "snippet": ""})
                    if len(found) >= top_k:
                        break
                return found

            def _parse_rss(xml_text: str) -> list[dict[str, str]]:
                items = []
                try:
                    root = ET.fromstring(xml_text)
                except ET.ParseError:
                    return items
                for item in root.findall("./channel/item"):
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    description = (item.findtext("description") or "").strip()
                    if not link:
                        continue
                    items.append({
                        "title": unescape(title) or link,
                        "link": unescape(link),
                        "snippet": _strip_html(description),
                    })
                    if len(items) >= top_k:
                        break
                return items

            html = await asyncio.to_thread(_fetch, url)
            items = _parse_html(html)
            if not items:
                rss_url = f"https://www.bing.com/search?format=rss&q={quote_plus(query)}"
                rss_text = await asyncio.to_thread(_fetch, rss_url)
                items = _parse_rss(rss_text)
            if not items:
                items = _parse_generic_links(html)
            if not items:
                output_text = _t("未解析到 Bing 搜索结果。", "No Bing search results were parsed.")
                return output_text
            output_text = json.dumps(items, ensure_ascii=False)
            return output_text
        except Exception as exc:
            logger.exception("bing search tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("Bing 搜索失败。", "Bing search failed.")
            else:
                output_text = _t(
                    f"Bing 搜索失败: {type(exc).__name__}",
                    f"Bing search failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class FeedbackTool(BaseTool):
    name: str = "feedback"
    description: str = _bi("反馈 Agent 自己需要的功能，追加记录到 ./agentIdeas/feedback.txt", "Record desired agent features, appended to ./agentIdeas/feedback.txt")
    args_schema: Any = AgentIdeaInput

    async def _arun(self, content: str) -> str:
        call_id = start_current_tool_call(self.name, {"content": content})
        output_text = ""
        try:
            text = (content or "").strip()
            if not text:
                output_text = _t("反馈内容不能为空。", "Feedback content cannot be empty.")
                return output_text

            target = _workspace_root() / "agentIdeas" / "feedback.txt"
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)

            def _append() -> None:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")

            await asyncio.to_thread(_append)
            output_text = _t("已记录到 agentIdeas/feedback.txt", "Recorded to agentIdeas/feedback.txt")
            return output_text
        except Exception as exc:
            logger.exception("feedback tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("写入反馈失败。", "Failed to write feedback.")
            else:
                output_text = _t(
                    f"写入反馈失败: {type(exc).__name__}",
                    f"Failed to write feedback: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class BugTool(BaseTool):
    name: str = "bug"
    description: str = _bi("反馈 Agent 自己遇到的 bug，追加记录到 ./agentIdeas/bug.txt", "Record agent bugs encountered, appended to ./agentIdeas/bug.txt")
    args_schema: Any = AgentIdeaInput

    async def _arun(self, content: str) -> str:
        call_id = start_current_tool_call(self.name, {"content": content})
        output_text = ""
        try:
            text = (content or "").strip()
            if not text:
                output_text = _t("缺陷内容不能为空。", "Bug content cannot be empty.")
                return output_text

            target = _workspace_root() / "agentIdeas" / "bug.txt"
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)

            def _append() -> None:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")

            await asyncio.to_thread(_append)
            output_text = _t("已记录到 agentIdeas/bug.txt", "Recorded to agentIdeas/bug.txt")
            return output_text
        except Exception as exc:
            logger.exception("bug tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("写入缺陷记录失败。", "Failed to write bug record.")
            else:
                output_text = _t(
                    f"写入缺陷记录失败: {type(exc).__name__}",
                    f"Failed to write bug record: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


def _build_skill_tools() -> list[BaseTool]:
    return [SkillBridgeTool(skill=skill) for skill in get_agent_skills()]


tools = [
    # RAG 工具
    RAGDocListTool(),
    RAGDocCatalogTool(),
    RAGRegexSearchTool(),
    RAGVectorSearchTool(),
    RAGLastSearchPagingTool(),
    RAGGetPagesTool(),

    # Agent 反馈
    FeedbackTool(),
    BugTool(),

    # 辅助工具
    MathComputeTool(),
    FetchWebpageTool(),
    ShellTool(),
    FileIOTool(),
    BingSearchTool(),
] + _build_skill_tools()
