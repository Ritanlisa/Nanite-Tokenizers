from __future__ import annotations

import asyncio
import json
import logging
import random
import time
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
from html import unescape
import os
from pathlib import Path
import sympy as sp
from locale_context import get_current_language
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.common.exceptions import TimeoutException

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


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<.*?>", "", text or "")).strip()


def _sleep_with_jitter(base: float, jitter: float = 0.35) -> None:
    delay = max(0.0, base + random.uniform(0.0, max(0.0, jitter)))
    time.sleep(delay)


def _configure_driver_options(kind: str) -> tuple[str, Any]:
    normalized = (kind or "edge").strip().lower()
    user_agent = (config.settings.SEARCH_USER_AGENT or "").strip()

    if normalized == "chrome":
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1366,900")
        if user_agent:
            options.add_argument(f"--user-agent={user_agent}")
        return normalized, options

    if normalized == "firefox":
        options = FirefoxOptions()
        options.add_argument("-headless")
        if user_agent:
            options.set_preference("general.useragent.override", user_agent)
        options.set_preference("dom.webdriver.enabled", False)
        return normalized, options

    options = EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,900")
    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")
    return "edge", options


def _build_webdriver() -> Any:
    preferred_kind = (config.settings.SEARCH_WEBDRIVER_KIND or "edge").strip().lower() or "edge"
    fallback_order = [preferred_kind, "chrome", "firefox", "edge"]
    tried_kinds: list[str] = []
    last_exc: Exception | None = None

    configured_path = (config.settings.SEARCH_WEBDRIVER_PATH or "").strip()
    resolved_path = str(_safe_path(configured_path)) if configured_path else ""

    for candidate in fallback_order:
        if candidate in tried_kinds:
            continue
        tried_kinds.append(candidate)
        kind, options = _configure_driver_options(candidate)
        candidate_path = resolved_path if resolved_path and kind == preferred_kind else ""

        try:
            if kind == "chrome":
                service = ChromeService(executable_path=candidate_path) if candidate_path else ChromeService()
                driver = ChromeDriver(service=service, options=options)
            elif kind == "firefox":
                service = FirefoxService(executable_path=candidate_path) if candidate_path else FirefoxService()
                driver = FirefoxDriver(service=service, options=options)
            elif kind == "edge":
                service = EdgeService(executable_path=candidate_path) if candidate_path else EdgeService()
                driver = EdgeDriver(service=service, options=options)
            else:
                raise ValueError(_t(f"不支持的浏览器类型: {kind}", f"Unsupported browser type: {kind}"))

            try:
                driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {
                        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                    },
                )
            except Exception:
                pass
            return driver
        except Exception as exc:
            last_exc = exc
            logger.warning("webdriver init failed for %s, trying fallback: %s", kind, type(exc).__name__)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(_t("未能初始化任何 WebDriver。", "Failed to initialize any WebDriver."))


def _parse_search_results_with_pattern(html: str, pattern: re.Pattern[str], top_k: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_links: set[str] = set()
    for match in pattern.finditer(html or ""):
        groups = match.groupdict()
        link = unescape((groups.get("link") or "").strip())
        title = _strip_html(groups.get("title") or "")
        snippet = _strip_html(groups.get("snippet") or "")
        if not link or not link.startswith("http"):
            continue
        if link in seen_links:
            continue
        seen_links.add(link)
        results.append({
            "title": title or link,
            "link": link,
            "snippet": snippet,
        })
        if len(results) >= top_k:
            break
    return results


def _parse_search_results_from_html(html: str, top_k: int) -> list[dict[str, str]]:
    regex_text = (config.settings.SEARCH_RESULT_REGEX or "").strip()
    if not regex_text:
        return []
    try:
        pattern = re.compile(regex_text, re.S | re.I)
    except re.error:
        return []
    return _parse_search_results_with_pattern(html=html, pattern=pattern, top_k=top_k)


def _set_url_query_param(url: str, key: str, value: str) -> str:
    parsed = urlsplit(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    replaced = False
    normalized: list[tuple[str, str]] = []
    for item_key, item_value in query_pairs:
        if item_key == key:
            normalized.append((item_key, value))
            replaced = True
        else:
            normalized.append((item_key, item_value))
    if not replaced:
        normalized.append((key, value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(normalized), parsed.fragment))


def _fill_search_input(element: Any, query: str) -> None:
    tag_name = (getattr(element, "tag_name", "") or "").lower()
    target = element
    if tag_name not in {"input", "textarea"}:
        inputs = element.find_elements(By.XPATH, ".//input[not(@type='hidden')]")
        if inputs:
            target = inputs[0]
    target.clear()
    for ch in query:
        target.send_keys(ch)
        time.sleep(random.uniform(0.02, 0.08))


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
            scope_error = _require_selected_rag_db()
            if scope_error:
                output_text = scope_error
                return output_text
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

class RAGVectorSearchTool(BaseTool):
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
    

class FetchWebpageInput(BaseModel):
    url: str = Field(description=_bi("要抓取的网页 URL", "Webpage URL to fetch"))


class SeleniumWebVisitInput(BaseModel):
    url: str = Field(description=_bi("要访问的网页 URL", "Webpage URL to visit"))
    wait_xpath: str = Field(
        default="",
        description=_bi("可选：等待页面元素可见的 XPath", "Optional: XPath to wait until visible"),
    )
    max_chars: int = Field(
        default=8000,
        ge=500,
        le=50000,
        description=_bi("返回正文最大字符数", "Maximum returned text length"),
    )


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


class WebVisitTool(BaseTool):
    name: str = "skill_web_visit"
    description: str = _bi(
        "基于 Selenium 访问网页并提取正文文本（适用于需要 JS 渲染的页面）。",
        "Visit webpages with Selenium and extract main text (for JS-rendered pages).",
    )
    args_schema: Any = SeleniumWebVisitInput

    async def _arun(self, url: str, wait_xpath: str = "", max_chars: int = 8000) -> str:
        call_id = start_current_tool_call(self.name, {"url": url, "wait_xpath": wait_xpath, "max_chars": max_chars})
        output_text = ""
        try:
            target_url = (url or "").strip()
            if not target_url:
                output_text = _t("缺少 URL 参数。", "Missing URL argument.")
                return output_text
            if not re.match(r"^https?://", target_url, re.I):
                output_text = _t("仅支持 http/https URL。", "Only http/https URLs are supported.")
                return output_text

            timeout = int(config.settings.SEARCH_TIMEOUT)
            wait_xpath = (wait_xpath or "").strip()
            max_chars = max(500, min(int(max_chars), 50000))

            def _visit() -> str:
                driver = _build_webdriver()
                wait = WebDriverWait(driver, timeout)
                try:
                    driver.get(target_url)
                    _sleep_with_jitter(0.8)

                    if wait_xpath:
                        try:
                            wait.until(EC.visibility_of_element_located((By.XPATH, wait_xpath)))
                        except TimeoutException:
                            pass

                    try:
                        body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                        driver.execute_script("arguments[0].scrollIntoView({block:'start'});", body)
                    except Exception:
                        pass

                    _sleep_with_jitter(0.5)
                    text = (driver.execute_script("return (document.body && document.body.innerText) || '';") or "").strip()
                    if not text:
                        text = _strip_html(driver.page_source or "")
                    return text[:max_chars]
                finally:
                    driver.quit()

            output_text = await asyncio.to_thread(_visit)
            if not output_text:
                output_text = _t("页面内容为空或未提取到文本。", "Page content is empty or no text extracted.")
            return output_text
        except Exception as exc:
            logger.exception("selenium web visit failed")
            if config.settings.ENV == "prod":
                output_text = _t("网页访问失败。", "Web visit failed.")
            else:
                output_text = _t(
                    f"网页访问失败: {type(exc).__name__}",
                    f"Web visit failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
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


class SearchInput(BaseModel):
    query: str = Field(description=_bi("搜索关键词", "Search keywords"))
    top_k: int = Field(default=5, ge=1, le=10, description=_bi("返回结果数量", "Number of results to return"))


class URLRegSearchInput(BaseModel):
    url: str = Field(description=_bi("要访问并解析的 URL", "URL to visit and parse"))
    regex: str = Field(
        description=_bi(
            "用于提取结果的正则（必须使用命名组 link，建议提供 title/snippet）",
            "Regex to extract results (must include named group 'link'; title/snippet are recommended)",
        )
    )
    top_k: int = Field(default=5, ge=1, le=20, description=_bi("返回结果数量", "Number of results to return"))
    max_html_chars: int = Field(
        default=300000,
        ge=5000,
        le=800000,
        description=_bi("用于正则匹配的最大 HTML 字符数", "Maximum HTML characters used for regex matching"),
    )


class AgentIdeaInput(BaseModel):
    content: str = Field(description=_bi("要记录的内容", "Content to record"))

class SearchTool(BaseTool):
    name: str = "skill_search"
    description: str = _bi(
        "使用 Selenium 驱动搜索引擎页面并返回结果，支持可配置 XPath/Regex 与翻页。",
        "Use Selenium to drive search pages and return results, with configurable XPath/Regex and paging.",
    )
    args_schema: Any = SearchInput

    async def _arun(self, query: str, top_k: int = 5) -> str:
        call_id = start_current_tool_call(self.name, {"query": query, "top_k": top_k})
        output_text = ""
        try:
            if not config.settings.ENABLE_SEARCH_SKILL:
                output_text = _t("搜索工具已被配置禁用。", "Search tool is disabled by configuration.")
                return output_text

            query = query.strip()
            if not query:
                output_text = _t("缺少查询关键词。", "Missing search query.")
                return output_text

            top_k = max(1, min(top_k, 10))
            timeout = int(config.settings.SEARCH_TIMEOUT)

            def _run_search() -> list[dict[str, str]]:
                driver = _build_webdriver()
                wait = WebDriverWait(driver, timeout)
                try:
                    search_url = (config.settings.SEARCH_URL or "https://cn.bing.com/").strip()
                    box_xpath = (config.settings.SEARCH_BOX_XPATH or "//input[@name='q']").strip()
                    button_xpath = (config.settings.SEARCH_BUTTON_XPATH or "").strip()
                    page_param = (config.settings.SEARCH_PAGE_PARAM or "first").strip() or "first"
                    page_size = max(1, int(config.settings.SEARCH_PAGE_SIZE or 10))

                    driver.get(search_url)
                    _sleep_with_jitter(0.7)

                    box = wait.until(EC.presence_of_element_located((By.XPATH, box_xpath)))
                    _fill_search_input(box, query)
                    if button_xpath:
                        try:
                            button = wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
                            button.click()
                        except Exception:
                            box.send_keys(Keys.ENTER)
                    else:
                        box.send_keys(Keys.ENTER)

                    max_pages = max(1, (top_k + page_size - 1) // page_size)
                    collected: list[dict[str, str]] = []
                    seen_links: set[str] = set()

                    base_results_url = driver.current_url
                    for page_index in range(max_pages):
                        if page_index > 0:
                            offset = page_index * page_size
                            paged_url = _set_url_query_param(base_results_url, page_param, str(offset))
                            driver.get(paged_url)
                        _sleep_with_jitter(0.9)

                        page_html = driver.page_source or ""
                        parsed = _parse_search_results_from_html(page_html, top_k=top_k)
                        for item in parsed:
                            link = item.get("link") or ""
                            if not link or link in seen_links:
                                continue
                            seen_links.add(link)
                            collected.append(item)
                            if len(collected) >= top_k:
                                break

                        if len(collected) >= top_k:
                            break

                    return collected[:top_k]
                finally:
                    driver.quit()

            items = await asyncio.to_thread(_run_search)
            if not items:
                output_text = _t("未解析到搜索结果。", "No search results were parsed.")
                return output_text
            output_text = json.dumps(items, ensure_ascii=False)
            return output_text
        except Exception as exc:
            logger.exception("search tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("搜索失败。", "Search failed.")
            else:
                output_text = _t(
                    f"搜索失败: {type(exc).__name__}",
                    f"Search failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class URLRegSearchTool(BaseTool):
    name: str = "skill_url_reg_search"
    description: str = _bi(
        "从指定 URL 抓取页面 HTML，并按给定正则匹配搜索结果（命名组：link/title/snippet）。",
        "Fetch HTML from a specific URL and extract search-like results using provided regex (named groups: link/title/snippet).",
    )
    args_schema: Any = URLRegSearchInput

    async def _arun(self, url: str, regex: str, top_k: int = 5, max_html_chars: int = 300000) -> str:
        call_id = start_current_tool_call(
            self.name,
            {"url": url, "regex": regex, "top_k": top_k, "max_html_chars": max_html_chars},
        )
        output_text = ""
        try:
            if not config.settings.ENABLE_SEARCH_SKILL:
                output_text = _t("搜索工具已被配置禁用。", "Search tool is disabled by configuration.")
                return output_text

            target_url = (url or "").strip()
            if not target_url:
                output_text = _t("缺少 URL 参数。", "Missing URL argument.")
                return output_text
            if not re.match(r"^https?://", target_url, re.I):
                output_text = _t("仅支持 http/https URL。", "Only http/https URLs are supported.")
                return output_text

            regex_text = (regex or "").strip()
            if not regex_text:
                output_text = _t("缺少正则表达式参数。", "Missing regex argument.")
                return output_text

            try:
                pattern = re.compile(regex_text, re.S | re.I)
            except re.error as exc:
                output_text = _t(f"正则表达式无效: {exc}", f"Invalid regex: {exc}")
                return output_text

            if "link" not in pattern.groupindex:
                output_text = _t(
                    "正则必须包含命名组 'link'。",
                    "Regex must include named group 'link'.",
                )
                return output_text

            top_k = max(1, min(int(top_k), 20))
            max_html_chars = max(5000, min(int(max_html_chars), 800000))

            def _run_parse() -> list[dict[str, str]]:
                driver = _build_webdriver()
                try:
                    driver.get(target_url)
                    _sleep_with_jitter(1.0)
                    page_html = (driver.page_source or "")[:max_html_chars]
                    return _parse_search_results_with_pattern(page_html, pattern=pattern, top_k=top_k)
                finally:
                    driver.quit()

            items = await asyncio.to_thread(_run_parse)
            if not items:
                output_text = _t("未匹配到任何结果。", "No results matched.")
                return output_text
            output_text = json.dumps(items, ensure_ascii=False)
            return output_text
        except Exception as exc:
            logger.exception("url reg search tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("URL 正则搜索失败。", "URL regex search failed.")
            else:
                output_text = _t(
                    f"URL 正则搜索失败: {type(exc).__name__}",
                    f"URL regex search failed: {type(exc).__name__}",
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

    # 网络工具
    SearchTool(),
    URLRegSearchTool(),
    WebVisitTool(),

    # IPMI协议工具
    # TODO: IPMI协议工具待实现
] + _build_skill_tools()
