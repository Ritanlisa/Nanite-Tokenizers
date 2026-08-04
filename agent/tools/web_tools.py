from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from html import unescape
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import config
from exceptions import MCPFatalError
from mcp_client.client import get_mcp_client
from tool_usage import (
    end_current_tool_call,
    get_current_session_id,
    get_tool_usage,
    start_current_tool_call,
)

from ._common import (
    InputSugarTool,
    _bi,
    _consume_sugar_url_mark,
    _filter_completed_calls_by_scope,
    _safe_path,
    _sleep_with_jitter,
    _strip_html,
    _t,
)
from .rag_tools import _HIDDEN_LINK_STORE

logger = logging.getLogger(__name__)

def _extract_domain(url: str) -> str:
    try:
        netloc = (urlsplit(url).netloc or "").strip().lower()
    except Exception:
        netloc = ""
    if not netloc:
        return ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _mask_result_url(
    item: dict[str, str],
    call_index: int | None = None,
    result_index: int | None = None,
) -> dict[str, str]:
    full_url = str(item.get("link") or "").strip()
    domain = _extract_domain(full_url)
    domain_text = domain or _t("未知域名", "unknown-domain")
    sugar_expr = ""
    if call_index is not None and result_index is not None:
        sugar_expr = f"tool[{call_index}][{result_index}][link]"
    if sugar_expr:
        masked_link = _t(
            f"来自于 {domain_text} ，使用{sugar_expr}在工具调用中使用此URL",
            f"From {domain_text}, use {sugar_expr} to use this URL in tool calls",
        )
    else:
        masked_link = _t(
            f"来自于 {domain_text} ，使用tool[调用序号][结果序号][link]在工具调用中使用此URL",
            f"From {domain_text}, use tool[call_index][result_index][link] to use this URL in tool calls",
        )
    return {
        "title": item.get("title") or domain or full_url,
        "link": masked_link,
        "snippet": item.get("snippet") or "",
    }


def _predict_next_tool_call_index() -> int:
    usage = get_tool_usage(get_current_session_id())
    calls_obj = usage.get("calls") if isinstance(usage, dict) else []
    calls: list[Any] = calls_obj if isinstance(calls_obj, list) else []
    completed_count = len(_filter_completed_calls_by_scope(calls))
    return max(1, completed_count + 1)


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


_DOMAIN_OR_IP_PATTERN = re.compile(
    r"^(?:"
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    r"|(?:\d{1,3}\.){3}\d{1,3}"
    r")"
    r"(?::\d+)?"
    r"(?:/[^\s]*)?"
    r"$"
)


def _normalize_url_with_warning(raw_url: str) -> tuple[str, str]:
    candidate = (raw_url or "").strip()
    if not candidate:
        return "", ""
    if re.match(r"^https?://", candidate, re.I):
        normalized = candidate
        if _consume_sugar_url_mark(normalized):
            return normalized, ""
        warning = _t(
            "⚠️ 检测到直接传入 URL（未通过 tool[i]... 语法糖引用）。建议使用语法糖以避免 URL 明文泄露。",
            "⚠️ Direct URL input detected (not via tool[i]... sugar reference). Prefer sugar references to avoid exposing raw URLs.",
        )
        return normalized, warning
    if _DOMAIN_OR_IP_PATTERN.match(candidate):
        normalized = f"https://{candidate}"
        if _consume_sugar_url_mark(normalized):
            return normalized, ""
        warning = _t(
            "⚠️ 检测到直接传入 URL（未通过 tool[i]... 语法糖引用）。建议使用语法糖以避免 URL 明文泄露。",
            "⚠️ Direct URL input detected (not via tool[i]... sugar reference). Prefer sugar references to avoid exposing raw URLs.",
        )
        return normalized, warning
    return candidate, ""

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

class FetchWebpageInput(BaseModel):
    url: str = Field(description=_bi(
        "要抓取的网页 URL，建议使用引用语法，如 tool[-1][1][link]，而非直接复制长URL。", 
        "Webpage URL to fetch, it's recommended to use reference syntax like tool[-1][1][link] instead of directly copying long URLs."
        ))


class SeleniumWebVisitInput(BaseModel):
    url: str = Field(description=_bi(
        "要抓取的网页 URL，建议使用引用语法，如 tool[-1][1][link]，而非直接复制长URL。", 
        "Webpage URL to fetch, it's recommended to use reference syntax like tool[-1][1][link] instead of directly copying long URLs."
        ))
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


class FetchWebpageTool(InputSugarTool):
    name: str = "fetch_webpage"
    description: str = _bi("抓取指定网页 URL 的内容。", "Fetch content from a specified webpage URL.")
    args_schema: Any = FetchWebpageInput

    @retry(
        stop=stop_after_attempt(config.settings.MCP_RETRY_TIMES),
        wait=wait_exponential(multiplier=config.settings.MCP_RETRY_DELAY),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
    )
    async def _arun(self, url: str) -> str:
        started_at = time.perf_counter()
        call_id = start_current_tool_call(self.name, {"url": url})
        output_text = ""
        client = get_mcp_client()
        try:
            target_url, warning = _normalize_url_with_warning(url)
            if not target_url:
                output_text = _t("缺少 URL 参数。", "Missing URL argument.")
                return output_text
            if not re.match(r"^https?://", target_url, re.I):
                output_text = _t("仅支持 http/https URL。", "Only http/https URLs are supported.")
                return output_text

            fetch_started_at = time.perf_counter()
            fetched = await client.fetch(target_url)
            fetch_elapsed_ms = (time.perf_counter() - fetch_started_at) * 1000.0

            max_chars = int(getattr(config.settings, "FETCH_WEBPAGE_MAX_CHARS", 16000) or 16000)
            max_chars = max(1000, min(max_chars, 200000))
            original_len = len(fetched or "")
            if original_len > max_chars:
                fetched = (
                    f"{(fetched or '')[:max_chars]}\n\n---\n"
                    + _t(
                        f"⚠️ 页面内容已截断：原始 {original_len} 字符，保留前 {max_chars} 字符以提升工具与模型处理速度。",
                        f"⚠️ Web content truncated: original {original_len} chars, kept first {max_chars} chars to improve tool/model latency.",
                    )
                )

            logger.info(
                "fetch_webpage done: url=%s fetch_ms=%.1f total_ms=%.1f len=%s->%s",
                target_url,
                fetch_elapsed_ms,
                (time.perf_counter() - started_at) * 1000.0,
                original_len,
                len(fetched or ""),
            )
            output_text = f"{warning}\n\n{fetched}" if warning else fetched
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


class WebVisitTool(InputSugarTool):
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
            target_url, warning = _normalize_url_with_warning(url)
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
                    return f"```html\n{text[:max_chars]}\n```"
                finally:
                    driver.quit()

            visited_text = await asyncio.to_thread(_visit)
            if not visited_text:
                output_text = _t("页面内容为空或未提取到文本。", "Page content is empty or no text extracted.")
            elif warning:
                output_text = f"{warning}\n\n{visited_text}"
            else:
                output_text = visited_text
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

class SearchInput(BaseModel):
    query: str = Field(description=_bi("搜索关键词", "Search keywords"))
    top_k: int = Field(default=5, ge=1, le=10, description=_bi("返回结果数量", "Number of results to return"))


class URLRegSearchInput(BaseModel):
    url: str = Field(description=_bi(
        "要访问并解析的 URL，建议使用引用语法，如 tool[-1][1][link]，而非直接复制长URL。", 
        "URL to access and parse, it's recommended to use reference syntax like tool[-1][1][link] instead of directly copying long URLs."
        ))
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

class SearchTool(InputSugarTool):
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
            _HIDDEN_LINK_STORE[call_id] = [str(item.get("link") or "") for item in items]
            call_index = _predict_next_tool_call_index()
            masked_items = [
                _mask_result_url(item, call_index=call_index, result_index=idx)
                for idx, item in enumerate(items, start=1)
            ]
            output_text = json.dumps(masked_items, ensure_ascii=False)
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


class URLRegSearchTool(InputSugarTool):
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

            target_url, warning = _normalize_url_with_warning(url)
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
            _HIDDEN_LINK_STORE[call_id] = [str(item.get("link") or "") for item in items]
            call_index = _predict_next_tool_call_index()
            masked_items = [
                _mask_result_url(item, call_index=call_index, result_index=idx)
                for idx, item in enumerate(items, start=1)
            ]
            if warning:
                payload = {
                    "warning": warning,
                    "results": masked_items,
                }
                output_text = json.dumps(payload, ensure_ascii=False)
            else:
                output_text = json.dumps(masked_items, ensure_ascii=False)
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
