from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


FAIL_HINTS = (
    "失败",
    "failed",
    "error",
    "超时",
    "timed out",
    "不可用",
    "unavailable",
)


def _parse_json_safely(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


async def _invoke_tool(tool: Any, kwargs: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    started = time.time()
    try:
        text = await asyncio.wait_for(tool._arun(**kwargs), timeout=timeout)
        return {
            "ok": True,
            "latency_ms": int((time.time() - started) * 1000),
            "kwargs": kwargs,
            "text": text,
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": int((time.time() - started) * 1000),
            "kwargs": kwargs,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _has_any(text: str, candidates: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(token.lower() in lower for token in candidates)


def _looks_successful_text(text: str) -> bool:
    if not text:
        return False
    return not _has_any(text, FAIL_HINTS)


async def main() -> None:
    import config
    from agent.tools import tools

    config.settings = config.settings.update(ENABLE_SEARCH_SKILL=True)

    tools_by_name: dict[str, Any] = {tool.name: tool for tool in tools}

    search_tool = tools_by_name.get("skill_search")
    web_visit_tool = tools_by_name.get("skill_web_visit")
    url_reg_tool = tools_by_name.get("skill_url_reg_search")

    reports: list[dict[str, Any]] = []

    reports.append({
        "scenario": "tool_presence",
        "ok": bool(search_tool and web_visit_tool and url_reg_tool),
        "missing": [
            name
            for name, tool in (
                ("skill_search", search_tool),
                ("skill_web_visit", web_visit_tool),
                ("skill_url_reg_search", url_reg_tool),
            )
            if tool is None
        ],
    })

    if not all([search_tool, web_visit_tool, url_reg_tool]):
        print(json.dumps({"total": len(reports), "success": 0, "failed": len(reports), "reports": reports}, ensure_ascii=False, indent=2))
        return

    search_missing = await _invoke_tool(search_tool, {"query": "", "top_k": 2})
    reports.append({
        "scenario": "search_missing_query",
        "ok": bool(search_missing.get("ok")) and _has_any(str(search_missing.get("text", "")), ("缺少", "missing")),
        "response_preview": str(search_missing.get("text", ""))[:240],
    })

    search_success = await _invoke_tool(search_tool, {"query": "Nanite Tokenizers", "top_k": 2}, timeout=90)
    search_data = _parse_json_safely(str(search_success.get("text", ""))) if search_success.get("ok") else None
    search_items_ok = isinstance(search_data, list) and len(search_data) > 0
    reports.append({
        "scenario": "search_success_path",
        "ok": bool(search_success.get("ok")) and (search_items_ok or _looks_successful_text(str(search_success.get("text", "")))),
        "latency_ms": search_success.get("latency_ms"),
        "result_kind": "json_list" if search_items_ok else "text",
        "response_preview": str(search_success.get("text", ""))[:280],
    })

    web_invalid = await _invoke_tool(web_visit_tool, {"url": "ftp://example.com"})
    reports.append({
        "scenario": "web_visit_invalid_url",
        "ok": bool(web_invalid.get("ok")) and _has_any(str(web_invalid.get("text", "")), ("仅支持", "only http/https")),
        "response_preview": str(web_invalid.get("text", ""))[:240],
    })

    web_success = await _invoke_tool(web_visit_tool, {"url": "https://example.com", "max_chars": 1200}, timeout=90)
    web_text = str(web_success.get("text", ""))
    reports.append({
        "scenario": "web_visit_success_path",
        "ok": bool(web_success.get("ok")) and len(web_text.strip()) > 0 and _looks_successful_text(web_text),
        "latency_ms": web_success.get("latency_ms"),
        "response_preview": web_text[:280],
    })

    reg_invalid_regex = await _invoke_tool(
        url_reg_tool,
        {
            "url": "https://example.com",
            "regex": "(?P<link>",
            "top_k": 2,
        },
    )
    reports.append({
        "scenario": "url_reg_invalid_regex",
        "ok": bool(reg_invalid_regex.get("ok")) and _has_any(str(reg_invalid_regex.get("text", "")), ("正则", "invalid regex")),
        "response_preview": str(reg_invalid_regex.get("text", ""))[:240],
    })

    reg_invalid_url = await _invoke_tool(
        url_reg_tool,
        {
            "url": "file:///tmp/example.html",
            "regex": "(?P<link>https?://[^\"']+)",
            "top_k": 2,
        },
    )
    reports.append({
        "scenario": "url_reg_invalid_url",
        "ok": bool(reg_invalid_url.get("ok")) and _has_any(str(reg_invalid_url.get("text", "")), ("仅支持", "only http/https")),
        "response_preview": str(reg_invalid_url.get("text", ""))[:240],
    })

    reg_success = await _invoke_tool(
        url_reg_tool,
        {
            "url": "https://example.com",
            "regex": "<a[^>]+href=\"(?P<link>https?://[^\"]+)\"[^>]*>(?P<title>.*?)</a>",
            "top_k": 3,
            "max_html_chars": 80000,
        },
        timeout=90,
    )
    reg_data = _parse_json_safely(str(reg_success.get("text", ""))) if reg_success.get("ok") else None
    reg_items_ok = isinstance(reg_data, list) and len(reg_data) > 0
    reports.append({
        "scenario": "url_reg_success_path",
        "ok": bool(reg_success.get("ok")) and reg_items_ok,
        "latency_ms": reg_success.get("latency_ms"),
        "result_count": len(reg_data) if isinstance(reg_data, list) else 0,
        "response_preview": str(reg_success.get("text", ""))[:280],
    })

    success = sum(1 for item in reports if item.get("ok"))
    failed = len(reports) - success

    print(
        json.dumps(
            {
                "total": len(reports),
                "success": success,
                "failed": failed,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
