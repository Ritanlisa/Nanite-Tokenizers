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


def _default_value_from_annotation(annotation: Any) -> Any:
    text = str(annotation)
    if "int" in text:
        return 1
    if "float" in text:
        return 1.0
    if "bool" in text:
        return False
    if "list" in text:
        return []
    if "dict" in text:
        return {}
    return "test"


def _build_generic_kwargs(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict):
        return {}
    result: dict[str, Any] = {}
    for field_name, field in fields.items():
        default = getattr(field, "default", None)
        if default is not None and str(default) != "PydanticUndefined":
            result[field_name] = default
            continue
        annotation = getattr(field, "annotation", str)
        result[field_name] = _default_value_from_annotation(annotation)
    return result


async def _call_tool(tool: Any, kwargs: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    started = time.time()
    try:
        result = await asyncio.wait_for(tool._arun(**kwargs), timeout=timeout)
        return {
            "ok": True,
            "latency_ms": int((time.time() - started) * 1000),
            "kwargs": kwargs,
            "result_preview": str(result)[:300],
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": int((time.time() - started) * 1000),
            "kwargs": kwargs,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _call_tool_text(tool: Any, kwargs: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
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


def _parse_json_safely(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


async def _run_rag_focus_scenarios(tools_by_name: dict[str, list[Any]], first_doc: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    rag_list_tool = tools_by_name.get("rag_doc_list", [None])[0]
    rag_catalog_tool = tools_by_name.get("rag_doc_catalog", [None])[0]
    rag_regex_tool = tools_by_name.get("rag_regex_search", [None])[0]
    rag_vector_tool = tools_by_name.get("rag_vector_search", [None])[0]

    if not all([rag_list_tool, rag_catalog_tool, rag_regex_tool, rag_vector_tool]):
        return [{"scenario": "rag_tools_presence", "ok": False, "error": "missing rag tools"}]

    list_result = await _call_tool_text(rag_list_tool, {})
    list_data = _parse_json_safely(list_result.get("text", "")) if list_result.get("ok") else None
    doc_count = len(list_data) if isinstance(list_data, list) else 0
    reports.append(
        {
            "scenario": "rag_doc_list_full",
            "ok": bool(list_result.get("ok")),
            "doc_count": doc_count,
            "result_preview": str(list_result.get("text", ""))[:300],
        }
    )

    if not first_doc:
        reports.append({"scenario": "rag_doc_variants", "ok": False, "error": "no document available in rag-db"})
        return reports

    full_path = os.path.abspath(os.path.join("/", first_doc.lstrip("/")))
    relative_path = first_doc
    base_name = os.path.basename(first_doc)
    no_ext = os.path.splitext(base_name)[0]

    variants = [
        ("full_path", full_path),
        ("relative_path", relative_path),
        ("base_name", base_name),
        ("no_ext", no_ext),
    ]

    for variant_name, variant_value in variants:
        catalog_result = await _call_tool_text(rag_catalog_tool, {"doc_name": variant_value})
        catalog_data = _parse_json_safely(catalog_result.get("text", "")) if catalog_result.get("ok") else None
        catalog_count = len(catalog_data) if isinstance(catalog_data, list) else 0
        reports.append(
            {
                "scenario": f"rag_doc_catalog::{variant_name}",
                "ok": bool(catalog_result.get("ok")),
                "doc_name": variant_value,
                "count": catalog_count,
                "result_preview": str(catalog_result.get("text", ""))[:280],
            }
        )

        regex_result = await _call_tool_text(
            rag_regex_tool,
            {
                "regex": "(维护|系统|chapter|section)",
                "doc_name": variant_value,
                "limit": 3,
            },
        )
        regex_data = _parse_json_safely(regex_result.get("text", "")) if regex_result.get("ok") else None
        regex_count = int(regex_data.get("count", 0)) if isinstance(regex_data, dict) else 0
        reports.append(
            {
                "scenario": f"rag_regex_search::{variant_name}",
                "ok": bool(regex_result.get("ok")),
                "doc_name": variant_value,
                "count": regex_count,
                "result_preview": str(regex_result.get("text", ""))[:280],
            }
        )

        vector_result = await _call_tool_text(
            rag_vector_tool,
            {
                "query": "硬件维护手册 系统组成 功能说明",
                "doc_name": variant_value,
                "limit": 3,
            },
        )
        vector_data = _parse_json_safely(vector_result.get("text", "")) if vector_result.get("ok") else None
        vector_count = int(vector_data.get("count", 0)) if isinstance(vector_data, dict) else 0
        reports.append(
            {
                "scenario": f"rag_vector_search::{variant_name}",
                "ok": bool(vector_result.get("ok")),
                "doc_name": variant_value,
                "count": vector_count,
                "result_preview": str(vector_result.get("text", ""))[:280],
            }
        )

    return reports


async def _run_mcp_scenarios() -> list[dict[str, Any]]:
    from mcp_client.client import get_mcp_client

    client = get_mcp_client()
    await client.initialize()
    scenarios: list[dict[str, Any]] = [
        {"name": "fetch_ok", "url": "https://www.bing.com/search?q=Nanite+Tokenizers", "timeout": 15},
        {"name": "fetch_404", "url": "https://www.bing.com/this-page-should-not-exist-404", "timeout": 15},
        {"name": "fetch_bad_domain", "url": "https://nonexistent.nanite-tokenizers.invalid", "timeout": 10},
        {"name": "fetch_timeout", "url": "https://10.255.255.1", "timeout": 2},
    ]
    reports: list[dict[str, Any]] = []

    for scenario in scenarios:
        started = time.time()
        try:
            text = await client.fetch(scenario["url"], timeout=float(scenario["timeout"]))
            reports.append(
                {
                    "scenario": scenario["name"],
                    "ok": True,
                    "url": scenario["url"],
                    "timeout": scenario["timeout"],
                    "latency_ms": int((time.time() - started) * 1000),
                    "result_preview": str(text)[:300],
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "scenario": scenario["name"],
                    "ok": False,
                    "url": scenario["url"],
                    "timeout": scenario["timeout"],
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    reports.append(
        {
            "scenario": "client_state",
            "ok": True,
            "fallback_mode": bool(getattr(client, "_fallback_mode", False)),
            "initialized": bool(getattr(client, "_initialized", False)),
            "restart_count": int(getattr(client, "_restart_count", 0)),
        }
    )

    try:
        await client.close()
        reports.append({"scenario": "client_close", "ok": True})
    except Exception as exc:
        reports.append({"scenario": "client_close", "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return reports


async def main() -> None:
    import config
    from agent.tools import rag_engine, tools

    config.settings = config.settings.update(RAG_DB_NAME="rag-db", RAG_DB_NAMES=["rag-db"])

    docs = rag_engine.list_documents()
    first_doc = docs[0]["title"] if docs else ""

    plan: list[tuple[str, dict[str, Any]]] = [
        ("rag_doc_list", {}),
        ("rag_doc_catalog", {"doc_name": first_doc or "README.md"}),
        (
            "rag_regex_search",
            {
                "regex": "(error|traceback|section|chapter)",
                "doc_name": first_doc,
                "limit": 3,
            },
        ),
        (
            "rag_vector_search",
            {
                "query": "系统架构与功能说明",
                "doc_name": first_doc,
                "limit": 3,
            },
        ),
        ("math_compute", {"mode": "eval", "expression": "2+3*4"}),
        ("math_compute", {"mode": "simplify", "expression": "(x**2-1)/(x-1)"}),
        ("math_compute", {"mode": "function", "expression": "sin(x)+x**2", "variables": {"x": 1.2}}),
        ("math_compute", {"mode": "solve", "equation": "x**2-4=0", "symbol": "x"}),
        (
            "math_compute",
            {
                "mode": "matrix",
                "matrix_op": "mul",
                "matrix_a": [[1, 2], [3, 4]],
                "matrix_b": [[2, 0], [1, 2]],
            },
        ),
        ("fetch_webpage", {"url": "https://www.bing.com/search?q=Nanite+Tokenizers"}),
        ("skill_shell", {"command": "echo nanite_tools_test", "timeout": 8}),
        ("skill_file_io", {"action": "mkdir", "path": "tmp/tools-test"}),
        (
            "skill_file_io",
            {"action": "write", "path": "tmp/tools-test/sample.txt", "content": "hello"},
        ),
        ("skill_file_io", {"action": "append", "path": "tmp/tools-test/sample.txt", "content": " world"}),
        ("skill_file_io", {"action": "read", "path": "tmp/tools-test/sample.txt"}),
        ("skill_file_io", {"action": "list", "path": "tmp/tools-test"}),
        ("skill_search", {"query": "Nanite Tokenizers", "top_k": 2}),
    ]

    tools_by_name: dict[str, list[Any]] = {}
    for tool in tools:
        tools_by_name.setdefault(tool.name, []).append(tool)

    reports: list[dict[str, Any]] = []

    for name, kwargs in plan:
        target = tools_by_name.get(name, [])
        if not target:
            reports.append({"tool": name, "ok": False, "error": "tool not found"})
            continue
        rep = await _call_tool(target[0], kwargs)
        rep["tool"] = name
        reports.append(rep)

    planned_names = {name for name, _ in plan}
    for tool in tools:
        if tool.name in planned_names:
            continue
        kwargs = _build_generic_kwargs(tool)
        rep = await _call_tool(tool, kwargs)
        rep["tool"] = tool.name
        rep["auto_generated_args"] = True
        reports.append(rep)

    success = sum(1 for item in reports if item.get("ok"))
    failed = len(reports) - success

    rag_focus_reports = await _run_rag_focus_scenarios(tools_by_name, first_doc)
    rag_focus_success = sum(1 for item in rag_focus_reports if item.get("ok"))
    rag_focus_failed = len(rag_focus_reports) - rag_focus_success

    mcp_reports = await _run_mcp_scenarios()
    mcp_success = sum(1 for item in mcp_reports if item.get("ok"))
    mcp_failed = len(mcp_reports) - mcp_success
    output = {
        "rag_db": config.settings.RAG_DB_NAME,
        "total": len(reports),
        "success": success,
        "failed": failed,
        "reports": reports,
        "rag_focus_note": "rag_* tools are local RAGEngine calls; they do not use MCP fetch transport.",
        "rag_focus_total": len(rag_focus_reports),
        "rag_focus_success": rag_focus_success,
        "rag_focus_failed": rag_focus_failed,
        "rag_focus_reports": rag_focus_reports,
        "mcp_total": len(mcp_reports),
        "mcp_success": mcp_success,
        "mcp_failed": mcp_failed,
        "mcp_reports": mcp_reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
