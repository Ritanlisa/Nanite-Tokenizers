from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_RAG_EXTENSIONS: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".xlsx",
    ".xls",
    ".csv",
}

UPLOAD_DIR = Path("tmp") / "doc_tree_debug_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_JSON_DIR = Path("tmp") / "doc_tree_debug_json"
DEBUG_JSON_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", str(name).strip())
    return cleaned or "document"


def _json_safe(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1, max_depth=max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1, max_depth=max_depth) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value), depth=depth + 1, max_depth=max_depth)
    return str(value)


def _page_variable_snapshot(page: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any]
    try:
        attrs = dict(vars(page))
    except Exception:
        attrs = {}

    snapshot: Dict[str, Any] = {
        "class_name": str(type(page).__name__),
        "category": str(getattr(page, "category", "") or ""),
        "title": str(getattr(page, "title", "") or ""),
        "markdown_text": str(getattr(page, "markdown_text", "") or ""),
        "metadata": _json_safe(getattr(page, "metadata", {}) or {}),
        "assets": _json_safe(getattr(page, "assets", None)),
        "to_payload": _json_safe(page.to_payload()) if hasattr(page, "to_payload") else {},
    }

    for key, value in attrs.items():
        if key == "SubContent":
            snapshot[key] = f"<{len(value) if isinstance(value, list) else 0} children>"
            continue
        snapshot[key] = _json_safe(value)
    return snapshot


def _serialize_page_node(page: Any, node_id_prefix: str = "node") -> Dict[str, Any]:
    title = str(getattr(page, "title", "") or "")
    category = str(getattr(page, "category", "") or "unknown")
    children = list(getattr(page, "SubContent", []) or [])

    node: Dict[str, Any] = {
        "id": f"{node_id_prefix}:{id(page)}",
        "title": title,
        "category": category,
        "class_name": type(page).__name__,
        "metadata": _json_safe(getattr(page, "metadata", {}) or {}),
        "variables": _page_variable_snapshot(page),
        "children": [],
    }
    for child in children:
        node["children"].append(_serialize_page_node(child, node_id_prefix=node["id"]))
    return node


def _catalog_tree_lines(nodes: list[Dict[str, Any]], *, indent: int = 0) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        meta = dict(node.get("metadata") or {})
        start = meta.get("section_start_page") or meta.get("page") or "?"
        end = meta.get("section_end_page") or start
        cls = str(node.get("class_name") or "Page")
        title = str(node.get("title") or "")
        lines.append(f"{'  ' * indent}- [{cls}] {title} ({start}-{end})")
        children = node.get("children") or []
        if isinstance(children, list) and children:
            lines.extend(_catalog_tree_lines(children, indent=indent + 1))
    return lines


def _first_nonempty_line(text: Any) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:180]
    return ""


def _last_nonempty_line(text: Any) -> str:
    lines = str(text or "").splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:180]
    return ""


def _display_page_value(node: Dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    hint = str(metadata.get("page_number_hint") or "").strip()
    if hint:
        return hint
    logical = metadata.get("logical_page")
    if logical not in (None, ""):
        return str(logical)
    page = metadata.get("page")
    return str(page) if page not in (None, "") else ""


def _collect_content_nodes(node: Dict[str, Any], out: list[Dict[str, Any]]) -> None:
    if str(node.get("category") or "") == "content":
        out.append(node)
    for child in list(node.get("children") or []):
        _collect_content_nodes(child, out)


def _collect_chapter_boundary_debug(tree: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    chapter_rows: list[Dict[str, Any]] = []

    def _walk(nodes: list[Dict[str, Any]]) -> None:
        for node in nodes:
            if str(node.get("category") or "") == "chapter":
                content_nodes: list[Dict[str, Any]] = []
                _collect_content_nodes(node, content_nodes)
                if content_nodes:
                    content_nodes.sort(
                        key=lambda item: int((item.get("metadata") or {}).get("page") or 10**9)
                    )
                    first_node = content_nodes[0]
                    last_node = content_nodes[-1]
                    first_vars = dict(first_node.get("variables") or {})
                    last_vars = dict(last_node.get("variables") or {})
                    first_text = str(first_vars.get("markdown_text") or "")
                    last_text = str(last_vars.get("markdown_text") or "")

                    chapter_meta = dict(node.get("metadata") or {})
                    first_meta = dict(first_node.get("metadata") or {})
                    last_meta = dict(last_node.get("metadata") or {})

                    chapter_rows.append(
                        {
                            "chapter_title": str(node.get("title") or ""),
                            "chapter_start": chapter_meta.get("section_start_page"),
                            "chapter_end": chapter_meta.get("section_end_page"),
                            "first_actual_page": first_meta.get("page"),
                            "first_display_page": _display_page_value(first_node),
                            "first_line": _first_nonempty_line(first_text),
                            "last_actual_page": last_meta.get("page"),
                            "last_display_page": _display_page_value(last_node),
                            "last_line": _last_nonempty_line(last_text),
                        }
                    )
            _walk(list(node.get("children") or []))

    _walk(list(tree or []))
    return chapter_rows


def _build_pipeline_debug(rag_doc: Any) -> Dict[str, Any]:
    debug: Dict[str, Any] = {
        "doc_parser": str(getattr(rag_doc, "metadata", {}).get("doc_parser") or ""),
        "docx_parser": str(getattr(rag_doc, "metadata", {}).get("docx_parser") or ""),
        "native_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("native_catalog") or [])),
        "style_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("style_catalog") or [])),
        "font_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("font_catalog") or [])),
    }

    cleaned_text = str(getattr(rag_doc, "cleaned_text", "") or "")
    if not cleaned_text:
        return debug

    raw_pages = [part.strip() for part in cleaned_text.split("\f")]
    source_page_texts = [part for part in raw_pages]
    if not source_page_texts:
        source_page_texts = [cleaned_text.strip()]
    debug["source_page_count"] = len(source_page_texts)

    if hasattr(rag_doc, "enforce_native_page_count"):
        page_texts = list(getattr(rag_doc, "enforce_native_page_count")(source_page_texts))
    else:
        page_texts = list(source_page_texts)
    debug["target_page_count"] = len(page_texts)

    if hasattr(rag_doc, "_extract_structured_catalog_ranges"):
        try:
            source_ranges = getattr(rag_doc, "_extract_structured_catalog_ranges")(
                source_page_texts,
                len(source_page_texts),
                len(source_page_texts),
            )
            debug["source_ranges"] = _json_safe(source_ranges)
        except Exception as exc:
            debug["source_ranges_error"] = str(exc)

    if hasattr(rag_doc, "_extract_main_compatible_markers"):
        try:
            main_markers = getattr(rag_doc, "_extract_main_compatible_markers")(page_texts)
            debug["main_markers"] = _json_safe(main_markers)
        except Exception as exc:
            debug["main_markers_error"] = str(exc)

    if hasattr(rag_doc, "_ranges_from_main_markers") and "main_markers" in debug:
        try:
            main_ranges = getattr(rag_doc, "_ranges_from_main_markers")(list(debug.get("main_markers") or []), len(page_texts))
            debug["main_ranges"] = _json_safe(main_ranges)
        except Exception as exc:
            debug["main_ranges_error"] = str(exc)

    if hasattr(rag_doc, "_extract_main_page_section_map"):
        try:
            main_section_map = getattr(rag_doc, "_extract_main_page_section_map")(page_texts)
            debug["main_section_map"] = _json_safe(main_section_map)
            if hasattr(rag_doc, "_ranges_from_page_section_map"):
                debug["main_section_ranges"] = _json_safe(
                    getattr(rag_doc, "_ranges_from_page_section_map")(main_section_map, len(page_texts))
                )
        except Exception as exc:
            debug["main_section_map_error"] = str(exc)

    return debug


def _build_payload_from_file(file_path: str, *, include_build_debug: bool = False) -> Dict[str, Any]:
    from rag.documents import load_rag_documents_from_paths, _probe_libreoffice_physical_page_count

    source_file = Path(file_path).expanduser().resolve()
    if not source_file.exists() or not source_file.is_file():
        raise FileNotFoundError(f"文件不存在: {source_file}")

    ext = source_file.suffix.lower()
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        raise RuntimeError(f"不支持的文档类型: {ext}")

    rag_docs = load_rag_documents_from_paths([str(source_file)], SUPPORTED_RAG_EXTENSIONS)
    if not rag_docs:
        raise RuntimeError("文档树构建失败：未返回 RAG 文档对象")

    rag_doc = rag_docs[0]
    page_nodes = list(rag_doc.get_page_nodes() or [])
    tree = [_serialize_page_node(node, node_id_prefix="root") for node in page_nodes]
    mono_pages = list(getattr(rag_doc, "get_mono_pages")() or []) if hasattr(rag_doc, "get_mono_pages") else []
    mono_page_count = len(mono_pages)
    native_page_count_raw = getattr(rag_doc, "metadata", {}).get("native_page_count") if hasattr(rag_doc, "metadata") else None
    native_page_count = int(native_page_count_raw or 0) if str(native_page_count_raw or "").strip() else 0
    runtime_native_page_count = int(_probe_libreoffice_physical_page_count(str(source_file)) or 0)
    page_count = int(getattr(rag_doc, "page_count", 0) or 0)
    expected_pages = runtime_native_page_count or native_page_count
    is_aligned = bool(expected_pages > 0 and mono_page_count == expected_pages and page_count == expected_pages)
    print(f"\n[DocTreeDebug] Structured catalog for: {source_file.name}")
    for line in _catalog_tree_lines(tree):
      print(line)

    chapter_boundary_debug = _collect_chapter_boundary_debug(tree)
    if chapter_boundary_debug:
      print("\n[DocTreeDebug] Chapter Boundary Diagnostics")
      for row in chapter_boundary_debug:
        print(
          "- "
          f"{row.get('chapter_title')} "
          f"({row.get('chapter_start')}-{row.get('chapter_end')}) | "
          f"first: actual={row.get('first_actual_page')} display={row.get('first_display_page')} "
          f"line={row.get('first_line')} | "
          f"last: actual={row.get('last_actual_page')} display={row.get('last_display_page')} "
          f"line={row.get('last_line')}"
        )

    payload = {
      "status": "ok",
      "document": source_file.name,
      "doc_name": str(getattr(rag_doc, "doc_name", "") or source_file.name),
      "title": str(getattr(rag_doc, "title", "") or source_file.name),
      "pagination_mode": str(getattr(rag_doc, "pagination_mode", "") or ""),
      "page_count": page_count,
      "mono_page_count": mono_page_count,
      "native_page_count": native_page_count,
      "runtime_native_page_count": runtime_native_page_count,
      "expected_page_count": expected_pages,
      "is_page_aligned": is_aligned,
      "catalog": _json_safe(getattr(rag_doc, "catalog", []) or []),
      "variables": _page_variable_snapshot(rag_doc),
      "tree": tree,
      "chapter_boundary_debug": chapter_boundary_debug,
    }
    if include_build_debug:
      payload["build_debug"] = _build_pipeline_debug(rag_doc)

    safe_name = _sanitize_filename(source_file.stem)
    out_path = DEBUG_JSON_DIR / f"{safe_name}_{uuid.uuid4().hex[:8]}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["debug_json_path"] = str(out_path)
    print(f"[DocTreeDebug] JSON sidecar: {out_path}")
    print(
      "[DocTreeDebug] Alignment: "
      f"page_count={page_count}, mono_page_count={mono_page_count}, "
      f"native_page_count={native_page_count}, runtime_native_page_count={runtime_native_page_count}, "
      f"expected_page_count={expected_pages}, aligned={is_aligned}"
    )

    return payload


def _save_upload(upload: UploadFile) -> Path:
    filename = _sanitize_filename(upload.filename or "uploaded")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    saved = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    content = upload.file.read()
    saved.write_bytes(content)
    return saved


def _build_page_html() -> str:
    return """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Doc Tree Debug</title>
  <style>
    :root {
      --bg: #0f1412;
      --bg2: #1a231f;
      --panel: #202d27;
      --line: rgba(235, 244, 239, 0.15);
      --text: #ebf4ef;
      --muted: #9eb3a6;
      --accent: #ffd166;
      --accent2: #4cc9a6;
      --err: #ff7b72;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% -8%, rgba(76, 201, 166, 0.18), transparent 42%),
        radial-gradient(circle at 90% 0%, rgba(255, 209, 102, 0.16), transparent 40%),
        linear-gradient(180deg, var(--bg2), var(--bg));
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      min-height: 100vh;
    }
    .page { max-width: 1500px; margin: 0 auto; padding: 16px; }
    .bar {
      border: 1px solid var(--line);
      background: rgba(10, 16, 13, 0.55);
      border-radius: 12px;
      padding: 10px;
      margin-bottom: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; }
    input[type=file], button {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.22);
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      color: #0f1e16;
      border: none;
      background: linear-gradient(130deg, var(--accent), var(--accent2));
      font-weight: 700;
    }
    .status { color: var(--muted); min-height: 22px; }
    .status.error { color: var(--err); }
    .meta {
      border: 1px solid var(--line);
      background: rgba(10, 16, 13, 0.55);
      border-radius: 12px;
      padding: 10px;
      color: var(--muted);
      margin-bottom: 12px;
      white-space: pre-wrap;
    }
    .layout { display: grid; grid-template-columns: minmax(320px, 36%) 1fr; gap: 12px; }
    .panel { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); overflow: hidden; min-height: 72vh; }
    .head { padding: 10px 12px; color: var(--muted); border-bottom: 1px solid var(--line); }
    .body { padding: 10px; max-height: calc(72vh - 44px); overflow: auto; }
    .tree, .tree ul { list-style: none; margin: 0; padding-left: 16px; }
    .tree li { margin: 6px 0; }
    .node-row { display: flex; align-items: center; gap: 6px; }
    .toggle {
      width: 24px;
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.08);
      color: var(--text);
      cursor: pointer;
      flex-shrink: 0;
    }
    .toggle.placeholder {
      visibility: hidden;
      cursor: default;
    }
    .node {
      width: 100%; text-align: left;
      border: 1px solid transparent;
      border-radius: 9px;
      padding: 8px;
      color: var(--text);
      background: rgba(255,255,255,0.05);
      cursor: pointer;
    }
    .node.active { border-color: rgba(76, 201, 166, 0.9); }
    .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .badge {
      display: inline-block;
      margin-left: 8px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 11px;
      border: 1px solid var(--line);
      color: var(--muted);
    }
    .badge.chapter { color: #8bd3ff; }
    .badge.mono { color: #ffd166; }
    .vars { display: grid; grid-template-columns: 1fr; gap: 10px; }
    .assets { display: grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 10px; }
    .asset-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: rgba(10, 16, 13, 0.55);
    }
    .asset-title {
      font-size: 12px;
      color: var(--accent2);
      margin-bottom: 6px;
      font-weight: 700;
    }
    .asset-empty {
      color: var(--muted);
      font-size: 12px;
    }
    .asset-list {
      margin: 0;
      padding-left: 16px;
      color: var(--text);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .asset-kv {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 6px;
      font-size: 12px;
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .asset-k {
      color: var(--muted);
    }
    .card { border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: rgba(10, 16, 13, 0.55); }
    .name { font-size: 12px; color: var(--accent); margin-bottom: 6px; }
    textarea {
      width: 100%; min-height: 120px; resize: vertical;
      border: 1px solid var(--line); border-radius: 8px;
      background: rgba(0,0,0,0.2); color: var(--text);
      padding: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px;
    }
    @media (max-width: 980px) {
      .bar { grid-template-columns: 1fr; }
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"page\">
    <div class=\"bar\">
      <div class=\"controls\">
        <input id=\"fileInput\" type=\"file\" />
        <button id=\"buildBtn\" type=\"button\">构建文档树</button>
      </div>
      <div id=\"status\" class=\"status\">请选择文件后构建</div>
    </div>

    <div class=\"meta\" id=\"meta\">尚未构建文档树</div>

    <div class="layout">
      <section class="panel">
        <div class="head">文档树结构</div>
        <div class="body"><ul class="tree" id="treeRoot"></ul></div>
      </section>
      <section class="panel">
        <div class="head">Page 子页详情（含资产）</div>
        <div class="body">
          <div class="assets" id="assets"></div>
          <div class="vars" id="vars"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const buildBtn = document.getElementById('buildBtn');
    const statusEl = document.getElementById('status');
    const treeRoot = document.getElementById('treeRoot');
    const assetsEl = document.getElementById('assets');
    const varsEl = document.getElementById('vars');
    const meta = document.getElementById('meta');

    let payload = null;
    let selectedNodeId = '';
    let buildTimer = null;
    let buildStartTs = 0;
    const collapsedChapterIds = new Set();

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? 'status error' : 'status';
    }

    function pretty(v) {
      if (typeof v === 'string') {
        return v
          .replace(/\\\\n/g, '\\n')
          .replace(/\\\\t/g, '\\t')
          .replace(/\\\\r/g, '\\r');
      }
      let text = '';
      try {
        text = JSON.stringify(v, null, 2);
      } catch (_) {
        text = String(v);
      }
      return String(text)
        .replace(/\\\\n/g, '\\n')
        .replace(/\\\\t/g, '\\t')
        .replace(/\\\\r/g, '\\r');
    }

    window.addEventListener('error', (event) => {
      setStatus(`页面脚本错误: ${event.message || 'unknown error'}`, true);
    });

    function collect(nodes, out = []) {
      for (const n of nodes || []) {
        out.push(n);
        collect(n.children || [], out);
      }
      return out;
    }

    function isMonoPageNode(node) {
      if (!node || typeof node !== 'object') return false;
      const cls = String(node.class_name || '');
      if (['MonoPage', 'Cover', 'Catalogue', 'Introduction', 'Content', 'Appendix'].includes(cls)) {
        return true;
      }
      const category = String(node.category || '').toLowerCase();
      const children = Array.isArray(node.children) ? node.children : [];
      return category !== 'chapter' && children.length === 0;
    }

    function toList(value) {
      if (Array.isArray(value)) return value.filter(v => String(v || '').trim() !== '');
      if (value === null || value === undefined) return [];
      const text = String(value).trim();
      return text ? [text] : [];
    }

    function renderAssetList(title, items) {
      const card = document.createElement('div');
      card.className = 'asset-card';
      const head = document.createElement('div');
      head.className = 'asset-title';
      head.textContent = title;
      card.appendChild(head);

      const list = toList(items);
      if (list.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'asset-empty';
        empty.textContent = '无';
        card.appendChild(empty);
        return card;
      }

      const ul = document.createElement('ul');
      ul.className = 'asset-list';
      for (const item of list) {
        const li = document.createElement('li');
        li.textContent = String(item);
        ul.appendChild(li);
      }
      card.appendChild(ul);
      return card;
    }

    function renderAssets(node) {
      if (!assetsEl) {
        return;
      }
      assetsEl.innerHTML = '';
      if (!node || typeof node !== 'object') {
        return;
      }

      const vars = node.variables && typeof node.variables === 'object' ? node.variables : {};
      const payload = vars.to_payload && typeof vars.to_payload === 'object' ? vars.to_payload : {};
      const metadata = node.metadata && typeof node.metadata === 'object' ? node.metadata : {};

      const headers = toList(payload.headers || metadata.headers || metadata.header_text);
      const footers = toList(payload.footers || metadata.footers || metadata.footer_text);
      const pageNumbers = toList(payload.page_numbers || metadata.page_number_hint || metadata.page);
      const images = toList(payload.images || metadata.images);

      const summary = document.createElement('div');
      summary.className = 'asset-card';
      const summaryTitle = document.createElement('div');
      summaryTitle.className = 'asset-title';
      summaryTitle.textContent = '资产统计';
      summary.appendChild(summaryTitle);
      const kv = document.createElement('div');
      kv.className = 'asset-kv';
      const rows = [
        ['Headers', String(headers.length)],
        ['Footers', String(footers.length)],
        ['Page No.', String(pageNumbers.length)],
        ['Images', String(images.length || Number(metadata.image_count || 0))],
      ];
      for (const [k, v] of rows) {
        const kEl = document.createElement('div');
        kEl.className = 'asset-k';
        kEl.textContent = k;
        const vEl = document.createElement('div');
        vEl.textContent = v;
        kv.appendChild(kEl);
        kv.appendChild(vEl);
      }
      summary.appendChild(kv);

      assetsEl.appendChild(summary);
      assetsEl.appendChild(renderAssetList('页眉 Header', headers));
      assetsEl.appendChild(renderAssetList('页脚 Footer', footers));
      assetsEl.appendChild(renderAssetList('页码 Page Number', pageNumbers));
      assetsEl.appendChild(renderAssetList('图片 Images', images));
    }

    function renderVars(node) {
      if (!varsEl) {
        return;
      }
      varsEl.innerHTML = '';
      const vars = node && node.variables && typeof node.variables === 'object' ? node.variables : {};
      for (const [k, v] of Object.entries(vars)) {
        const card = document.createElement('div');
        card.className = 'card';
        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = k;
        const box = document.createElement('textarea');
        box.readOnly = true;
        box.value = pretty(v);
        card.appendChild(name);
        card.appendChild(box);
        varsEl.appendChild(card);
      }
    }

    function renderTree(nodes) {
      if (!treeRoot) {
        return;
      }
      treeRoot.innerHTML = '';
      function build(items) {
        const ul = document.createElement('ul');
        ul.className = 'tree';
        for (const n of items || []) {
          const li = document.createElement('li');
          const row = document.createElement('div');
          row.className = 'node-row';
          const children = Array.isArray(n.children) ? n.children : [];
          const isChapter = String(n.category || '').toLowerCase() === 'chapter';

          const toggle = document.createElement('button');
          toggle.type = 'button';
          toggle.className = 'toggle';
          if (!isChapter || children.length === 0) {
            toggle.classList.add('placeholder');
            toggle.textContent = '·';
            toggle.disabled = true;
          } else {
            const collapsed = collapsedChapterIds.has(String(n.id || ''));
            toggle.textContent = collapsed ? '▸' : '▾';
            toggle.title = collapsed ? '展开 Chapter' : '折叠 Chapter';
            toggle.onclick = (event) => {
              event.stopPropagation();
              const nodeId = String(n.id || '');
              if (collapsedChapterIds.has(nodeId)) {
                collapsedChapterIds.delete(nodeId);
              } else {
                collapsedChapterIds.add(nodeId);
              }
              renderTree(payload ? (payload.tree || []) : []);
            };
          }

          const btn = document.createElement('button');
          btn.className = 'node' + (selectedNodeId === n.id ? ' active' : '');
          const sub = document.createElement('div');
          sub.className = 'sub';
          const badge = isChapter
            ? '<span class="badge chapter">Chapter</span>'
            : (isMonoPageNode(n) ? '<span class="badge mono">MonoPage</span>' : '');
          btn.innerHTML = `<div>${n.title || 'Untitled'} ${badge}</div>`;
          sub.textContent = `${n.class_name || 'Page'} | ${n.category || 'unknown'}`;
          btn.appendChild(sub);
          btn.onclick = () => {
            selectedNodeId = n.id;
            renderTree(payload.tree || []);
            renderAssets(n);
            renderVars(n);
          };

          row.appendChild(toggle);
          row.appendChild(btn);
          li.appendChild(row);

          if (children.length > 0 && !collapsedChapterIds.has(String(n.id || ''))) {
            li.appendChild(build(children));
          }
          ul.appendChild(li);
        }
        return ul;
      }
      treeRoot.appendChild(build(nodes || []));
    }

    function beginBuildUi(fileName) {
      buildStartTs = Date.now();
      buildBtn.disabled = true;
      buildBtn.textContent = '构建中...';
      setStatus(`正在构建: ${fileName}（文档较大时可能需要几十秒）`);
      if (buildTimer) {
        clearInterval(buildTimer);
        buildTimer = null;
      }
      buildTimer = setInterval(() => {
        const secs = Math.max(0, Math.floor((Date.now() - buildStartTs) / 1000));
        setStatus(`正在构建: ${fileName}（已耗时 ${secs}s）`);
      }, 1000);
    }

    function endBuildUi() {
      if (buildTimer) {
        clearInterval(buildTimer);
        buildTimer = null;
      }
      buildBtn.disabled = false;
      buildBtn.textContent = '构建文档树';
    }

    async function buildDocTree() {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus('请先选择文件', true);
        return;
      }

      beginBuildUi(file.name || '已选文件');
      const form = new FormData();
      form.append('file', file);

      const resp = await fetch('/api/build', { method: 'POST', body: form });
      const raw = await resp.text();
      let data = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch (_) {
        data = { detail: raw || '服务返回了非 JSON 响应' };
      }
      if (!resp.ok) {
        throw new Error(String(data.detail || '构建失败'));
      }

      payload = data;
      const nodes = collect(payload.tree || []);
      collapsedChapterIds.clear();
      const aligned = Boolean(payload.is_page_aligned);
      meta.textContent = [
        `doc=${payload.document || ''}`,
        `title=${payload.title || ''}`,
        `doc_name=${payload.doc_name || ''}`,
        `page_count=${payload.page_count || 0}`,
        `mono_pages=${payload.mono_page_count || 0}`,
        `native_pages=${payload.native_page_count || 0}`,
        `runtime_native_pages=${payload.runtime_native_page_count || 0}`,
        `expected_pages=${payload.expected_page_count || 0}`,
        `aligned=${aligned ? 'yes' : 'no'}`,
        `pagination=${payload.pagination_mode || ''}`,
        `nodes=${nodes.length}`
      ].join(' | ');
      if (!aligned && Number(payload.native_page_count || 0) > 0) {
        setStatus(`页数未对齐: mono=${payload.mono_page_count || 0}, expected=${payload.expected_page_count || payload.native_page_count || 0}`, true);
      }

      selectedNodeId = '';
      renderTree(payload.tree || []);
      if (nodes.length > 0) {
        selectedNodeId = nodes[0].id;
        renderTree(payload.tree || []);
        renderAssets(nodes[0]);
        renderVars(nodes[0]);
      } else {
        if (assetsEl) assetsEl.innerHTML = '';
        if (varsEl) varsEl.innerHTML = '';
      }
      if (aligned) {
        setStatus(`构建完成，页数已对齐（mono=${payload.mono_page_count || 0}）`);
      } else if (Number(payload.native_page_count || 0) <= 0) {
        setStatus(`构建完成，节点数: ${nodes.length}（未获取物理页数）`);
      }
    }

    fileInput.addEventListener('change', () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus('请选择文件后构建');
        return;
      }
      setStatus(`已选择文件: ${file.name}，点击“构建文档树”开始`);
    });

    buildBtn.addEventListener('click', async () => {
      try {
        await buildDocTree();
      } catch (err) {
        setStatus(`构建失败: ${err.message || err}`, true);
      } finally {
        endBuildUi();
      }
    });
  </script>
</body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title="Doc Tree Debug GUI")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _build_page_html()

    @app.post("/api/build")
    async def build(file: UploadFile = File(...)) -> Dict[str, Any]:
        saved_path = _save_upload(file)
        try:
            payload = await asyncio.to_thread(_build_payload_from_file, str(saved_path))
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="文档树调试前端（浏览器选文件）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7869)
    parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--input-file", default="", help="直接构建指定文件并输出结果，不启动 Web")
    parser.add_argument("--output-json", default="", help="配合 --input-file 使用：将 payload 写入指定 JSON 文件")
    parser.add_argument("--print-tree", action="store_true", help="配合 --input-file 使用：在终端打印 catalog 树")
    parser.add_argument("--include-build-debug", action="store_true", help="配合 --input-file 使用：输出构建过程中的 marker/range/section-map 调试信息")
    args = parser.parse_args()

    input_file = str(args.input_file or "").strip()
    if input_file:
        payload = _build_payload_from_file(input_file, include_build_debug=bool(args.include_build_debug))
        output_json = str(args.output_json or "").strip()
        if output_json:
            out_path = Path(output_json).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[DocTreeDebug] Payload JSON: {out_path}")
        if args.print_tree:
            print("\n[DocTreeDebug] Tree Lines")
            for line in _catalog_tree_lines(payload.get("tree") or []):
                print(line)
        return

    if not args.no_open:
        url = f"http://{args.host}:{args.port}"
        try:
            webbrowser.open(url)
        except Exception:
            pass

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
