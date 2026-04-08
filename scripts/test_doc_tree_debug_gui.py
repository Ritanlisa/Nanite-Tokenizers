from __future__ import annotations

import argparse
import asyncio
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
    title = str(getattr(page, "title", "") or "Untitled")
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


def _build_payload_from_file(file_path: str) -> Dict[str, Any]:
    from rag.documents import load_rag_documents_from_paths

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
    tree = [_serialize_page_node(node, node_id_prefix="root") for node in rag_doc.get_page_nodes()]

    return {
        "status": "ok",
        "document": source_file.name,
        "doc_name": str(getattr(rag_doc, "doc_name", "") or source_file.name),
        "title": str(getattr(rag_doc, "title", "") or source_file.name),
        "pagination_mode": str(getattr(rag_doc, "pagination_mode", "") or ""),
        "page_count": int(getattr(rag_doc, "page_count", 0) or 0),
        "catalog": _json_safe(getattr(rag_doc, "catalog", []) or []),
        "variables": _page_variable_snapshot(rag_doc),
        "tree": tree,
    }


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
    .vars { display: grid; grid-template-columns: 1fr; gap: 10px; }
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

    <div class=\"layout\">
      <section class=\"panel\">
        <div class=\"head\">文档树结构</div>
        <div class=\"body\"><ul class=\"tree\" id=\"treeRoot\"></ul></div>
      </section>
      <section class=\"panel\">
        <div class=\"head\">Page 子页详情</div>
        <div class=\"body\"><div class=\"vars\" id=\"vars\"></div></div>
      </section>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const buildBtn = document.getElementById('buildBtn');
    const statusEl = document.getElementById('status');
    const treeRoot = document.getElementById('treeRoot');
    const varsEl = document.getElementById('vars');
    const meta = document.getElementById('meta');

    let payload = null;
    let selectedNodeId = '';
    let buildTimer = null;
    let buildStartTs = 0;

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

    function renderVars(node) {
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
      treeRoot.innerHTML = '';
      function build(items) {
        const ul = document.createElement('ul');
        ul.className = 'tree';
        for (const n of items || []) {
          const li = document.createElement('li');
          const btn = document.createElement('button');
          btn.className = 'node' + (selectedNodeId === n.id ? ' active' : '');
          const sub = document.createElement('div');
          sub.className = 'sub';
          btn.innerHTML = `<div>${n.title || 'Untitled'}</div>`;
          sub.textContent = `${n.class_name || 'Page'} | ${n.category || 'unknown'}`;
          btn.appendChild(sub);
          btn.onclick = () => {
            selectedNodeId = n.id;
            renderTree(payload.tree || []);
            renderVars(n);
          };
          li.appendChild(btn);
          const children = Array.isArray(n.children) ? n.children : [];
          if (children.length > 0) li.appendChild(build(children));
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
      meta.textContent = [
        `doc=${payload.document || ''}`,
        `title=${payload.title || ''}`,
        `doc_name=${payload.doc_name || ''}`,
        `page_count=${payload.page_count || 0}`,
        `pagination=${payload.pagination_mode || ''}`,
        `nodes=${nodes.length}`
      ].join(' | ');

      selectedNodeId = '';
      renderTree(payload.tree || []);
      if (nodes.length > 0) {
        selectedNodeId = nodes[0].id;
        renderTree(payload.tree || []);
        renderVars(nodes[0]);
      } else {
        varsEl.innerHTML = '';
      }
      setStatus(`构建完成，节点数: ${nodes.length}`);
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
    args = parser.parse_args()

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
