from __future__ import annotations

import argparse
import asyncio
import hashlib
from io import BytesIO
import json
import mimetypes
import re
import subprocess
import sys
import tempfile
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Set

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

DEBUG_IMAGE_STORE: Dict[str, Any] = {}
DEBUG_IMAGE_RENDER_CACHE: Dict[str, tuple[bytes, str]] = {}
DEBUG_PREVIEW_IMAGE_MAX_PX = 960
DEBUG_PREVIEW_WEBP_QUALITY = 72
HEAVY_METADATA_KEYS: Set[str] = {
  "structured_page_assets",
  "page_layout",
  "pdf_reference_pages",
  "txt_reference_pages",
  "structured_sections",
  "native_catalog",
  "style_catalog",
  "font_catalog",
  "pdf_reference_catalog",
}


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", str(name).strip())
    return cleaned or "document"


def _json_safe(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "byte_size": len(value)}
    if hasattr(value, "to_debug_payload"):
        try:
            return _json_safe(value.to_debug_payload(), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass
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


def _asset_field(asset: Any, key: str, default: Any = None) -> Any:
  if isinstance(asset, dict):
    return asset.get(key, default)
  return getattr(asset, key, default)


def _refresh_debug_image_store(rag_doc: Any) -> None:
  DEBUG_IMAGE_STORE.clear()
  pages = []
  if hasattr(rag_doc, "get_mono_pages"):
    try:
      pages = list(getattr(rag_doc, "get_mono_pages")() or [])
    except Exception:
      pages = []
  for page in pages:
    try:
      images = list(getattr(page, "get_images")() or [])
    except Exception:
      images = []
    for image in images:
      asset_id = str(getattr(image, "asset_id", "") or "").strip()
      if not asset_id:
        continue
      raw_data = bytes(getattr(image, "data", b"") or b"")
      media_type = str(getattr(image, "media_type", "") or "application/octet-stream").strip()
      filename = str(getattr(image, "filename", "") or f"{asset_id}.bin").strip()
      preview_data = raw_data
      preview_media_type = media_type
      if raw_data and media_type.startswith("image/"):
        try:
          compressed, compressed_media_type = _build_debug_image_bytes(
            raw_data,
            media_type,
            filename=filename,
            preview=True,
            max_px=DEBUG_PREVIEW_IMAGE_MAX_PX,
          )
          if compressed:
            preview_data = compressed
            preview_media_type = compressed_media_type or media_type
        except Exception:
          preview_data = raw_data
          preview_media_type = media_type
      DEBUG_IMAGE_STORE[asset_id] = {
        "asset_id": asset_id,
        "filename": filename,
        "media_type": preview_media_type or media_type,
        "data": preview_data,
        "source": str(getattr(image, "source", "") or ""),
        "page": str(getattr(image, "page", "") or ""),
        "width": str(getattr(image, "width", "") or ""),
        "height": str(getattr(image, "height", "") or ""),
        "caption": str(getattr(image, "caption", "") or ""),
        "has_binary": bool(preview_data),
      }


def _compact_image_debug_payload(item: Any) -> Dict[str, Any]:
  raw: Dict[str, Any] = {}
  if hasattr(item, "to_debug_payload"):
    try:
      payload = item.to_debug_payload()
      if isinstance(payload, dict):
        raw = dict(payload)
    except Exception:
      raw = {}
  elif isinstance(item, dict):
    raw = dict(item)

  asset_id = str(raw.get("asset_id") or _asset_field(item, "asset_id", "") or "").strip()
  cached = DEBUG_IMAGE_STORE.get(asset_id) if asset_id else None
  media_type = str(
    _asset_field(cached, "media_type", "")
    or raw.get("media_type")
    or _asset_field(item, "media_type", "")
    or "application/octet-stream"
  ).strip()
  filename = str(
    _asset_field(cached, "filename", "")
    or raw.get("filename")
    or _asset_field(item, "filename", "")
    or ""
  ).strip()
  page = str(_asset_field(cached, "page", "") or raw.get("page") or "").strip()
  width = str(_asset_field(cached, "width", "") or raw.get("width") or "").strip()
  height = str(_asset_field(cached, "height", "") or raw.get("height") or "").strip()
  data_size = len(bytes(_asset_field(cached, "data", b"") or b"")) if cached is not None else int(raw.get("byte_size") or 0)

  return {
    "asset_id": asset_id,
    "filename": filename,
    "media_type": media_type,
    "byte_size": str(data_size),
    "width": width,
    "height": height,
    "page": page,
    "has_binary": bool(data_size > 0),
    "preview_only": True,
  }


def _summarize_large_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) <= 240:
            return text
        return {"char_count": len(text), "preview": text[:240]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "size": len(value),
            "keys": [str(key) for key in list(value.keys())[:12]],
        }
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "size": len(value)}
    if hasattr(value, "doc_id") or hasattr(value, "text"):
        metadata = getattr(value, "metadata", {}) or {}
        doc_text = str(getattr(value, "text", "") or "")
        return {
            "type": type(value).__name__,
            "doc_id": str(getattr(value, "doc_id", "") or ""),
            "metadata_keys": [str(key) for key in list(dict(metadata).keys())[:12]],
            "text_chars": len(doc_text),
        }
    return str(value)


def _metadata_snapshot(metadata: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    meta = dict(metadata or {})
    for key, value in meta.items():
        key_text = str(key)
        if key_text in HEAVY_METADATA_KEYS:
            result[key_text] = _summarize_large_value(value)
            continue
        result[key_text] = _json_safe(value, max_depth=4)
    return result


def _local_asset_payload(page: Any) -> Dict[str, Any]:
    assets = getattr(page, "assets", None)
    if assets is None:
        return {"headers": [], "footers": [], "page_numbers": [], "images": []}

    def _clean_text_list(values: Any) -> list[str]:
        cleaned: list[str] = []
        for item in list(values or []):
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    images: list[Any] = []
    for item in list(getattr(assets, "images", []) or []):
      if hasattr(item, "to_debug_payload") or isinstance(item, dict):
        images.append(_compact_image_debug_payload(item))
        continue
      text = str(item or "").strip()
      if text:
        images.append(text)

    return {
        "headers": _clean_text_list(getattr(assets, "headers", [])),
        "footers": _clean_text_list(getattr(assets, "footers", [])),
        "page_numbers": _clean_text_list(getattr(assets, "page_numbers", [])),
        "images": images,
    }


def _page_to_payload_snapshot(page: Any) -> Dict[str, Any]:
    to_payload = getattr(page, "to_payload", None)
    if not callable(to_payload):
        return _local_asset_payload(page)

    try:
        payload_raw = to_payload()
    except Exception:
        return _local_asset_payload(page)

    if not isinstance(payload_raw, dict):
        return _local_asset_payload(page)

    payload: Dict[str, Any] = dict(payload_raw)
    payload.pop("SubContent", None)
    payload.pop("title", None)
    payload.pop("category", None)
    payload.pop("metadata", None)
    payload.pop("markdown_text", None)
    images_raw = payload.get("images")
    if isinstance(images_raw, list):
      compact_images: list[Any] = []
      for item in images_raw:
        if hasattr(item, "to_debug_payload") or isinstance(item, dict):
          compact_images.append(_compact_image_debug_payload(item))
        else:
          text = str(item or "").strip()
          if text:
            compact_images.append(text)
      payload["images"] = compact_images
    return payload


def _build_debug_render_markdown_text(page: Any) -> str:
  if hasattr(page, "merged_markdown"):
    try:
      merged = str(getattr(page, "merged_markdown")() or "")
      if merged.strip():
        return merged
    except Exception:
      pass
  return str(getattr(page, "markdown_text", "") or "")


def _snapshot_attr_value(key: str, value: Any) -> Any:
    if key == "SubContent":
        return f"<{len(value) if isinstance(value, list) else 0} children>"
    if key in {"page_nodes", "chunk_documents", "catalog", "source_document", "cleaned_text"}:
        return _summarize_large_value(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _summarize_large_value(value)
    if isinstance(value, dict):
        if len(value) > 20:
            return _summarize_large_value(value)
        return {str(sub_key): _json_safe(sub_value, max_depth=2) for sub_key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return _summarize_large_value(value)
    return str(value)


def _page_variable_snapshot(page: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any]
    try:
        attrs = dict(vars(page))
    except Exception:
        attrs = {}

    raw_markdown_text = str(getattr(page, "markdown_text", "") or "")
    render_markdown_text = _build_debug_render_markdown_text(page)
    payload_snapshot = _page_to_payload_snapshot(page)
    payload_images = list(payload_snapshot.get("images") or []) if isinstance(payload_snapshot, dict) else []
    snapshot: Dict[str, Any] = {
        "class_name": str(type(page).__name__),
        "category": str(getattr(page, "category", "") or ""),
        "title": str(getattr(page, "title", "") or ""),
        "markdown_text": render_markdown_text,
        "raw_markdown_text": raw_markdown_text,
        "render_markdown_text": render_markdown_text,
        "metadata": _metadata_snapshot(getattr(page, "metadata", {}) or {}),
      "assets": {
        "image_count": len(payload_images),
        "image_preview": _json_safe(payload_images[:3], max_depth=2),
      },
      "to_payload": payload_snapshot,
    }

    for key, value in attrs.items():
        if key in {"title", "markdown_text", "metadata", "assets"}:
            continue
        snapshot[key] = _snapshot_attr_value(key, value)
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
        "metadata": _metadata_snapshot(getattr(page, "metadata", {}) or {}),
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


def _count_markdown_image_tokens(text: Any) -> int:
    count = 0
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", str(text or "")):
        target = str(match.group(1) or "").strip()
        if re.fullmatch(r"image://page-\d+/\d+", target, flags=re.IGNORECASE):
            count += 1
    return count


def _collect_image_alignment_debug(tree: list[Dict[str, Any]]) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []

    def _walk(nodes: list[Dict[str, Any]]) -> None:
        for node in nodes:
            vars_ = dict(node.get("variables") or {})
            payload = dict(vars_.get("to_payload") or {}) if isinstance(vars_.get("to_payload"), dict) else {}
            markdown_text = str(vars_.get("render_markdown_text") or vars_.get("markdown_text") or "")
            token_count = _count_markdown_image_tokens(markdown_text)
            image_count = len(list(payload.get("images") or []))
            row = {
                "title": str(node.get("title") or ""),
                "category": str(node.get("category") or ""),
                "page": (node.get("metadata") or {}).get("page"),
                "token_count": token_count,
                "image_count": image_count,
            }
            if token_count != image_count:
                rows.append(row)
            _walk(list(node.get("children") or []))

    _walk(list(tree or []))
    return {
        "mismatch_count": len(rows),
        "mismatches": rows,
    }


def _build_pipeline_debug(rag_doc: Any) -> Dict[str, Any]:
    debug: Dict[str, Any] = {
        "doc_parser": str(getattr(rag_doc, "metadata", {}).get("doc_parser") or ""),
        "docx_parser": str(getattr(rag_doc, "metadata", {}).get("docx_parser") or ""),
        "native_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("native_catalog") or [])),
        "style_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("style_catalog") or [])),
        "font_catalog_count": len(list(getattr(rag_doc, "metadata", {}).get("font_catalog") or [])),
    }

    build_trace = dict(getattr(rag_doc, "metadata", {}).get("_doc_tree_build_trace") or {})
    if build_trace:
        debug["build_trace"] = _json_safe(build_trace)
        return debug

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


def _export_markdown_from_rag_doc(rag_doc: Any) -> str:
    if hasattr(rag_doc, "export_markdown_from_tree"):
        return str(getattr(rag_doc, "export_markdown_from_tree")() or "")
    return ""


def _build_payload_and_rag_doc_from_file(
    file_path: str,
    *,
    include_build_debug: bool = False,
    emit_summary: bool = False,
    emit_tree: bool = False,
    emit_boundary_debug: bool = False,
  assert_image_alignment: bool = False,
) -> tuple[Dict[str, Any], Any]:
    from rag.documents import load_rag_documents_from_paths, _probe_effective_native_page_count

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
    _refresh_debug_image_store(rag_doc)
    page_nodes = list(rag_doc.get_page_nodes() or [])
    tree = [_serialize_page_node(node, node_id_prefix="root") for node in page_nodes]
    mono_pages = list(getattr(rag_doc, "get_mono_pages")() or []) if hasattr(rag_doc, "get_mono_pages") else []
    single_pages = list(getattr(rag_doc, "get_single_pages")() or mono_pages) if hasattr(rag_doc, "get_single_pages") else mono_pages
    mono_page_count = len(single_pages)
    leaf_mono_page_count = len(mono_pages)
    native_page_count_raw = getattr(rag_doc, "metadata", {}).get("native_page_count") if hasattr(rag_doc, "metadata") else None
    native_page_count = int(native_page_count_raw or 0) if str(native_page_count_raw or "").strip() else 0
    runtime_native_page_count = int(_probe_effective_native_page_count(str(source_file)) or 0)
    page_count = int(getattr(rag_doc, "page_count", 0) or 0)
    expected_pages = max(runtime_native_page_count, native_page_count)
    is_aligned = bool(expected_pages > 0 and mono_page_count == expected_pages and page_count == expected_pages)
    if emit_tree:
      print(f"\n[DocTreeDebug] Structured catalog for: {source_file.name}")
      for line in _catalog_tree_lines(tree):
        print(line)

    chapter_boundary_debug = _collect_chapter_boundary_debug(tree)
    image_alignment_debug = _collect_image_alignment_debug(tree)
    if emit_boundary_debug and chapter_boundary_debug:
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
      "leaf_mono_page_count": leaf_mono_page_count,
      "native_page_count": native_page_count,
      "runtime_native_page_count": runtime_native_page_count,
      "expected_page_count": expected_pages,
      "is_page_aligned": is_aligned,
      "catalog": _json_safe(getattr(rag_doc, "catalog", []) or []),
      "variables": _page_variable_snapshot(rag_doc),
      "tree": tree,
      "chapter_boundary_debug": chapter_boundary_debug,
      "image_alignment_debug": image_alignment_debug,
    }
    if include_build_debug:
      payload["build_debug"] = _build_pipeline_debug(rag_doc)

    safe_name = _sanitize_filename(source_file.stem)
    out_path = DEBUG_JSON_DIR / f"{safe_name}_{uuid.uuid4().hex[:8]}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["debug_json_path"] = str(out_path)
    if emit_summary:
      print(f"[DocTreeDebug] JSON sidecar: {out_path}")
      print(
        "[DocTreeDebug] Alignment: "
        f"page_count={page_count}, mono_page_count={mono_page_count}, leaf_mono_page_count={leaf_mono_page_count}, "
        f"native_page_count={native_page_count}, runtime_native_page_count={runtime_native_page_count}, "
        f"expected_page_count={expected_pages}, aligned={is_aligned}"
      )
      print(
        "[DocTreeDebug] Image Alignment: "
        f"mismatch_count={int(image_alignment_debug.get('mismatch_count') or 0)}"
      )

    if bool(assert_image_alignment) and int(image_alignment_debug.get("mismatch_count") or 0) > 0:
      head = list(image_alignment_debug.get("mismatches") or [])[:10]
      raise AssertionError(f"Image alignment mismatch: {json.dumps(head, ensure_ascii=False)}")

    return payload, rag_doc


def _build_payload_from_file(file_path: str, *, include_build_debug: bool = False) -> Dict[str, Any]:
    payload, _ = _build_payload_and_rag_doc_from_file(
        file_path,
        include_build_debug=include_build_debug,
        emit_summary=False,
        emit_tree=False,
        emit_boundary_debug=False,
    )
    return payload


def _save_upload(upload: Any) -> Path:
    from fastapi import HTTPException
    filename = _sanitize_filename(upload.filename or "uploaded")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    saved = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    content = upload.file.read()
    saved.write_bytes(content)
    return saved


def _resolve_workspace_asset_path(raw_path: str) -> Path:
    from fastapi import HTTPException
    asset_path = str(raw_path or "").strip()
    if not asset_path:
        raise HTTPException(status_code=400, detail="缺少资产路径")

    candidate = Path(asset_path).expanduser()
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (PROJECT_ROOT / candidate).resolve()

    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="资产路径超出工作区范围") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="资产文件不存在")
    return candidate


def _resolve_debug_image_asset(asset_id: str) -> Any:
    from fastapi import HTTPException
    key = str(asset_id or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="缺少 asset_id")
    asset = DEBUG_IMAGE_STORE.get(key)
    if asset is None:
        raise HTTPException(status_code=404, detail="调试图片不存在")
    data = bytes(_asset_field(asset, "data", b"") or b"")
    if not data:
        raise HTTPException(status_code=404, detail="调试图片无二进制内容")
    media_type = str(_asset_field(asset, "media_type", "") or "application/octet-stream").strip()
    return asset, data, media_type


def _detect_content_bbox_in_image_bytes(data: bytes) -> Optional[tuple[int, int, int, int]]:
    payload = bytes(data or b"")
    if not payload:
        return None
    try:
        from PIL import Image, ImageChops  # type: ignore

        with Image.open(BytesIO(payload)) as image:
            image.load()
            rgba = image.convert("RGBA")

            alpha_bbox = None
            try:
                alpha_bbox = rgba.getchannel("A").getbbox()
                if alpha_bbox == (0, 0, rgba.width, rgba.height):
                    alpha_bbox = None
            except Exception:
                alpha_bbox = None

            rgb = rgba.convert("RGB")
            white_bg = Image.new("RGB", rgb.size, (255, 255, 255))
            diff = ImageChops.difference(rgb, white_bg).convert("L")
            threshold_lut = [0] * 256
            for index in range(5, 256):
                threshold_lut[index] = 255
            diff = diff.point(threshold_lut)
            rgb_bbox = diff.getbbox()

            bbox = alpha_bbox or rgb_bbox
            if alpha_bbox and rgb_bbox:
                bbox = (
                    min(alpha_bbox[0], rgb_bbox[0]),
                    min(alpha_bbox[1], rgb_bbox[1]),
                    max(alpha_bbox[2], rgb_bbox[2]),
                    max(alpha_bbox[3], rgb_bbox[3]),
                )
            if bbox is None:
                return None
            return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except Exception:
        return None


def _crop_rendered_image_bytes(data: bytes, *, padding: int = 8) -> bytes:
    payload = bytes(data or b"")
    if not payload:
        return payload
    try:
        from PIL import Image  # type: ignore

        bbox = _detect_content_bbox_in_image_bytes(payload)
        if bbox is None:
            return payload

        with Image.open(BytesIO(payload)) as image:
            image.load()
            rgba = image.convert("RGBA")

            left = max(0, int(bbox[0]) - int(padding))
            top = max(0, int(bbox[1]) - int(padding))
            right = min(rgba.width, int(bbox[2]) + int(padding))
            bottom = min(rgba.height, int(bbox[3]) + int(padding))
            if left == 0 and top == 0 and right == rgba.width and bottom == rgba.height:
                return payload

            cropped = rgba.crop((left, top, right, bottom))
            rendered = BytesIO()
            cropped.save(rendered, format="PNG", optimize=True)
            output = rendered.getvalue()
            return output or payload
    except Exception:
        return payload

def _looks_like_vector_metafile(media_type: str, filename: str = "") -> bool:
    normalized_media_type = str(media_type or "").strip().lower()
    normalized_name = str(filename or "").strip().lower()
    if normalized_name.endswith((".wmf", ".emf")):
        return True
    return normalized_media_type in {
        "image/wmf",
        "image/x-wmf",
        "image/emf",
        "image/x-emf",
        "application/x-msmetafile",
    }


def _render_vector_metafile_to_png(data: bytes, *, filename: str = "") -> tuple[bytes, str]:
    payload = bytes(data or b"")
    if not payload:
        return payload, "application/octet-stream"

    cache_key = hashlib.sha1((str(filename or "") + "\0").encode("utf-8") + payload).hexdigest()
    cached = DEBUG_IMAGE_RENDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
      from rag.documents import _enable_python3_uno_bridge, _get_libreoffice_executable

      office = _get_libreoffice_executable()
      if office and _enable_python3_uno_bridge():
        import os
        import random
        import time
        import uno
        import fitz  # type: ignore

        PropertyValue = __import__("com.sun.star.beans", fromlist=["PropertyValue"]).PropertyValue
        port = str(random.randint(41000, 43999))
        with tempfile.TemporaryDirectory(prefix="doc_tree_render_") as tmp_dir:
          suffix = Path(str(filename or "asset.wmf")).suffix.lower() or ".wmf"
          staged_path = Path(tmp_dir) / f"asset{suffix}"
          pdf_path = Path(tmp_dir) / "asset.pdf"
          staged_path.write_bytes(payload)

          profile_url = "file://" + os.path.abspath(tmp_dir).replace("\\", "/")
          cmd = [
            office,
            "--headless",
            "--invisible",
            "--norestore",
            "--nodefault",
            "--nofirststartwizard",
            "--nolockcheck",
            "--nologo",
            f"-env:UserInstallation={profile_url}",
            f"--accept=socket,host=127.0.0.1,port={port};urp;",
          ]
          proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
            ctx = None
            deadline = time.time() + 30.0
            while time.time() < deadline:
              try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                break
              except Exception:
                time.sleep(0.2)

            if ctx is not None:
              desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
              writer = desktop.loadComponentFromURL(
                "private:factory/swriter",
                "_blank",
                0,
                (PropertyValue("Hidden", 0, True, 0),),
              )
              if writer is not None:
                text = writer.Text
                cursor = text.createTextCursor()
                graphic = writer.createInstance("com.sun.star.text.TextGraphicObject")
                graphic.GraphicURL = uno.systemPathToFileUrl(str(staged_path))
                text.insertTextContent(cursor, graphic, False)
                writer.storeToURL(
                  uno.systemPathToFileUrl(str(pdf_path)),
                  (PropertyValue("FilterName", 0, "writer_pdf_Export", 0),),
                )
                writer.close(True)
                if pdf_path.exists():
                  with fitz.open(str(pdf_path)) as pdf_doc:
                    page = pdf_doc.load_page(0)
                    inspect_scale = 6.0
                    inspect_pix = page.get_pixmap(matrix=fitz.Matrix(inspect_scale, inspect_scale), alpha=False)
                    inspect_png = inspect_pix.tobytes("png")
                    bbox = _detect_content_bbox_in_image_bytes(inspect_png)

                    clip_rect = page.rect
                    if bbox is not None:
                      clip_rect = fitz.Rect(
                        float(bbox[0]) / inspect_scale,
                        float(bbox[1]) / inspect_scale,
                        float(bbox[2]) / inspect_scale,
                        float(bbox[3]) / inspect_scale,
                      )
                      clip_padding = max(0.5, 4.0 / inspect_scale)
                      clip_rect = fitz.Rect(
                        max(page.rect.x0, clip_rect.x0 - clip_padding),
                        max(page.rect.y0, clip_rect.y0 - clip_padding),
                        min(page.rect.x1, clip_rect.x1 + clip_padding),
                        min(page.rect.y1, clip_rect.y1 + clip_padding),
                      )

                    content_longest_edge = max(float(clip_rect.width or 0.0), float(clip_rect.height or 0.0), 1.0)
                    render_scale = max(6.0, min(96.0, 1600.0 / content_longest_edge))
                    pix = page.get_pixmap(
                      matrix=fitz.Matrix(render_scale, render_scale),
                      clip=clip_rect,
                      alpha=False,
                    )
                    output = _crop_rendered_image_bytes(pix.tobytes("png"), padding=1)
                    if output:
                      DEBUG_IMAGE_RENDER_CACHE[cache_key] = (output, "image/png")
                      return output, "image/png"
          finally:
            try:
              proc.terminate()
              proc.wait(timeout=5)
            except Exception:
              pass
    except Exception:
      pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(BytesIO(payload)) as image:
            image.load()
            rendered = BytesIO()
            save_image = image if image.mode in {"RGB", "RGBA", "L"} else image.convert("RGBA")
            save_image.save(rendered, format="PNG", optimize=True)
            output = _crop_rendered_image_bytes(rendered.getvalue(), padding=2)
            if output:
                DEBUG_IMAGE_RENDER_CACHE[cache_key] = (output, "image/png")
                return output, "image/png"
    except Exception:
        pass

    return payload, mimetypes.guess_type(str(filename or ""))[0] or "application/octet-stream"


def _build_debug_image_bytes(
    data: bytes,
    media_type: str,
    *,
    filename: str = "",
    preview: bool = False,
    max_px: int = 360,
) -> tuple[bytes, str]:
    payload = bytes(data or b"")
    resolved_media_type = str(media_type or mimetypes.guess_type(str(filename or ""))[0] or "application/octet-stream").strip() or "application/octet-stream"
    if payload and _looks_like_vector_metafile(resolved_media_type, filename):
        rendered_payload, rendered_media_type = _render_vector_metafile_to_png(payload, filename=filename)
        if rendered_payload and rendered_media_type.startswith("image/"):
            payload = rendered_payload
            resolved_media_type = rendered_media_type
    if not preview or not payload or not resolved_media_type.startswith("image/"):
        return payload, resolved_media_type
    try:
        from PIL import Image  # type: ignore

        with Image.open(BytesIO(payload)) as image:
            image.load()
            resampling_owner = getattr(Image, "Resampling", None)
            resampling = getattr(resampling_owner, "LANCZOS", None)
            if resampling is None:
                resampling = getattr(Image, "LANCZOS", None)
            if resampling is None:
                resampling = getattr(Image, "BICUBIC", 3)
            resample_filter: Any = resampling
            image.thumbnail((max(64, int(max_px or 360)), max(64, int(max_px or 360))), resample_filter)
            fmt = "WEBP"
            save_image = image.convert("RGB") if image.mode not in {"RGB", "L"} else image
            output = BytesIO()
            save_image.save(output, format=fmt, optimize=True, quality=DEBUG_PREVIEW_WEBP_QUALITY, method=4)
            output_bytes = output.getvalue()
            if output_bytes:
                resolved_media_type = str(getattr(Image, "MIME", {}).get(fmt) or resolved_media_type)
                return output_bytes, resolved_media_type
    except Exception:
        return payload, resolved_media_type
    return payload, resolved_media_type


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
    .asset-media-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .asset-media-entry {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: rgba(255,255,255,0.03);
      min-height: 68px;
    }
    .asset-media-item {
      display: block;
      color: inherit;
      text-decoration: none;
    }
    .asset-link {
      color: var(--accent);
      text-decoration: none;
      font-size: 12px;
      font-weight: 700;
    }
    .asset-link:hover {
      text-decoration: underline;
    }
    .asset-thumb {
      display: block;
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.22);
      margin-bottom: 6px;
    }
    .asset-path {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
      word-break: break-word;
      white-space: pre-wrap;
      margin-top: 6px;
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
    .markdown-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: rgba(10, 16, 13, 0.55);
      margin-bottom: 10px;
    }
    .markdown-render {
      color: var(--text);
      font-size: 13px;
      line-height: 1.65;
      white-space: normal;
      word-break: break-word;
    }
    .markdown-render h1,
    .markdown-render h2,
    .markdown-render h3,
    .markdown-render h4,
    .markdown-render h5,
    .markdown-render h6 {
      margin: 12px 0 8px;
      line-height: 1.35;
      color: var(--accent);
      font-weight: 700;
    }
    .markdown-render h1 { font-size: 28px; }
    .markdown-render h2 { font-size: 24px; }
    .markdown-render h3 { font-size: 20px; }
    .markdown-render h4 { font-size: 17px; }
    .markdown-render h5 { font-size: 15px; }
    .markdown-render h6 { font-size: 14px; }
    .markdown-render p,
    .markdown-render ul,
    .markdown-render ol,
    .markdown-render table,
    .markdown-render pre,
    .markdown-render blockquote {
      margin: 0 0 12px;
    }
    .markdown-render code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: rgba(255,255,255,0.08);
      padding: 1px 4px;
      border-radius: 4px;
    }
    .markdown-render pre {
      overflow: auto;
      padding: 10px;
      border-radius: 8px;
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--line);
    }
    .markdown-render pre code {
      background: transparent;
      padding: 0;
    }
    .markdown-table-wrap {
      overflow-x: auto;
      margin: 0 0 12px;
    }
    .markdown-render table {
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
    }
    .markdown-render th,
    .markdown-render td {
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
    }
    .markdown-render th {
      color: var(--accent2);
      background: rgba(255,255,255,0.04);
    }
    .markdown-render figure {
      margin: 0 0 14px;
      padding: 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }
    .markdown-render figure a {
      display: inline-block;
      max-width: 100%;
    }
    .markdown-render img {
      display: block;
      max-width: 100%;
      max-height: 520px;
      width: auto;
      height: auto;
      object-fit: contain;
      border-radius: 8px;
      background: rgba(0,0,0,0.22);
      border: 1px solid var(--line);
    }
    .markdown-render figcaption {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
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
      </div>
      <div id=\"status\" class=\"status\">请选择文件，选择后将自动构建</div>
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
          <div class="markdown-card" id="renderMarkdownCard" hidden>
            <div class="asset-title">Markdown 渲染预览（render_markdown_text）</div>
            <div class="markdown-render" id="renderMarkdownEl"></div>
          </div>
          <div class="assets" id="assets"></div>
          <div class="vars" id="vars"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const statusEl = document.getElementById('status');
    const treeRoot = document.getElementById('treeRoot');
    const assetsEl = document.getElementById('assets');
    const renderMarkdownCard = document.getElementById('renderMarkdownCard');
    const renderMarkdownEl = document.getElementById('renderMarkdownEl');
    const varsEl = document.getElementById('vars');
    const meta = document.getElementById('meta');
    const IMAGE_PREVIEW_MAX_PX = 960;

    let payload = null;
    let selectedNodeId = '';
    let buildTimer = null;
    let buildStartTs = 0;
    let activeBuildToken = 0;
    const preloadedAssetUrls = new Set();
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
      if (Array.isArray(value)) {
        return value.filter(v => {
          if (v && typeof v === 'object') return true;
          return String(v || '').trim() !== '';
        });
      }
      if (value === null || value === undefined) return [];
      if (typeof value === 'object') return [value];
      const text = String(value).trim();
      return text ? [text] : [];
    }

    function isImageAssetObject(value) {
      return Boolean(value && typeof value === 'object' && String(value.asset_id || '').trim());
    }

    function isBrowserPreviewableMediaType(value) {
      const text = String(value || '').trim().toLowerCase();
      return /^(image\\/(png|jpe?g|gif|webp|bmp|svg\\+xml))$/.test(text);
    }

    function isServerRenderableMediaType(value) {
      const text = String(value || '').trim().toLowerCase();
      return /^(image\\/(wmf|x-wmf|emf|x-emf)|application\\/x-msmetafile)$/.test(text);
    }

    function isLikelyFilePath(value) {
      const text = String(value || '').trim();
      if (!text) return false;
      return text.startsWith('/') || text.startsWith('./') || text.startsWith('../');
    }

    function isPreviewableImagePath(value) {
      const text = String(value || '').trim().toLowerCase();
      if (!isLikelyFilePath(text)) return false;
      return /\\.(png|jpe?g|gif|webp|bmp|svg|wmf|emf)$/.test(text);
    }

    function buildAssetHref(value, options = {}) {
      const params = new URLSearchParams({ path: String(value || '').trim() });
      if (options.preview) {
        params.set('preview', '1');
      }
      if (options.maxPx) {
        params.set('max_px', String(options.maxPx));
      }
      return `/api/asset?${params.toString()}`;
    }

    function buildAssetObjectHref(value, options = {}) {
      const assetId = String((value && value.asset_id) || '').trim();
      const params = new URLSearchParams({ asset_id: assetId });
      if (options.preview) {
        params.set('preview', '1');
      }
      if (options.maxPx) {
        params.set('max_px', String(options.maxPx));
      }
      return `/api/asset-object?${params.toString()}`;
    }

    function escapeHtml(value) {
      return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function applyInlineMarkdown(text) {
      let html = escapeHtml(text);
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
      return html;
    }

    function isBlankLine(line) {
      return String(line || '').trim() === '';
    }

    function splitMarkdownTableRow(line) {
      const source = String(line || '').trim();
      if (!source || !source.includes('|')) {
        return [];
      }
      const backslash = String.fromCharCode(92);
      let text = source;
      if (text.startsWith('|')) {
        text = text.slice(1);
      }
      if (text.endsWith('|')) {
        text = text.slice(0, -1);
      }

      const cells = [];
      let current = '';
      for (let index = 0; index < text.length; index += 1) {
        const char = text[index];
        const nextChar = index + 1 < text.length ? text[index + 1] : '';
        if (char === backslash && (nextChar === '|' || nextChar === backslash)) {
          current += nextChar;
          index += 1;
          continue;
        }
        if (char === '|') {
          cells.push(current.trim());
          current = '';
          continue;
        }
        current += char;
      }
      cells.push(current.trim());
      return cells;
    }

    function parseTableAlignment(cell) {
      const value = String(cell || '').trim();
      if (!value) {
        return '';
      }
      if (value.startsWith(':') && value.endsWith(':')) {
        return 'center';
      }
      if (value.endsWith(':')) {
        return 'right';
      }
      if (value.startsWith(':')) {
        return 'left';
      }
      return '';
    }

    function getMarkdownTableInfo(lines, startIndex) {
      const headerLine = String(lines[startIndex] || '').trim();
      const headerCells = splitMarkdownTableRow(headerLine);
      if (headerCells.length < 2) {
        return null;
      }

      let separatorIndex = startIndex + 1;
      while (separatorIndex < lines.length && isBlankLine(lines[separatorIndex])) {
        separatorIndex += 1;
      }
      if (separatorIndex >= lines.length) {
        return null;
      }

      const separatorLine = String(lines[separatorIndex] || '').trim();
      const separatorCells = splitMarkdownTableRow(separatorLine);
      if (separatorCells.length !== headerCells.length) {
        return null;
      }
      if (!separatorCells.every((cell) => /^:?-{3,}:?$/.test(String(cell || '').trim()))) {
        return null;
      }

      return {
        headerCells,
        separatorIndex,
        alignments: separatorCells.map((cell) => parseTableAlignment(cell)),
      };
    }

    function collectNodeImageLookup(node) {
      const byPage = new Map();
      const byAssetId = new Map();
      const looseImages = [];
      const stack = [node];
      while (stack.length > 0) {
        const current = stack.pop();
        if (!current || typeof current !== 'object') {
          continue;
        }
        const metadata = current.metadata && typeof current.metadata === 'object' ? current.metadata : {};
        const pageNo = Number(metadata.page || metadata.physical_page || 0);
        const pageImageIndexes = toList(metadata.page_image_indexes)
          .map((item) => Number(item || 0))
          .filter((item) => Number.isFinite(item) && item > 0);
        const payload = getLocalAssetPayload(current);
        const pageImages = pageNo > 0 ? (byPage.get(pageNo) || []) : [];
        for (const [index, item] of toList(payload.images).entries()) {
          if (isImageAssetObject(item)) {
            byAssetId.set(String(item.asset_id || '').trim(), item);
          }
          if (pageNo <= 0) {
            looseImages.push(item);
            continue;
          }
          const pageImageIndex = Number(pageImageIndexes[index] || 0);
          if (pageImageIndex > 0) {
            while (pageImages.length < pageImageIndex) {
              pageImages.push(null);
            }
            if (!pageImages[pageImageIndex - 1]) {
              pageImages[pageImageIndex - 1] = item;
            } else {
              pageImages.push(item);
            }
            continue;
          }
          pageImages.push(item);
        }
        if (pageNo > 0 && pageImages.length > 0) {
          byPage.set(pageNo, pageImages);
        }
        const children = Array.isArray(current.children) ? current.children : [];
        for (let index = children.length - 1; index >= 0; index -= 1) {
          stack.push(children[index]);
        }
      }

      const images = [];
      const orderedPages = Array.from(byPage.keys()).sort((a, b) => a - b);
      for (const pageNo of orderedPages) {
        for (const item of byPage.get(pageNo) || []) {
          if (item) {
            images.push(item);
          }
        }
      }
      images.push(...looseImages);
      return { byPage, byAssetId, images, imageCount: images.length };
    }

    function resolveMarkdownImageTarget(target, node, lookup) {
      const text = String(target || '').trim();
      if (!text) {
        return '';
      }
      const pageMatch = text.match(/^image:\\/\\/page-(\\d+)\\/(\\d+)$/i);
      if (pageMatch) {
        const pageNo = Number(pageMatch[1] || 0);
        const imageIndex = Math.max(1, Number(pageMatch[2] || 0));
        const pageImages = lookup.byPage.get(pageNo) || [];
        const item = pageImages[imageIndex - 1];
        if (isImageAssetObject(item)) {
          return buildAssetObjectHref(item, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
        }
        if (typeof item === 'string' && isPreviewableImagePath(item)) {
          return buildAssetHref(item, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
        }
        return '';
      }

      const assetMatch = text.match(/^debug-image:\\/\\/asset\\/([^\\s]+)$/i);
      if (assetMatch) {
        const assetId = String(assetMatch[1] || '').trim();
        const asset = lookup.byAssetId.get(assetId);
        if (asset) {
          return buildAssetObjectHref(asset, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
        }
        return buildAssetObjectHref({ asset_id: assetId }, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
      }

      const pathMatch = text.match(/^debug-image:\\/\\/path\\/(.+)$/i);
      if (pathMatch) {
        const decoded = decodeURIComponent(String(pathMatch[1] || '').trim());
        if (decoded) {
          return buildAssetHref(decoded, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
        }
      }

      if (isPreviewableImagePath(text)) {
        return buildAssetHref(text, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
      }
      return text;
    }

    function renderMarkdownText(node) {
      if (!renderMarkdownEl) {
        return;
      }
      renderMarkdownEl.innerHTML = '';
      if (renderMarkdownCard) {
        renderMarkdownCard.hidden = true;
      }
      if (!node || typeof node !== 'object') {
        return;
      }

      if (renderMarkdownCard) {
        renderMarkdownCard.hidden = false;
      }

      const vars = node.variables && typeof node.variables === 'object' ? node.variables : {};
      const markdown = String(vars.render_markdown_text || vars.markdown_text || '').trim();
      if (!markdown) {
        renderMarkdownEl.textContent = '无';
        return;
      }

      const lookup = collectNodeImageLookup(node);
      const lines = markdown.replace(/\\r\\n/g, '\\n').split('\\n');
      let index = 0;

      function appendParagraph(blockLines) {
        const paragraph = document.createElement('p');
        paragraph.innerHTML = blockLines.map((line) => applyInlineMarkdown(line)).join('<br/>');
        renderMarkdownEl.appendChild(paragraph);
      }

      while (index < lines.length) {
        const line = lines[index];
        const trimmed = String(line || '').trim();
        if (!trimmed) {
          index += 1;
          continue;
        }

        const headingMatch = trimmed.match(/^(#{1,6})\\s+(.*)$/);
        if (headingMatch) {
          const level = Math.min(6, Math.max(1, headingMatch[1].length));
          const heading = document.createElement(`h${level}`);
          heading.innerHTML = applyInlineMarkdown(headingMatch[2]);
          renderMarkdownEl.appendChild(heading);
          index += 1;
          continue;
        }

        const imageMatch = trimmed.match(/^!\\[([^\\]]*)\\]\\(([^)]+)\\)$/);
        if (imageMatch) {
          const altText = String(imageMatch[1] || '').trim();
          const target = String(imageMatch[2] || '').trim();
          const src = resolveMarkdownImageTarget(target, node, lookup);
          const figure = document.createElement('figure');
          if (src) {
            const link = document.createElement('a');
            link.href = src;
            link.target = '_blank';
            link.rel = 'noreferrer';
            const img = document.createElement('img');
            img.src = src;
            img.alt = altText || 'image';
            img.loading = 'eager';
            img.decoding = 'async';
            link.appendChild(img);
            figure.appendChild(link);
          }
          const nextLine = String(lines[index + 1] || '').trim();
          const nextLineStartsTable = Boolean(getMarkdownTableInfo(lines, index + 1));
          const captionText = nextLine && !nextLineStartsTable && !/^(!\\[|#{1,6}\\s|[-*]\\s|\\d+\\.\\s|\\|)/.test(nextLine) ? nextLine : altText;
          if (captionText) {
            const caption = document.createElement('figcaption');
            caption.innerHTML = applyInlineMarkdown(captionText);
            figure.appendChild(caption);
          }
          renderMarkdownEl.appendChild(figure);
          index += captionText && nextLine === captionText ? 2 : 1;
          continue;
        }

        const tableInfo = getMarkdownTableInfo(lines, index);
        if (tableInfo) {
          const columnCount = tableInfo.headerCells.length;
          const tableWrap = document.createElement('div');
          tableWrap.className = 'markdown-table-wrap';
          const table = document.createElement('table');
          const thead = document.createElement('thead');
          const tbody = document.createElement('tbody');
          const headerRow = document.createElement('tr');
          for (const [cellIndex, cell] of tableInfo.headerCells.entries()) {
            const th = document.createElement('th');
            const alignment = tableInfo.alignments[cellIndex] || '';
            if (alignment) {
              th.style.textAlign = alignment;
            }
            th.innerHTML = applyInlineMarkdown(cell);
            headerRow.appendChild(th);
          }
          thead.appendChild(headerRow);
          table.appendChild(thead);
          index = tableInfo.separatorIndex + 1;
          while (index < lines.length) {
            const rowLine = String(lines[index] || '').trim();
            if (!rowLine) {
              break;
            }
            const cells = splitMarkdownTableRow(rowLine);
            if (cells.length < 2) {
              break;
            }
            const row = document.createElement('tr');
            const normalizedCells = cells.slice(0, columnCount);
            while (normalizedCells.length < columnCount) {
              normalizedCells.push('');
            }
            for (const [cellIndex, cell] of normalizedCells.entries()) {
              const td = document.createElement('td');
              const alignment = tableInfo.alignments[cellIndex] || '';
              if (alignment) {
                td.style.textAlign = alignment;
              }
              td.innerHTML = applyInlineMarkdown(cell);
              row.appendChild(td);
            }
            tbody.appendChild(row);
            index += 1;
          }
          table.appendChild(tbody);
          tableWrap.appendChild(table);
          renderMarkdownEl.appendChild(tableWrap);
          continue;
        }

        const listMatch = trimmed.match(/^([-*]|\\d+\\.)\\s+(.*)$/);
        if (listMatch) {
          const ordered = /\\d+\\./.test(listMatch[1]);
          const list = document.createElement(ordered ? 'ol' : 'ul');
          while (index < lines.length) {
            const current = String(lines[index] || '').trim();
            const currentMatch = current.match(/^([-*]|\\d+\\.)\\s+(.*)$/);
            if (!currentMatch) {
              break;
            }
            const li = document.createElement('li');
            li.innerHTML = applyInlineMarkdown(currentMatch[2]);
            list.appendChild(li);
            index += 1;
          }
          renderMarkdownEl.appendChild(list);
          continue;
        }

        const paragraphLines = [];
        while (index < lines.length) {
          const current = String(lines[index] || '');
          const currentTrimmed = current.trim();
          if (!currentTrimmed) {
            break;
          }
          if (/^(#{1,6})\\s+/.test(currentTrimmed) || /^!\\[[^\\]]*\\]\\(([^)]+)\\)$/.test(currentTrimmed) || /^([-*]|\\d+\\.)\\s+/.test(currentTrimmed)) {
            break;
          }
          if (getMarkdownTableInfo(lines, index)) {
            break;
          }
          paragraphLines.push(currentTrimmed);
          index += 1;
        }
        if (paragraphLines.length > 0) {
          appendParagraph(paragraphLines);
          continue;
        }
        index += 1;
      }
    }

    function renderAssetTextItem(item) {
      if (item && typeof item === 'object') {
        const li = document.createElement('li');
        li.textContent = pretty(item);
        return li;
      }
      const text = String(item || '').trim();
      const li = document.createElement('li');
      if (isLikelyFilePath(text)) {
        const link = document.createElement('a');
        link.className = 'asset-link';
        link.href = buildAssetHref(text);
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.textContent = text;
        li.appendChild(link);
      } else {
        li.textContent = text;
      }
      return li;
    }

    function renderImageAssetItem(item) {
      const entry = document.createElement('div');
      entry.className = 'asset-media-entry';

      if (isImageAssetObject(item)) {
        const asset = item;
        const hasBinary = Boolean(asset.has_binary);
        const filename = String(asset.filename || asset.caption || asset.asset_id || 'asset').trim();
        const mediaType = String(asset.media_type || '').trim();
        const byteSize = Number(asset.byte_size || 0);
        const source = String(asset.source || '').trim();
        const canPreviewInBrowser = (
          isBrowserPreviewableMediaType(mediaType)
          || isServerRenderableMediaType(mediaType)
          || /\\.(png|jpe?g|gif|webp|bmp|svg|wmf|emf)$/i.test(filename)
        );

        if (hasBinary && mediaType.startsWith('image/')) {
          const box = document.createElement('div');
          box.className = 'asset-media-item';

          if (canPreviewInBrowser) {
            const img = document.createElement('img');
            img.className = 'asset-thumb';
            img.loading = 'eager';
            img.decoding = 'async';
            img.alt = filename;
            img.src = buildAssetObjectHref(asset, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
            box.appendChild(img);
          }

          const controls = document.createElement('div');
          controls.className = 'asset-path';
          controls.textContent = [filename, mediaType, byteSize ? `${byteSize} bytes` : '', source].filter(Boolean).join('\\n');
          box.appendChild(controls);

          if (!canPreviewInBrowser) {
            const unsupported = document.createElement('div');
            unsupported.className = 'asset-empty';
            unsupported.textContent = `当前浏览器不支持直接预览 ${mediaType || '该格式'}，请打开原图查看。`;
            box.appendChild(unsupported);
          }

          const openLink = document.createElement('a');
          openLink.className = 'asset-link';
          openLink.href = buildAssetObjectHref(asset);
          openLink.target = '_blank';
          openLink.rel = 'noreferrer';
          openLink.textContent = '打开原图';
          box.appendChild(openLink);

          entry.appendChild(box);
          return entry;
        }

        const info = document.createElement('div');
        info.className = 'asset-path';
        info.textContent = [filename, mediaType, byteSize ? `${byteSize} bytes` : '', source, pretty(asset)].filter(Boolean).join('\\n');
        entry.appendChild(info);
        return entry;
      }

      const text = String(item || '').trim();

      if (isPreviewableImagePath(text)) {
        const link = document.createElement('a');
        link.className = 'asset-media-item';
        link.href = buildAssetHref(text);
        link.target = '_blank';
        link.rel = 'noreferrer';

        const img = document.createElement('img');
        img.className = 'asset-thumb';
        img.loading = 'eager';
        img.decoding = 'async';
        img.src = buildAssetHref(text, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX });
        img.alt = text.split(/[\\/]/).pop() || 'asset image';
        link.appendChild(img);

        const pathText = document.createElement('div');
        pathText.className = 'asset-path';
        pathText.textContent = text;
        link.appendChild(pathText);
        entry.appendChild(link);
        return entry;
      }

      if (isLikelyFilePath(text)) {
        const link = document.createElement('a');
        link.className = 'asset-link';
        link.href = buildAssetHref(text);
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.textContent = '打开文件';
        entry.appendChild(link);
      }

      const pathText = document.createElement('div');
      pathText.className = 'asset-path';
      pathText.textContent = text;
      entry.appendChild(pathText);
      return entry;
    }

    function renderAssetList(title, items, options = {}) {
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

      if (String(options.kind || '') === 'image') {
        const grid = document.createElement('div');
        grid.className = 'asset-media-list';
        for (const item of list) {
          grid.appendChild(renderImageAssetItem(item));
        }
        card.appendChild(grid);
        return card;
      }

      const ul = document.createElement('ul');
      ul.className = 'asset-list';
      for (const item of list) {
        ul.appendChild(renderAssetTextItem(item));
      }
      card.appendChild(ul);
      return card;
    }

    function getLocalAssetPayload(node) {
      const vars = node && node.variables && typeof node.variables === 'object' ? node.variables : {};
      return vars.to_payload && typeof vars.to_payload === 'object' ? vars.to_payload : {};
    }

    function collectNodeAssets(node, options = {}) {
      const includeLists = Boolean(options.includeLists);
      const imageStoreLimit = Number.isFinite(options.imageStoreLimit) ? Number(options.imageStoreLimit) : Number.POSITIVE_INFINITY;
      const headers = [];
      const footers = [];
      const pageNumbers = [];
      const images = [];
      const seenHeaders = new Set();
      const seenFooters = new Set();
      const seenPageNumbers = new Set();
      const seenImages = new Set();
      let imageCount = 0;
      const stack = [node];

      function pushUniqueText(items, seen, out) {
        for (const item of toList(items)) {
          const text = String(item || '').trim();
          if (!text || seen.has(text)) {
            continue;
          }
          seen.add(text);
          if (includeLists) {
            out.push(text);
          }
        }
      }

      function pushUniqueImage(items) {
        for (const item of toList(items)) {
          const key = item && typeof item === 'object'
            ? String(item.asset_id || item.source || item.filename || JSON.stringify(item))
            : String(item || '').trim();
          if (!key || seenImages.has(key)) {
            continue;
          }
          seenImages.add(key);
          imageCount += 1;
          if (includeLists && images.length < imageStoreLimit) {
            images.push(item);
          }
        }
      }

      while (stack.length > 0) {
        const current = stack.pop();
        if (!current || typeof current !== 'object') {
          continue;
        }
        const metadata = current.metadata && typeof current.metadata === 'object' ? current.metadata : {};
        const payload = getLocalAssetPayload(current);
        pushUniqueText(payload.headers, seenHeaders, headers);
        pushUniqueText(payload.footers, seenFooters, footers);
        const localPageNumbers = toList(payload.page_numbers);
        if (localPageNumbers.length > 0) {
          pushUniqueText(localPageNumbers, seenPageNumbers, pageNumbers);
        } else {
          pushUniqueText(toList(metadata.page_number_hint || metadata.page), seenPageNumbers, pageNumbers);
        }
        pushUniqueImage(payload.images);
        const children = Array.isArray(current.children) ? current.children : [];
        for (let index = children.length - 1; index >= 0; index -= 1) {
          stack.push(children[index]);
        }
      }

      return {
        headers,
        footers,
        pageNumbers,
        images,
        headerCount: seenHeaders.size,
        footerCount: seenFooters.size,
        pageNumberCount: seenPageNumbers.size,
        imageCount,
      };
    }

    function renderAssets(node) {
      if (!assetsEl) {
        return;
      }
      assetsEl.innerHTML = '';
      if (!node || typeof node !== 'object') {
        return;
      }

      const metadata = node.metadata && typeof node.metadata === 'object' ? node.metadata : {};
  const payload = getLocalAssetPayload(node);
  const headers = toList(payload.headers).length > 0 ? toList(payload.headers) : toList(metadata.headers || metadata.header_text);
  const footers = toList(payload.footers).length > 0 ? toList(payload.footers) : toList(metadata.footers || metadata.footer_text);
  const localPageNumbers = toList(payload.page_numbers);
  const pageNumbers = localPageNumbers.length > 0 ? localPageNumbers : toList(metadata.page_number_hint || metadata.page);
  const images = toList(payload.images);
  const imageCount = images.length;

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
        ['Images', String(imageCount)],
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
      assetsEl.appendChild(renderAssetList('图片 Images', images, { kind: 'image' }));
    }

    function summarizeVarValue(value) {
      if (typeof value === 'string') {
        if (value.length <= 8000) {
          return value;
        }
        return `${value.slice(0, 8000)}\n\n...[truncated ${value.length - 8000} chars]`;
      }
      const text = pretty(value);
      if (text.length <= 16000) {
        return text;
      }
      return `${text.slice(0, 16000)}\n\n...[truncated ${text.length - 16000} chars]`;
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
        box.value = summarizeVarValue(v);
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
          const titleText = String(n.title || '');
          const safeTitle = escapeHtml(titleText);
          const titleMarkup = `${safeTitle}${safeTitle && badge ? ' ' : ''}${badge}`;
          btn.innerHTML = `<div>${titleMarkup}</div>`;
          sub.textContent = `${n.class_name || 'Page'} | ${n.category || 'unknown'}`;
          btn.appendChild(sub);
          btn.onclick = () => {
            selectedNodeId = n.id;
            renderTree(payload.tree || []);
            renderAssets(n);
            renderMarkdownText(n);
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

    function collectPreviewUrls(nodes) {
      const urls = [];
      const seen = new Set();

      function pushUrl(url) {
        const text = String(url || '').trim();
        if (!text || seen.has(text) || preloadedAssetUrls.has(text)) {
          return;
        }
        seen.add(text);
        urls.push(text);
      }

      const stack = Array.isArray(nodes) ? [...nodes] : [];
      while (stack.length > 0) {
        const current = stack.pop();
        if (!current || typeof current !== 'object') {
          continue;
        }
        const payload = getLocalAssetPayload(current);
        for (const item of toList(payload.images)) {
          if (isImageAssetObject(item)) {
            pushUrl(buildAssetObjectHref(item, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX }));
            continue;
          }
          const text = String(item || '').trim();
          if (isPreviewableImagePath(text)) {
            pushUrl(buildAssetHref(text, { preview: true, maxPx: IMAGE_PREVIEW_MAX_PX }));
          }
        }
        for (const child of Array.isArray(current.children) ? current.children : []) {
          stack.push(child);
        }
      }
      return urls;
    }

    async function preloadImagesForPayload(nodes, buildToken) {
      const urls = collectPreviewUrls(nodes);
      if (urls.length === 0) {
        return { total: 0, loaded: 0 };
      }

      let loaded = 0;
      const concurrency = 6;

      async function loadSingle(url) {
        try {
          const resp = await fetch(url, { cache: 'force-cache' });
          if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
          }
          await resp.blob();
          preloadedAssetUrls.add(url);
          loaded += 1;
        } catch (_) {
          // Ignore individual preview failures so the tree remains usable.
        }
      }

      for (let index = 0; index < urls.length; index += concurrency) {
        if (buildToken !== activeBuildToken) {
          break;
        }
        const batch = urls.slice(index, index + concurrency);
        await Promise.all(batch.map((url) => loadSingle(url)));
      }
      return { total: urls.length, loaded };
    }

    function beginBuildUi(fileName) {
      buildStartTs = Date.now();
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
    }

    async function buildDocTree() {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus('请先选择文件', true);
        return;
      }

      activeBuildToken += 1;
      const buildToken = activeBuildToken;
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
        `leaf_pages=${payload.leaf_mono_page_count || 0}`,
        `native_pages=${payload.native_page_count || 0}`,
        `runtime_native_pages=${payload.runtime_native_page_count || 0}`,
        `expected_pages=${payload.expected_page_count || 0}`,
        `aligned=${aligned ? 'yes' : 'no'}`,
        `pagination=${payload.pagination_mode || ''}`,
        `nodes=${nodes.length}`,
        `debug_json=${payload.debug_json_path || ''}`
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
        renderMarkdownText(nodes[0]);
        renderVars(nodes[0]);
      } else {
        if (assetsEl) assetsEl.innerHTML = '';
        if (renderMarkdownCard) renderMarkdownCard.hidden = true;
        if (renderMarkdownEl) renderMarkdownEl.innerHTML = '';
        if (varsEl) varsEl.innerHTML = '';
      }
      const preloadResult = await preloadImagesForPayload(payload.tree || [], buildToken);
      if (buildToken !== activeBuildToken) {
        return;
      }
      if (aligned) {
        setStatus(`构建完成，页数已对齐（mono=${payload.mono_page_count || 0}, leaf=${payload.leaf_mono_page_count || 0}，图片缓存 ${preloadResult.loaded}/${preloadResult.total}）`);
      } else if (Number(payload.native_page_count || 0) <= 0) {
        setStatus(`构建完成，节点数: ${nodes.length}（未获取物理页数，图片缓存 ${preloadResult.loaded}/${preloadResult.total}）`);
      }
    }

    fileInput.addEventListener('change', async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus('请选择文件，选择后将自动构建');
        return;
      }
      const requestToken = activeBuildToken + 1;
      setStatus(`已选择文件: ${file.name}，即将自动构建`);
      try {
        await buildDocTree();
      } catch (err) {
        setStatus(`构建失败: ${err.message || err}`, true);
      } finally {
        if (requestToken === activeBuildToken) {
          endBuildUi();
        }
      }
    });
  </script>
</body>
</html>
"""


def create_app() -> Any:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response, FileResponse
    from fastapi import File, UploadFile, HTTPException
    app = FastAPI(title="Doc Tree Debug GUI")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

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

    @app.get("/api/asset")
    async def asset(path: str, preview: int = 0, max_px: int = 360) -> Response:
      resolved = _resolve_workspace_asset_path(path)
      guessed_media_type = str(mimetypes.guess_type(str(resolved))[0] or "application/octet-stream").strip()
      preview_mode = bool(int(preview or 0))
      if not preview_mode and not _looks_like_vector_metafile(guessed_media_type, resolved.name):
        return FileResponse(resolved, headers={"Cache-Control": "public, max-age=3600"})
      data = resolved.read_bytes()
      data, media_type = _build_debug_image_bytes(
          data,
          guessed_media_type,
          filename=resolved.name,
          preview=preview_mode,
          max_px=max(64, min(1200, int(max_px or 360))),
      )
      filename = resolved.name.replace('"', "")
      headers = {
          "Content-Disposition": f'inline; filename="{filename}"',
          "Cache-Control": "public, max-age=3600",
      }
      return Response(content=data, media_type=media_type, headers=headers)

    @app.get("/api/asset-object")
    async def asset_object(asset_id: str, preview: int = 0, max_px: int = 360) -> Response:
      asset, data, media_type = _resolve_debug_image_asset(asset_id)
      data, media_type = _build_debug_image_bytes(
          data,
          media_type,
        filename=str(_asset_field(asset, "filename", "") or "asset.bin"),
          preview=bool(int(preview or 0)),
          max_px=max(64, min(1200, int(max_px or 360))),
      )
      filename = str(_asset_field(asset, "filename", "") or "asset.bin").replace('"', "")
      headers = {
          "Content-Disposition": f'inline; filename="{filename}"',
          "Cache-Control": "public, max-age=3600",
      }
      return Response(content=data, media_type=media_type, headers=headers)

    return app


def main() -> None:
  parser = argparse.ArgumentParser(description="文档树调试前端（浏览器选文件）")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=7869)
  parser.add_argument("--open-browser", action="store_true", help="启动后自动打开外部浏览器")
  parser.add_argument("--no-open", action="store_true", help="兼容旧参数：禁止自动打开浏览器")
  parser.add_argument("--input-file", default="", help="直接构建指定文件并输出结果，不启动 Web")
  parser.add_argument("--input-files", nargs="*", default=[], help="批量构建多个文件并输出结果，不启动 Web")
  parser.add_argument("--output-json", default="", help="配合 --input-file 使用：将 payload 写入指定 JSON 文件")
  parser.add_argument("--output-markdown", default="", help="配合 --input-file 使用：将整棵文档树导出为 Markdown 文件")
  parser.add_argument("--output-dir", default="", help="配合 --input-files 使用：将每个文件的 JSON/Markdown 写入该目录")
  parser.add_argument("--print-tree", action="store_true", help="配合 --input-file 使用：在终端打印 catalog 树")
  parser.add_argument("--include-build-debug", action="store_true", help="配合 --input-file 使用：输出构建过程中的 marker/range/section-map 调试信息")
  parser.add_argument("--assert-image-alignment", action="store_true", help="断言每个节点的 markdown 图片 token 数与 to_payload.images 数量一致")
  args = parser.parse_args()

  input_files = [str(item or "").strip() for item in list(args.input_files or []) if str(item or "").strip()]
  if input_files:
    output_dir = Path(str(args.output_dir or "tmp/doc_tree_debug_batch")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in input_files:
      payload, rag_doc = _build_payload_and_rag_doc_from_file(
        file_path,
        include_build_debug=bool(args.include_build_debug),
        emit_summary=True,
        emit_tree=False,
        emit_boundary_debug=False,
        assert_image_alignment=bool(args.assert_image_alignment),
      )
      stem = _sanitize_filename(Path(file_path).stem)
      json_path = output_dir / f"{stem}.json"
      markdown_path = output_dir / f"{stem}.md"
      json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
      markdown_path.write_text(_export_markdown_from_rag_doc(rag_doc), encoding="utf-8")
      print(f"[DocTreeDebug] Batch JSON: {json_path}")
      print(f"[DocTreeDebug] Batch Markdown: {markdown_path}")
    return

  input_file = str(args.input_file or "").strip()
  if input_file:
    payload, rag_doc = _build_payload_and_rag_doc_from_file(
      input_file,
      include_build_debug=bool(args.include_build_debug),
      emit_summary=True,
      emit_tree=bool(args.print_tree),
      emit_boundary_debug=bool(args.print_tree),
      assert_image_alignment=bool(args.assert_image_alignment),
    )
    output_json = str(args.output_json or "").strip()
    if output_json:
      out_path = Path(output_json).expanduser().resolve()
      out_path.parent.mkdir(parents=True, exist_ok=True)
      out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
      print(f"[DocTreeDebug] Payload JSON: {out_path}")
    output_markdown = str(args.output_markdown or "").strip()
    if output_markdown:
      markdown_path = Path(output_markdown).expanduser().resolve()
      markdown_path.parent.mkdir(parents=True, exist_ok=True)
      markdown_text = _export_markdown_from_rag_doc(rag_doc)
      markdown_path.write_text(markdown_text, encoding="utf-8")
      print(f"[DocTreeDebug] Exported Markdown: {markdown_path}")
    if args.print_tree:
      print("\n[DocTreeDebug] Tree Lines")
      for line in _catalog_tree_lines(payload.get("tree") or []):
        print(line)
    return

  url = f"http://{args.host}:{args.port}"
  print(f"[DocTreeDebug] URL: {url}")
  if not bool(args.no_open):
    try:
      webbrowser.open(url)
    except Exception:
      pass

  import uvicorn
  app = create_app()
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
