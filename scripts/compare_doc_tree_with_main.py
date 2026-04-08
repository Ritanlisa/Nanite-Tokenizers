from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

DEFAULT_FILES: List[str] = [
    "/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc",
    "/home/ritanlisa/文档/初步验收与试运行分册-6-硬件维护手册 - 1227.doc",
    "/home/ritanlisa/文档/浪潮虚拟化InCloud Sphere 6.5.1运维手册.pdf",
    "/home/ritanlisa/文档/LID.pdf",
    "/home/ritanlisa/文档/TBP.pdf",
]


def _run_git_show(path_in_repo: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "show", f"main:{path_in_repo}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return str(proc.stdout or "")


def _load_main_class(path_in_repo: str, class_name: str) -> type:
    source = _run_git_show(path_in_repo)
    module_name = f"_main_branch_{path_in_repo.replace('/', '_').replace('.', '_')}"
    module = types.ModuleType(module_name)
    module.__file__ = f"main:{path_in_repo}"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__ or path_in_repo, "exec"), module.__dict__)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"无法从 main:{path_in_repo} 加载类 {class_name}")
    return cls


def _main_class_for_ext(ext: str) -> type:
    ext_norm = str(ext or "").strip().lower()
    if ext_norm == ".pdf":
        return _load_main_class("rag/document_pdf.py", "PDFRAGDocument")
    if ext_norm == ".doc":
        return _load_main_class("rag/document_doc.py", "DocRAGDocument")
    if ext_norm == ".docx":
        return _load_main_class("rag/document_docx.py", "DocxRAGDocument")
    if ext_norm in {".xlsx", ".xls"}:
        return _load_main_class("rag/document_spreadsheet.py", "SpreadsheetRAGDocument")
    return _load_main_class("rag/document_text.py", "TextRAGDocument")


def _to_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        number = int(float(str(value).strip()))
    except Exception:
        return None
    return number if number > 0 else None


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _derive_ranges_from_markers(markers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not markers:
        return []

    ranges: List[Dict[str, Any]] = []
    for idx, marker in enumerate(markers):
        title = str(marker.get("title") or "").strip()
        if not title:
            continue
        start_page = _to_positive_int(marker.get("page"))
        if start_page is None:
            continue
        level = _to_positive_int(marker.get("level")) or 1

        next_boundary = None
        cur_level = level
        for probe in range(idx + 1, len(markers)):
            probe_level = _to_positive_int(markers[probe].get("level")) or cur_level
            if probe_level > cur_level:
                continue
            probe_page = _to_positive_int(markers[probe].get("page"))
            if probe_page is None:
                continue
            next_boundary = probe_page
            break

        end_page = start_page if next_boundary is None else max(start_page, next_boundary - 1)
        ranges.append(
            {
                "title": title,
                "level": max(1, min(level, 6)),
                "start": start_page,
                "end": end_page,
            }
        )
    return ranges


def _derive_hierarchical_paths_from_ranges(ranges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ranges:
        return []
    stack: List[Tuple[int, str]] = []
    with_path: List[Dict[str, Any]] = []
    for item in ranges:
        level = int(item.get("level") or 1)
        title = str(item.get("title") or "").strip() or "Untitled"
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_path = "/".join(part for _, part in stack)
        path = f"{parent_path}/{title}" if parent_path else title
        with_path.append({**item, "path": path})
        stack.append((level, title))
    return with_path


def _build_main_baseline_page_assignment(
    main_doc: Any,
    total_pages: int,
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    page_map: Dict[int, str] = {}
    debug: Dict[str, Any] = {
        "ranges_source": "catalog",
        "ranges_count": 0,
        "markers_count": 0,
    }

    # Primary baseline: main branch chunk metadata (actual runtime output semantics).
    chunk_map: Dict[int, str] = {}
    chunk_best_score: Dict[int, Tuple[int, int]] = {}
    for chunk in list(getattr(main_doc, "chunk_documents", []) or []):
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        section_path = str(metadata.get("section_path") or metadata.get("section_title") or "").strip()
        if not section_path:
            continue
        p_start = _to_positive_int(metadata.get("section_start_page"))
        p_end = _to_positive_int(metadata.get("section_end_page"))
        p_page = _to_positive_int(metadata.get("page"))
        start = p_start or p_page
        end = p_end or p_start or p_page
        if start is None or end is None:
            continue
        if end < start:
            end = start
        depth = max(1, len([part for part in section_path.split(" > ") if part.strip()]))
        span = max(1, int(end) - int(start) + 1)
        for page_idx in range(int(start), int(end) + 1):
            score = (depth, -span)
            old_score = chunk_best_score.get(page_idx)
            if old_score is None or score > old_score:
                chunk_map[page_idx] = section_path
                chunk_best_score[page_idx] = score

    if chunk_map:
        debug["ranges_source"] = "main_chunk_metadata"
        debug["chunk_map_pages"] = len(chunk_map)
        for page in range(1, max(1, int(total_pages)) + 1):
            mapped = str(chunk_map.get(page) or "").strip()
            page_map[page] = mapped or "Document"
        debug["ranges"] = []
        return page_map, debug

    markers: List[Dict[str, Any]] = []
    try:
        cleaned_text = str(getattr(main_doc, "cleaned_text", "") or "")
        if cleaned_text and hasattr(main_doc, "_build_native_page_map") and hasattr(main_doc, "_build_markers"):
            lines = cleaned_text.split("\n")
            native_page_map, has_native_marker, has_explicit_page_number = main_doc._build_native_page_map(lines)
            page_by_line, _ = main_doc.resolve_page_map(lines, native_page_map, has_native_marker)
            markers = list(main_doc._build_markers(lines, page_by_line, has_explicit_page_number) or [])
    except Exception as exc:
        debug["marker_error"] = str(exc)
        markers = []

    if markers:
        ranges = _derive_ranges_from_markers(markers)
        ranges = _derive_hierarchical_paths_from_ranges(ranges)
        debug["ranges_source"] = "main_markers"
        debug["markers_count"] = len(markers)
    else:
        ranges = []
        raw_catalog = list(getattr(main_doc, "catalog", []) or [])
        for item in raw_catalog:
            title = str(getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "") or "").strip()
            start = _to_positive_int(getattr(item, "page", None) if not isinstance(item, dict) else item.get("page"))
            end = _to_positive_int(getattr(item, "end_page", None) if not isinstance(item, dict) else item.get("end_page"))
            if not title or start is None:
                continue
            ranges.append(
                {
                    "title": title,
                    "start": start,
                    "end": end if end is not None else start,
                    "level": 1,
                    "path": title,
                }
            )

    debug["ranges_count"] = len(ranges)

    for page in range(1, max(1, int(total_pages)) + 1):
        chosen: Optional[str] = None
        for item in ranges:
            start = int(item.get("start") or page)
            end = int(item.get("end") or start)
            if start <= page <= end:
                chosen = str(item.get("path") or item.get("title") or "").strip() or "Document"
                break
        if chosen is None and ranges:
            chosen = str(ranges[-1].get("path") or ranges[-1].get("title") or "").strip() or "Document"
        if chosen is None:
            chosen = "Document"
        page_map[page] = chosen

    debug["ranges"] = ranges
    return page_map, debug


def _extract_current_page_assignment(rag_doc: Any) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    mono_pages = list(getattr(rag_doc, "get_mono_pages")() or [])
    for fallback_idx, page in enumerate(mono_pages, start=1):
        metadata = dict(getattr(page, "metadata", {}) or {})
        page_no = _to_positive_int(metadata.get("page")) or fallback_idx
        section_path = str(metadata.get("section_path") or metadata.get("section_title") or getattr(page, "title", "") or "").strip()
        mapping[int(page_no)] = section_path or "Document"
    return mapping


def _extract_boundaries_from_page_map(page_map: Dict[int, str], max_page: int) -> Set[int]:
    boundaries: Set[int] = set()
    prev = ""
    for page in range(1, max_page + 1):
        cur = str(page_map.get(page, "")).strip()
        if page == 1 or _normalize_name(cur) != _normalize_name(prev):
            boundaries.add(page)
        prev = cur
    return boundaries


def compare_one(file_path: str) -> Dict[str, Any]:
    from rag.documents import create_rag_db_document, load_single_file_document, stable_doc_id

    source_file = Path(file_path).expanduser().resolve()
    result: Dict[str, Any] = {
        "file": str(source_file),
        "exists": source_file.exists(),
    }
    if (not source_file.exists()) or (not source_file.is_file()):
        result["status"] = "missing"
        return result

    ext = source_file.suffix.lower()
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        result["status"] = "unsupported"
        result["ext"] = ext
        return result

    loaded_doc = load_single_file_document(str(source_file), SUPPORTED_RAG_EXTENSIONS)
    if loaded_doc is None:
        result["status"] = "load_failed"
        return result

    doc_id = stable_doc_id(loaded_doc)
    current_doc = create_rag_db_document(loaded_doc, stable_doc_id=doc_id).build()
    current_page_map = _extract_current_page_assignment(current_doc)
    current_page_count = len(current_page_map)

    main_cls = _main_class_for_ext(ext)
    main_doc = main_cls(loaded_doc, doc_id).build()
    baseline_map, baseline_debug = _build_main_baseline_page_assignment(main_doc, total_pages=max(1, current_page_count))

    max_page = max(max(current_page_map.keys(), default=0), max(baseline_map.keys(), default=0), 1)
    matched = 0
    page_rows: List[Dict[str, Any]] = []
    for page in range(1, max_page + 1):
        current_name = str(current_page_map.get(page, "")).strip()
        baseline_name = str(baseline_map.get(page, "")).strip()
        is_match = _normalize_name(current_name) == _normalize_name(baseline_name)
        if is_match:
            matched += 1
        page_rows.append(
            {
                "page": page,
                "current": current_name,
                "main_baseline": baseline_name,
                "match": bool(is_match),
            }
        )

    match_ratio = matched / float(max_page)

    current_boundaries = _extract_boundaries_from_page_map(current_page_map, max_page)
    baseline_boundaries = _extract_boundaries_from_page_map(baseline_map, max_page)
    boundary_overlap = current_boundaries & baseline_boundaries
    boundary_precision = (
        len(boundary_overlap) / float(len(current_boundaries))
        if current_boundaries
        else 0.0
    )
    boundary_recall = (
        len(boundary_overlap) / float(len(baseline_boundaries))
        if baseline_boundaries
        else 0.0
    )
    if boundary_precision + boundary_recall > 0.0:
        boundary_f1 = 2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
    else:
        boundary_f1 = 0.0

    result.update(
        {
            "status": "ok",
            "ext": ext,
            "current_page_count": current_page_count,
            "current_catalog_count": len(list(getattr(current_doc, "catalog", []) or [])),
            "current_pagination_mode": str(getattr(current_doc, "pagination_mode", "") or ""),
            "main_baseline_catalog_count": len(list(getattr(main_doc, "catalog", []) or [])),
            "main_baseline_debug": baseline_debug,
            "total_compared_pages": max_page,
            "matched_pages": matched,
            "match_ratio": match_ratio,
            "boundary_count_current": len(current_boundaries),
            "boundary_count_main": len(baseline_boundaries),
            "boundary_overlap": len(boundary_overlap),
            "boundary_precision": boundary_precision,
            "boundary_recall": boundary_recall,
            "boundary_f1": boundary_f1,
            "page_diffs": [row for row in page_rows if not row["match"]],
            "sample_page_rows": page_rows[:20],
        }
    )
    return result


def main() -> None:
    files = list(DEFAULT_FILES)
    reports: List[Dict[str, Any]] = []
    for file in files:
        print(f"\n=== Compare: {file}")
        item = compare_one(file)
        reports.append(item)
        status = str(item.get("status") or "")
        print(f"status={status}")
        if status != "ok":
            continue
        print(
            " | ".join(
                [
                    f"ext={item.get('ext')}",
                    f"current_pages={item.get('current_page_count')}",
                    f"current_catalog={item.get('current_catalog_count')}",
                    f"baseline_catalog={item.get('main_baseline_catalog_count')}",
                    f"matched={item.get('matched_pages')}/{item.get('total_compared_pages')}",
                    f"match_ratio={float(item.get('match_ratio') or 0.0):.4f}",
                    f"boundary_f1={float(item.get('boundary_f1') or 0.0):.4f}",
                ]
            )
        )

    output_dir = PROJECT_ROOT / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "doc_tree_vs_main_report.json"
    output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细报告已写入: {output_path}")


if __name__ == "__main__":
    main()
