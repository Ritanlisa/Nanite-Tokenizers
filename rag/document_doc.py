from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from typing import Any, Dict, List

from llama_index.core import Document

from rag.document_interface import Content, PageAssets, RAG_DB_Document
from rag.preprocessor import clean_document


class DocRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        super().__init__(source_document, stable_doc_id)
        self.source_extension = self.source_extension or ".doc"
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n\f\n"))

    def build(self) -> RAG_DB_Document:
        if not self.cleaned_text.strip():
            self.set_page_nodes([])
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            self.pagination_mode = "doc-page-tree"
            return self

        raw_pages = [part.strip() for part in self.cleaned_text.split("\f")]
        page_texts = [part for part in raw_pages]
        if not page_texts:
            page_texts = [self.cleaned_text.strip()]

        ranges = self._extract_structured_catalog_ranges(len(page_texts))

        page_nodes: List[Content] = []
        chunks: List[Document] = []
        page_layouts = list(self.metadata.get("page_layout") or [])
        for page_idx, page_text in enumerate(page_texts, start=1):
            section = self._pick_catalog_section(page_idx, ranges)
            if section:
                section_title = str(section.get("title"))
                section_path = f"L{section['level']}/{section_title}"
            else:
                section_title = self._guess_page_title(page_text, page_idx)
                section_path = section_title

            page_layout = page_layouts[page_idx - 1] if page_idx - 1 < len(page_layouts) and isinstance(page_layouts[page_idx - 1], dict) else {}
            headers = [str(x).strip() for x in list(page_layout.get("headers") or []) if str(x).strip()]
            footers = [str(x).strip() for x in list(page_layout.get("footers") or []) if str(x).strip()]
            images = [str(x).strip() for x in list(page_layout.get("images") or []) if str(x).strip()]
            page_number_hint = str(page_layout.get("page_number") or "").strip()
            assets = PageAssets(headers=headers, footers=footers, page_numbers=[page_number_hint] if page_number_hint else [], images=images)
            page_meta: Dict[str, Any] = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": f"doc-page-{page_idx}",
                "section_title": section_title,
                "section_path": section_path,
                "section_start_page": page_idx,
                "section_end_page": page_idx,
                "page": page_idx,
                "header_text": "\n".join(headers),
                "footer_text": "\n".join(footers),
                "page_number_hint": page_number_hint,
                "image_count": len(images),
            }
            node = Content(title="Content", markdown_text=page_text, assets=assets, metadata=page_meta)
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_text), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::doc-page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

        self.set_page_nodes(self.build_catalog_tree(page_nodes, ranges))
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "doc-page-tree"
        self.catalog = self.catalog_payload()
        return self

    def _extract_structured_catalog_ranges(self, total_pages: int) -> List[Dict[str, Any]]:
        priority_keys: List[List[str]] = [["native_catalog"], ["style_catalog"], ["font_catalog"]]
        for keys in priority_keys:
            markers = self._extract_markers_from_metadata(keys)
            if not markers:
                continue
            ranges = self._ranges_from_raw_markers(markers, total_pages)
            if ranges:
                return ranges
        return []

    def _ranges_from_raw_markers(self, markers: List[Dict[str, Any]], total_pages: int) -> List[Dict[str, Any]]:
        ranges: List[Dict[str, Any]] = []
        if not markers:
            return ranges

        for idx, marker in enumerate(markers):
            title = str(marker.get("title") or "")
            start = self.coerce_page_number(marker.get("page"))
            level = self.coerce_page_number(marker.get("level")) or 1
            if start is None:
                continue

            next_start = self._next_boundary_start_page(markers, idx)
            if next_start is None:
                end = max(int(start), int(total_pages or start))
            else:
                end = max(int(start), int(next_start) - 1)

            ranges.append(
                {
                    "title": title,
                    "start": int(start),
                    "end": int(end),
                    "level": max(1, min(int(level), 6)),
                }
            )
        return ranges

    def _extract_main_catalog_ranges(self, page_texts: List[str], total_pages: int) -> List[Dict[str, Any]]:
        main_cls = self._load_main_pdf_class()
        if main_cls is None:
            return []
        try:
            merged_text = "\n\f\n".join(str(item or "") for item in page_texts)
            temp_doc = main_cls(self.source_document, str(self.base_doc_id))
            temp_doc.cleaned_text = merged_text
            temp_doc.build()

            catalog_rows: List[Dict[str, Any]] = []
            for item in list(getattr(temp_doc, "catalog", []) or []):
                if isinstance(item, dict):
                    title = str(item.get("title") or "").strip()
                    start = self.coerce_page_number(item.get("page"))
                    end = self.coerce_page_number(item.get("end_page"))
                else:
                    title = str(getattr(item, "title", "") or "").strip()
                    start = self.coerce_page_number(getattr(item, "page", None))
                    end = self.coerce_page_number(getattr(item, "end_page", None))
                if not title or start is None:
                    continue
                catalog_rows.append({"title": title, "start": int(start), "end": int(end) if end is not None else None})

            if not catalog_rows:
                return []
            catalog_rows.sort(key=lambda x: int(x["start"]))

            ranges: List[Dict[str, Any]] = []
            for idx, row in enumerate(catalog_rows):
                title = self._clean_range_title(row["title"])
                if not title:
                    continue
                start = int(row["start"])
                end = row["end"]
                if end is None:
                    next_start = catalog_rows[idx + 1]["start"] if idx + 1 < len(catalog_rows) else total_pages
                    end = max(start, int(next_start) - 1 if idx + 1 < len(catalog_rows) else int(total_pages))
                level = self.coerce_page_number(row.get("level")) or self._heading_level(title) or 1
                ranges.append({"title": title, "start": start, "end": int(end), "level": max(1, min(int(level), 6))})

            return ranges
        except Exception:
            return []

    @staticmethod
    def _load_main_pdf_class() -> type | None:
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            proc = subprocess.run(
                ["git", "-C", repo_root, "show", "main:rag/document_pdf.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            source = str(proc.stdout or "")
            if not source.strip():
                return None
            module_name = "_main_branch_rag_document_pdf_runtime_for_doc"
            module = types.ModuleType(module_name)
            module.__file__ = "main:rag/document_pdf.py"
            sys.modules[module_name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            return getattr(module, "PDFRAGDocument", None)
        except Exception:
            return None

    def _extract_main_compatible_markers(self, page_texts: List[str]) -> List[Dict[str, Any]]:
        main_cls = self._load_main_pdf_class()
        if main_cls is None:
            return []
        try:
            merged_text = "\n\f\n".join(str(item or "") for item in page_texts)
            temp_doc = main_cls(self.source_document, str(self.base_doc_id))
            temp_doc.cleaned_text = merged_text
            lines = temp_doc.cleaned_text.split("\n")
            native_page_map, has_native_marker, has_explicit_page_number = temp_doc._build_native_page_map(lines)
            page_by_line, _ = temp_doc.resolve_page_map(lines, native_page_map, has_native_marker)
            markers = list(temp_doc._build_markers(lines, page_by_line, has_explicit_page_number) or [])
            return [dict(item) for item in markers if isinstance(item, dict)]
        except Exception:
            return []

    def _extract_main_page_section_map(self, page_texts: List[str]) -> Dict[int, str]:
        main_cls = self._load_main_pdf_class()
        if main_cls is None:
            return {}
        try:
            merged_text = "\n\f\n".join(str(item or "") for item in page_texts)
            temp_doc = main_cls(self.source_document, str(self.base_doc_id))
            temp_doc.cleaned_text = merged_text
            temp_doc.build()

            best: Dict[int, tuple[int, int, str]] = {}
            for chunk in list(getattr(temp_doc, "chunk_documents", []) or []):
                metadata = dict(getattr(chunk, "metadata", {}) or {})
                section_path = str(metadata.get("section_path") or metadata.get("section_title") or "").strip()
                if not section_path:
                    continue
                p_start = self.coerce_page_number(metadata.get("section_start_page"))
                p_end = self.coerce_page_number(metadata.get("section_end_page"))
                p_page = self.coerce_page_number(metadata.get("page"))
                start = p_start or p_page
                end = p_end or p_start or p_page
                if start is None or end is None:
                    continue
                if end < start:
                    end = start
                depth = max(1, len([part for part in section_path.split(" > ") if part.strip()]))
                span = max(1, int(end) - int(start) + 1)
                score = (depth, -span)
                for page_idx in range(int(start), int(end) + 1):
                    old = best.get(page_idx)
                    if old is None or score > (old[0], old[1]):
                        best[page_idx] = (score[0], score[1], section_path)
            return {page_idx: item[2] for page_idx, item in best.items()}
        except Exception:
            return {}

    @staticmethod
    def _ranges_from_page_section_map(page_section_map: Dict[int, str], total_pages: int) -> List[Dict[str, Any]]:
        if total_pages <= 0:
            return []

        ranges: List[Dict[str, Any]] = []
        active: Dict[str, Dict[str, Any]] = {}

        for page in range(1, int(total_pages) + 1):
            path = str(page_section_map.get(page) or "").strip()
            current: Dict[str, Dict[str, Any]] = {}
            if path and path.lower() != "document":
                parts = [part.strip() for part in path.split(" > ") if part.strip()]
                for level, _ in enumerate(parts, start=1):
                    prefix = " > ".join(parts[:level])
                    current[prefix] = {
                        "title": parts[level - 1],
                        "level": max(1, min(level, 6)),
                    }

            for prefix in list(active.keys()):
                if prefix in current:
                    continue
                item = active.pop(prefix)
                ranges.append(
                    {
                        "title": str(item["title"]),
                        "start": int(item["start"]),
                        "end": int(item["end"]),
                        "level": int(item["level"]),
                    }
                )

            for prefix, meta in current.items():
                if prefix in active:
                    active[prefix]["end"] = page
                else:
                    active[prefix] = {
                        "title": meta["title"],
                        "level": meta["level"],
                        "start": page,
                        "end": page,
                    }

        for item in active.values():
            ranges.append(
                {
                    "title": str(item["title"]),
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "level": int(item["level"]),
                }
            )

        ranges.sort(key=lambda x: (int(x["start"]), int(x["level"]), str(x["title"])))
        return ranges

    @staticmethod
    def _guess_page_title(page_text: str, page_number: int) -> str:
        lines = [line.strip() for line in str(page_text or "").splitlines() if line and line.strip()]
        if not lines:
            if page_number <= 2:
                return "封面"
            return f"Page {page_number}"
        first = re.sub(r"[·•.]{6,}.*$", "", lines[0]).strip()
        if "目录" in first or first.lower().startswith("contents"):
            return "目录"
        if page_number <= 2 and len(lines) <= 3:
            return "封面"
        if not first:
            return f"Page {page_number}"
        if len(first) > 64:
            return f"Page {page_number}"
        if re.search(r"\[[^\]]{1,32}\]", first) and len(first) > 36:
            return f"Page {page_number}"
        if len(first) <= 60:
            return first
        return f"Page {page_number}"

    @staticmethod
    def _ranges_from_main_markers(markers: List[Dict[str, Any]], total_pages: int) -> List[Dict[str, Any]]:
        if not markers:
            return []
        ranges: List[Dict[str, Any]] = []
        for idx, marker in enumerate(markers):
            title = str(marker.get("title") or "").strip()
            start = RAG_DB_Document.coerce_page_number(marker.get("page"))
            level = RAG_DB_Document.coerce_page_number(marker.get("level")) or 1
            if (not title) or (start is None) or start <= 0:
                continue
            cur_level = int(level)
            next_start = None
            for probe in range(idx + 1, len(markers)):
                probe_level = RAG_DB_Document.coerce_page_number(markers[probe].get("level")) or cur_level
                if probe_level > cur_level:
                    continue
                probe_page = RAG_DB_Document.coerce_page_number(markers[probe].get("page"))
                if probe_page is None or probe_page <= 0:
                    continue
                next_start = probe_page
                break
            end = max(start, int(total_pages or start)) if next_start is None else max(start, int(next_start) - 1)
            ranges.append(
                {
                    "title": title,
                    "start": int(start),
                    "end": int(end),
                    "level": max(1, min(int(level), 6)),
                }
            )
        return ranges

