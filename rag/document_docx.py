from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from rag.document_interface import Content, PageAssets, RAG_DB_Document
from rag.preprocessor import clean_document

try:
    import pymupdf4llm  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pymupdf4llm = None


class DocxRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        super().__init__(source_document, stable_doc_id)
        self.source_extension = self.source_extension or ".docx"
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n\f\n"))

    def _extract_with_pymupdf4llm(self) -> Optional[List[str]]:
        if pymupdf4llm is None:
            return None
        file_path = str(self.metadata.get("file_name") or "").strip()
        if not file_path or not os.path.isfile(file_path):
            return None
        try:
            chunks = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                extract_words=True,
                show_progress=False,
            )
        except Exception:
            return None
        if not isinstance(chunks, list):
            return None
        page_texts: List[str] = []
        for item in chunks:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            page_texts.append(text)
        return page_texts or None

    def build(self) -> RAG_DB_Document:
        if not self.cleaned_text.strip():
            self.set_page_nodes([])
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            self.pagination_mode = "docx-page-tree"
            return self
        raw_pages = [part.strip() for part in self.cleaned_text.split("\f")]
        page_texts = [part for part in raw_pages]
        if not page_texts:
            page_texts = [self.cleaned_text.strip()]
        source_page_count = max(1, len(page_texts))
        source_page_texts = list(page_texts)
        page_texts = self.enforce_native_page_count(page_texts)

        source_ranges = self._extract_structured_catalog_ranges(source_page_texts, source_page_count, source_page_count)
        page_signals = self.build_page_signals(page_texts, source_ranges)
        source_page_map = self._map_target_to_source_pages(source_page_count, len(page_texts))

        page_nodes: List[Content] = []
        chunks: List[Document] = []
        page_layouts = list(self.metadata.get("page_layout") or [])
        for page_idx, page_text in enumerate(page_texts, start=1):
            page_markdown = self._render_tabular_lines_to_markdown(page_text)
            signal = page_signals[page_idx - 1] if page_idx - 1 < len(page_signals) else {}
            logical_page = int(signal.get("logical_page_number") or page_idx)
            source_page = source_page_map[page_idx - 1] if page_idx - 1 < len(source_page_map) else page_idx
            section = self._pick_catalog_section(source_page, source_ranges)
            if section:
                section_title = str(section.get("title"))
                section_path = f"L{section['level']}/{section_title}"
            else:
                section_title = self._guess_page_title(page_text, page_idx)
                section_path = section_title

            page_layout = page_layouts[page_idx - 1] if page_idx - 1 < len(page_layouts) and isinstance(page_layouts[page_idx - 1], dict) else {}
            headers = [str(x).strip() for x in list(page_layout.get("headers") or []) if str(x).strip()]
            if not headers and str(signal.get("header_text") or "").strip():
                headers = [str(signal.get("header_text") or "").strip()]
            footers = [str(x).strip() for x in list(page_layout.get("footers") or []) if str(x).strip()]
            if not footers and str(signal.get("footer_text") or "").strip():
                footers = [str(signal.get("footer_text") or "").strip()]
            images = [str(x).strip() for x in list(page_layout.get("images") or []) if str(x).strip()]
            page_number_hint = str(page_layout.get("page_number") or "").strip()
            if not page_number_hint:
                page_number_hint = str(signal.get("page_number_hint") or "").strip()
            assets = PageAssets(headers=headers, footers=footers, page_numbers=[page_number_hint] if page_number_hint else [], images=images)

            page_meta: Dict[str, Any] = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": f"docx-page-{page_idx}",
                "section_title": section_title,
                "section_path": section_path,
                "section_start_page": page_idx,
                "section_end_page": page_idx,
                "page": page_idx,
                "logical_page": logical_page,
                "source_page": source_page,
                "header_text": "\n".join(headers),
                "footer_text": "\n".join(footers),
                "page_number_hint": page_number_hint,
                "image_count": len(images),
            }
            node = Content(title=section_title, markdown_text=page_markdown, assets=assets, metadata=page_meta)
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_markdown), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::docx-page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

        physical_ranges = self._materialize_physical_ranges_from_pages(page_nodes, source_ranges)
        self.set_page_nodes(self.build_catalog_tree(page_nodes, physical_ranges))
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "docx-page-tree"
        self.catalog = self.catalog_payload()
        return self

    @staticmethod
    def _map_target_to_source_pages(source_count: int, target_count: int) -> List[int]:
        src = max(1, int(source_count or 1))
        dst = max(1, int(target_count or 1))
        mapping: List[int] = []
        for idx in range(dst):
            start = int((idx * src) // dst)
            mapping.append(min(src, max(1, start + 1)))
        return mapping

    def _materialize_physical_ranges_from_pages(
        self,
        pages: List[Content],
        source_ranges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        buckets: Dict[int, List[int]] = {}
        for page in pages:
            meta = dict(getattr(page, "metadata", {}) or {})
            page_no = int(self.coerce_page_number(meta.get("page")) or 0)
            source_no = int(self.coerce_page_number(meta.get("source_page")) or page_no)
            if page_no <= 0:
                continue
            section = self._pick_catalog_section(source_no, source_ranges)
            if not section:
                continue
            order = int(section.get("order", 0))
            buckets.setdefault(order, []).append(page_no)

        projected: List[Dict[str, Any]] = []
        for item in source_ranges:
            order = int(item.get("order", 0))
            pages_in = sorted(set(buckets.get(order, [])))
            if not pages_in:
                continue
            projected.append(
                {
                    "title": str(item.get("title") or ""),
                    "start": int(min(pages_in)),
                    "end": int(max(pages_in)),
                    "level": int(item.get("level") or 1),
                    "order": order,
                }
            )
        return projected

    @staticmethod
    def _render_tabular_lines_to_markdown(text: str) -> str:
        lines = str(text or "").splitlines()
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
            if len(cells) >= 2:
                table_rows: List[List[str]] = [cells]
                i += 1
                while i < len(lines):
                    next_cells = [cell.strip() for cell in lines[i].split("\t") if cell.strip()]
                    if len(next_cells) >= 2:
                        table_rows.append(next_cells)
                        i += 1
                    else:
                        break

                width = max(len(row) for row in table_rows)
                norm = [row + [""] * (width - len(row)) for row in table_rows]
                header = "| " + " | ".join(cell.replace("|", "\\|") for cell in norm[0]) + " |"
                sep = "| " + " | ".join(["---"] * width) + " |"
                out.append(header)
                out.append(sep)
                for row in norm[1:]:
                    out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
                continue

            out.append(line)
            i += 1
        return "\n".join(out)

    def _extract_structured_catalog_ranges(
        self,
        page_texts: List[str],
        total_pages: int,
        source_page_count: int,
    ) -> List[Dict[str, Any]]:
        priority_keys: List[List[str]] = [["native_catalog"], ["style_catalog"], ["font_catalog"]]
        for keys in priority_keys:
            markers = self._extract_markers_from_metadata(keys)
            if not markers:
                continue

            top_ranges = self._build_top_level_ranges_from_first_line_hits(markers, page_texts, total_pages)
            if top_ranges:
                return top_ranges

            markers = self._rescale_marker_pages_if_needed(markers, source_page_count, total_pages)
            markers = self._align_marker_pages_with_top_title_hits(markers, page_texts, total_pages)
            ranges = self._ranges_from_raw_markers(markers, total_pages)
            if ranges and self._has_informative_page_span(ranges, total_pages):
                return ranges
        return []

    def _build_top_level_ranges_from_first_line_hits(
        self,
        markers: List[Dict[str, Any]],
        page_texts: List[str],
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        top_markers: List[Dict[str, Any]] = []
        for item in markers:
            level = int(self.coerce_page_number(item.get("level")) or 1)
            if level != 1:
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            top_markers.append(item)
        if len(top_markers) < 3:
            return []

        first_lines: List[str] = []
        for text in page_texts:
            line = ""
            for raw in str(text or "").splitlines():
                raw = raw.strip()
                if raw:
                    line = raw
                    break
            first_lines.append(self._normalize_title(line))

        hits: List[Dict[str, Any]] = []
        for marker in top_markers:
            title = str(marker.get("title") or "").strip()
            key = self._normalize_title(title)
            if not key:
                continue
            hit_page = None
            for idx, line in enumerate(first_lines, start=1):
                if line.startswith(key):
                    hit_page = idx
                    break
            if hit_page is None:
                continue
            hits.append({
                "title": title,
                "start": int(hit_page),
                "level": 1,
                "order": int(marker.get("order", len(hits))),
            })

        if len(hits) < 3:
            return []

        hits.sort(key=lambda item: int(item["start"]))
        ranges: List[Dict[str, Any]] = []
        for idx, item in enumerate(hits):
            start = int(item["start"])
            next_start = int(hits[idx + 1]["start"]) if idx + 1 < len(hits) else int(total_pages)
            end = max(start, next_start - 1 if idx + 1 < len(hits) else int(total_pages))
            ranges.append(
                {
                    "title": str(item["title"]),
                    "start": start,
                    "end": end,
                    "level": 1,
                    "order": int(item.get("order", idx)),
                }
            )
        return ranges

    def _align_marker_pages_with_top_title_hits(
        self,
        markers: List[Dict[str, Any]],
        page_texts: List[str],
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        if not markers or not page_texts:
            return markers

        top_markers: List[Dict[str, Any]] = []
        for item in markers:
            level = int(self.coerce_page_number(item.get("level")) or 1)
            title = str(item.get("title") or "").strip()
            if level != 1 or not title or self._is_noise_heading_line(title):
                continue
            top_markers.append(item)

        if len(top_markers) < 3:
            return markers

        page_lines: List[List[str]] = []
        for text in page_texts:
            lines = [self._normalize_title(line) for line in str(text or "").splitlines()[:12] if line.strip()]
            page_lines.append(lines)

        deltas: List[int] = []
        for marker in top_markers:
            old_page = int(self.coerce_page_number(marker.get("page")) or 0)
            if old_page <= 0:
                continue
            key = self._normalize_title(str(marker.get("title") or ""))
            if len(key) < 4:
                continue
            hit_page = None
            for idx, lines in enumerate(page_lines, start=1):
                if any(line == key or line.startswith(key) for line in lines):
                    hit_page = idx
                    break
            if hit_page is None:
                continue
            deltas.append(int(hit_page) - int(old_page))

        if len(deltas) < 3:
            return markers
        deltas.sort()
        if deltas[-1] - deltas[0] > 6:
            return markers
        mid = len(deltas) // 2
        shift = deltas[mid] if len(deltas) % 2 == 1 else int(round((deltas[mid - 1] + deltas[mid]) / 2.0))
        if shift == 0:
            return markers

        aligned: List[Dict[str, Any]] = []
        for item in markers:
            row = dict(item)
            page = int(self.coerce_page_number(row.get("page")) or 0)
            if page > 0:
                row["page"] = max(1, min(int(total_pages), page + shift))
            aligned.append(row)
        return aligned

    def _rescale_marker_pages_if_needed(
        self,
        markers: List[Dict[str, Any]],
        source_page_count: int,
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        if not markers:
            return markers

        src_total = max(1, int(source_page_count or 1))
        dst_total = max(1, int(total_pages or 1))
        if dst_total <= src_total:
            return markers

        min_marker_page = min(int(self.coerce_page_number(item.get("page")) or 0) for item in markers)
        max_marker_page = max(int(self.coerce_page_number(item.get("page")) or 0) for item in markers)
        if min_marker_page <= 0 or max_marker_page <= 0 or max_marker_page > src_total:
            return markers

        aligned: List[Dict[str, Any]] = []
        for item in markers:
            row = dict(item)
            src_page = int(self.coerce_page_number(row.get("page")) or 0)
            if src_page > 0:
                if max_marker_page <= min_marker_page:
                    mapped = min_marker_page
                else:
                    ratio = float(src_page - min_marker_page) / float(max_marker_page - min_marker_page)
                    mapped = int(round(ratio * float(dst_total - min_marker_page))) + min_marker_page
                row["page"] = max(1, min(dst_total, mapped))
            aligned.append(row)
        return aligned

    @staticmethod
    def _normalize_title(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    def _remap_marker_pages_from_manual_toc(self, markers: List[Dict[str, Any]], page_texts: List[str]) -> List[Dict[str, Any]]:
        toc_markers = self._extract_markers_from_manual_toc([self.cleaned_text])
        if not toc_markers:
            toc_markers = self._extract_markers_from_manual_toc(page_texts)
        if not toc_markers:
            return markers

        page_by_title: Dict[str, int] = {}
        for item in toc_markers:
            key = self._normalize_title(str(item.get("title") or ""))
            page = self.coerce_page_number(item.get("page"))
            if not key or page is None:
                continue
            old = page_by_title.get(key)
            if old is None or int(page) < int(old):
                page_by_title[key] = int(page)

        patched: List[Dict[str, Any]] = []
        for idx, marker in enumerate(markers):
            updated = dict(marker)
            key = self._normalize_title(str(marker.get("title") or ""))
            mapped = page_by_title.get(key)
            if mapped is not None:
                updated["page"] = int(mapped)
            updated.setdefault("order", idx)
            patched.append(updated)
        return patched

    @staticmethod
    def _has_informative_page_span(ranges: List[Dict[str, Any]], total_pages: int) -> bool:
        if not ranges:
            return False
        unique_starts = {int(item.get("start") or 0) for item in ranges}
        if len(unique_starts) > 1:
            return True
        max_start = max(unique_starts) if unique_starts else 0
        return max_start > 1 and int(total_pages) > 1

    def _expand_pages_by_catalog_if_needed(self, page_texts: List[str]) -> List[str]:
        return page_texts

    def _max_catalog_page_hint(self) -> int:
        max_page = 1
        for key in ("native_catalog", "style_catalog", "font_catalog"):
            rows = self.metadata.get(key)
            if not isinstance(rows, list):
                continue
            for item in rows:
                if not isinstance(item, dict):
                    continue
                page = self.coerce_page_number(item.get("page"))
                if page is not None:
                    max_page = max(max_page, int(page))
        if max_page <= 1:
            manual = self._extract_markers_from_manual_toc([self.cleaned_text])
            for item in manual:
                page = self.coerce_page_number(item.get("page"))
                if page is not None:
                    max_page = max(max_page, int(page))
        return max_page

    @staticmethod
    def _split_single_text_into_pages(text: str, target_pages: int) -> List[str]:
        target = max(1, int(target_pages))
        raw_lines = str(text or "").splitlines()
        if not raw_lines:
            return [str(text or "").strip()] + [""] * (target - 1)

        pages: List[str] = []
        total = len(raw_lines)
        for idx in range(target):
            start = int(round(idx * total / target))
            end = int(round((idx + 1) * total / target))
            part = "\n".join(raw_lines[start:end]).strip()
            pages.append(part)
        if pages and not pages[0].strip():
            pages[0] = str(text or "").strip()
        return pages

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
                # The final TOC marker should extend to the end of document pages.
                end = max(int(start), int(total_pages or start))
            else:
                end = max(int(start), int(next_start) - 1)

            ranges.append(
                {
                    "title": title,
                    "start": int(start),
                    "end": int(end),
                    "level": max(1, min(int(level), 6)),
                    "order": int(marker.get("order", idx)),
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
                title = str(row["title"])
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
            module_name = "_main_branch_rag_document_pdf_runtime_for_docx"
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

