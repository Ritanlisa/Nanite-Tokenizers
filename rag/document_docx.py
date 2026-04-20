from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from rag.document_interface import MonoPage, PageAssets, RAG_DB_Document, _dedupe_text_values
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
        native_page_count = int(self.coerce_page_number(self.metadata.get("native_page_count")) or 0)
        if bool(self.metadata.get("native_pagination")) and native_page_count > source_page_count:
            page_texts = list(source_page_texts) + [""] * (native_page_count - source_page_count)
            source_page_map = list(range(1, source_page_count + 1)) + list(range(source_page_count + 1, native_page_count + 1))
        else:
            page_texts = self.enforce_native_page_count(page_texts)
            source_page_map = self._map_target_to_source_pages(source_page_count, len(page_texts))

        source_ranges = self._extract_structured_catalog_ranges(source_page_texts, source_page_count, source_page_count)
        main_markers = self._extract_main_compatible_markers(page_texts)
        main_ranges = self._ranges_from_main_markers(main_markers, len(page_texts))
        main_section_map = self._extract_main_page_section_map(page_texts)
        page_signals = self.build_page_signals(page_texts, source_ranges)

        physical_page_nodes: List[MonoPage] = []
        page_layouts = self.get_page_layouts()
        deferred_catalog_citations: List[str] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            signal = page_signals[page_idx - 1] if page_idx - 1 < len(page_signals) else {}
            logical_page = int(signal.get("logical_page_number") or page_idx)
            source_page = source_page_map[page_idx - 1] if page_idx - 1 < len(source_page_map) else page_idx
            source_section = self._pick_catalog_section(source_page, source_ranges)
            mapped_path = str(main_section_map.get(int(page_idx)) or "").strip()
            if re.match(r"^(?:document|chunk\s+\d+)$", mapped_path, flags=re.IGNORECASE):
                mapped_path = ""
            if mapped_path:
                mapped_parts = [part.strip() for part in mapped_path.split(" > ") if part.strip()]
                section_title = mapped_parts[-1] if mapped_parts else mapped_path
                section_path = mapped_path
                section_resolver = "main_section_map"
            elif source_section:
                section_title = str(source_section.get("title"))
                section_path = f"L{source_section['level']}/{section_title}"
                section_resolver = "catalog_range"
            else:
                section_title = self._guess_page_title(page_text, page_idx)
                section_path = section_title
                section_resolver = "guessed"

            page_layout = page_layouts[page_idx - 1] if page_idx - 1 < len(page_layouts) and isinstance(page_layouts[page_idx - 1], dict) else {}
            headers = [str(x).strip() for x in list(page_layout.get("headers") or []) if str(x).strip()]
            if not headers and str(signal.get("header_text") or "").strip():
                headers = [str(signal.get("header_text") or "").strip()]
            footers = [str(x).strip() for x in list(page_layout.get("footers") or []) if str(x).strip()]
            if not footers and str(signal.get("footer_text") or "").strip():
                footers = [str(signal.get("footer_text") or "").strip()]
            citations = [str(x).strip() for x in list(page_layout.get("citations") or []) if str(x).strip()]
            is_catalogue_page = self._looks_like_catalogue_page(page_text)
            if is_catalogue_page and citations:
                deferred_catalog_citations.extend(citations)
                citations = []
            elif (not is_catalogue_page) and deferred_catalog_citations:
                # Reattach notes that were incorrectly bound to TOC pages.
                citations = _dedupe_text_values(list(deferred_catalog_citations) + list(citations))
                deferred_catalog_citations = []
            images = self.resolve_page_images(
                page_text,
                list(page_layout.get("images") or []),
            )
            page_markdown = self._render_plaintext_page_to_markdown(
                page_text,
                page_number=page_idx,
                page_images=images,
                page_image_indexes=list(range(1, len(images) + 1)),
                page_citations=citations,
            )
            page_number_hint = str(page_layout.get("page_number") or "").strip()
            if not page_number_hint:
                page_number_hint = str(signal.get("page_number_hint") or "").strip()
            assets = PageAssets(
                headers=headers,
                footers=footers,
                citations=citations,
                page_numbers=[page_number_hint] if page_number_hint else [],
                images=images,
            )

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
                "source_section_title": str(source_section.get("title") or "") if source_section else "",
                "source_section_level": int(source_section.get("level") or 0) if source_section else 0,
                "resolved_section_path": section_path,
                "section_resolver": section_resolver,
                "header_text": "\n".join(headers),
                "footer_text": "\n".join(footers),
                "page_number_hint": page_number_hint,
                "image_count": len(images),
                "physical_page": page_idx,
                "page_image_indexes": list(range(1, len(images) + 1)),
                "raw_page_text": page_text,
            }
            node = self.create_mono_page_node(
                page_number=page_idx,
                page_text=page_text,
                markdown_text=page_markdown,
                assets=assets,
                metadata=page_meta,
            )
            node.add_page_number(page_idx)
            physical_page_nodes.append(node)

        self._apply_structured_section_fallback(physical_page_nodes, source_ranges)
        physical_ranges = self._materialize_physical_ranges_from_pages(physical_page_nodes, source_ranges)
        tree_ranges = list(physical_ranges)
        tree_range_source = "physical_ranges"
        if not tree_ranges and main_section_map:
            tree_ranges = self._ranges_from_page_section_map(main_section_map, len(page_texts))
            tree_range_source = "main_section_map"
        if not tree_ranges and main_ranges:
            tree_ranges = list(main_ranges)
            tree_range_source = "main_ranges"

        self.set_build_trace(
            builder="docx",
            source_page_count=source_page_count,
            target_page_count=len(page_texts),
            source_page_map=list(source_page_map),
            source_ranges=list(source_ranges),
            main_markers=list(main_markers),
            main_ranges=list(main_ranges),
            main_section_map=dict(main_section_map),
            physical_ranges=list(physical_ranges),
            tree_ranges=list(tree_ranges),
            tree_range_source=tree_range_source,
        )

        leaf_page_nodes = self.split_mono_pages_by_section_markers(physical_page_nodes, tree_ranges)
        chunks: List[Document] = []
        for leaf_index, leaf_page in enumerate(leaf_page_nodes, start=1):
            leaf_meta = dict(getattr(leaf_page, "metadata", {}) or {})
            leaf_page_no = int(self.coerce_page_number(leaf_meta.get("page")) or leaf_index)
            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(str(getattr(leaf_page, "markdown_text", "") or "")), start=1):
                if not str(chunk_text or "").strip():
                    continue
                chunk_meta = dict(leaf_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::docx-page::{leaf_page_no}::chunk::{chunk_idx}",
                    )
                )

            self.set_build_trace(
                leaf_page_count=len(leaf_page_nodes),
                physical_page_count=len(physical_page_nodes),
            )
        self.set_page_nodes(self.build_catalog_tree(leaf_page_nodes, tree_ranges))
        self.chunk_documents = chunks
        self.page_count = len(physical_page_nodes)
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
        pages: List[MonoPage],
        source_ranges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        source_to_pages: Dict[int, List[int]] = {}
        for page in pages:
            meta = dict(getattr(page, "metadata", {}) or {})
            page_no = int(self.coerce_page_number(meta.get("page")) or 0)
            source_no = int(self.coerce_page_number(meta.get("source_page")) or page_no)
            if page_no <= 0:
                continue
            source_to_pages.setdefault(source_no, []).append(page_no)

        projected: List[Dict[str, Any]] = []
        for item in source_ranges:
            start_src = int(self.coerce_page_number(item.get("start")) or 0)
            end_src = int(self.coerce_page_number(item.get("end")) or start_src)
            if start_src <= 0 or end_src <= 0:
                continue
            pages_in: List[int] = []
            for src in range(min(start_src, end_src), max(start_src, end_src) + 1):
                pages_in.extend(source_to_pages.get(src, []))
            pages_in = sorted(set(pages_in))
            if not pages_in:
                continue
            projected.append(
                {
                    "title": str(item.get("title") or ""),
                    "start": int(min(pages_in)),
                    "end": int(max(pages_in)),
                    "level": int(item.get("level") or 1),
                    "order": int(item.get("order", 0)),
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
        should_rescale = int(total_pages or 0) != int(source_page_count or 0)
        manual_markers = self._extract_markers_from_manual_toc(page_texts)
        if not manual_markers and self.cleaned_text.strip():
            manual_markers = self._extract_markers_from_manual_toc([self.cleaned_text])

        style_markers = self._extract_markers_from_metadata(["style_catalog"])
        font_markers = self._extract_markers_from_metadata(["font_catalog"])
        style_levels = {
            int(self.coerce_page_number(item.get("level")) or 1)
            for item in list(style_markers or [])
            if int(self.coerce_page_number(item.get("level")) or 1) > 1
        }
        prefer_style_hierarchy = bool(style_levels)

        candidate_sets: List[tuple[str, List[Dict[str, Any]]]] = [
            ("native_catalog", self._extract_markers_from_metadata(["native_catalog"])),
        ]
        if prefer_style_hierarchy:
            candidate_sets.extend(
                [
                    ("style_catalog", style_markers),
                    ("font_catalog", font_markers),
                    ("manual_toc", manual_markers),
                ]
            )
        else:
            candidate_sets.extend(
                [
                    ("manual_toc", manual_markers),
                    ("style_catalog", style_markers),
                    ("font_catalog", font_markers),
                ]
            )

        for source_name, raw_markers in candidate_sets:
            markers = [dict(item) for item in list(raw_markers or [])]
            if not markers:
                continue
            for idx, item in enumerate(markers):
                item.setdefault("order", idx)
            if should_rescale and source_name in {"style_catalog", "font_catalog"}:
                markers = self._rescale_marker_pages_if_needed(markers, source_page_count, total_pages)
            # DOC/DOCX 的目录页码通常来自目录区显示页码或样式页码，始终需要映射回正文物理页。
            markers = self._remap_markers_with_top_level_anchors(markers, page_texts, total_pages)
            markers = self._remap_markers_with_text_hits(markers, page_texts, total_pages)
            markers = self._prune_unmatched_tail_markers(markers, page_texts)
            ranges = self._ranges_from_raw_markers(markers, total_pages)
            if ranges and self._has_informative_page_span(ranges, total_pages):
                return ranges
        return []

    def _remap_markers_with_top_level_anchors(
        self,
        markers: List[Dict[str, Any]],
        page_texts: List[str],
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        if not markers or not page_texts:
            return markers

        first_lines: List[str] = []
        for text in page_texts:
            first = ""
            for raw in str(text or "").splitlines():
                raw = raw.strip()
                if raw:
                    first = raw
                    break
            first_lines.append(self._normalize_title(first))

        anchors: List[tuple[int, int]] = []
        for marker in markers:
            level = int(self.coerce_page_number(marker.get("level")) or 1)
            if level != 1:
                continue
            title = str(marker.get("title") or "").strip()
            if not title or self._is_noise_heading_line(title):
                continue
            old_page = int(self.coerce_page_number(marker.get("page")) or 0)
            if old_page <= 0:
                continue
            key = self._normalize_title(title)
            if len(key) < 4:
                continue
            hit = None
            for idx, line in enumerate(first_lines, start=1):
                if line.startswith(key):
                    hit = idx
                    break
            if hit is not None:
                anchors.append((old_page, int(hit)))

        if len(anchors) < 2:
            return markers
        anchors = sorted(set(anchors), key=lambda item: item[0])

        def map_page(old_page: int) -> int:
            if old_page <= anchors[0][0]:
                x1, y1 = anchors[0]
                x2, y2 = anchors[1]
            elif old_page >= anchors[-1][0]:
                x1, y1 = anchors[-2]
                x2, y2 = anchors[-1]
            else:
                x1 = y1 = x2 = y2 = 0
                for idx in range(len(anchors) - 1):
                    a1 = anchors[idx]
                    a2 = anchors[idx + 1]
                    if a1[0] <= old_page <= a2[0]:
                        x1, y1 = a1
                        x2, y2 = a2
                        break
            if x2 == x1:
                mapped = y1
            else:
                ratio = float(old_page - x1) / float(x2 - x1)
                mapped = int(round(y1 + ratio * float(y2 - y1)))
            return max(1, min(int(total_pages), mapped))

        aligned: List[Dict[str, Any]] = []
        for item in markers:
            row = dict(item)
            old_page = int(self.coerce_page_number(row.get("page")) or 0)
            if old_page > 0:
                mapped = map_page(old_page)
                row["page"] = mapped
            aligned.append(row)

        non_toc_top_pages: List[int] = []
        for row in aligned:
            level = int(self.coerce_page_number(row.get("level")) or 1)
            if level != 1:
                continue
            title_key = self._normalize_title(str(row.get("title") or ""))
            if "目录" in title_key or "toc" in title_key or "mulu" in title_key:
                continue
            page = int(self.coerce_page_number(row.get("page")) or 0)
            if page > 0:
                non_toc_top_pages.append(page)

        upper_bound = min(non_toc_top_pages) - 1 if non_toc_top_pages else int(total_pages)
        toc_start = self._detect_toc_start_page(page_texts, upper_bound)
        if toc_start is not None:
            for row in aligned:
                title_key = self._normalize_title(str(row.get("title") or ""))
                if "目录" in title_key or "toc" in title_key or "mulu" in title_key:
                    row["page"] = int(toc_start)
        return aligned

    def _detect_toc_start_page(self, page_texts: List[str], upper_bound: int) -> int | None:
        if not page_texts:
            return None
        limit = max(1, min(len(page_texts), int(upper_bound or len(page_texts))))
        toc_row = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+章|附录|\d+(?:\.\d+)+).{0,100}?\d{1,4}\s*$")
        for idx in range(1, limit + 1):
            lines = [line.strip() for line in str(page_texts[idx - 1] or "").splitlines() if line.strip()]
            if not lines:
                continue
            first_key = self._normalize_title(lines[0])
            if "目录" in first_key or "toc" in first_key or "mulu" in first_key:
                return idx
            score = 0
            for line in lines[:32]:
                compact = re.sub(r"[·•.\s]+", " ", line).strip()
                if toc_row.match(compact):
                    score += 1
            if score >= 4:
                return idx
        return None

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
        if page_number == 1:
            return "封面"
        lines = [line.strip() for line in str(page_text or "").splitlines() if line and line.strip()]
        if not lines:
            return ""
        toc_hits = sum(
            1
            for line in lines[:48]
            if re.match(r"^(.{1,200}?)(?:\s*\|\s*|\t+|[·•.]{2,}|\s{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*\|?\s*$", line, flags=re.IGNORECASE)
        )
        first_compact = re.sub(r"\s+", "", lines[0]).lower()
        if first_compact in {"目录", "目錄", "contents", "tableofcontents", "toc"} or toc_hits >= 2:
            return "目录"
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
                ["git", "-C", repo_root, "show", "main:rag/document_docx.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            source = str(proc.stdout or "")
            if not source.strip():
                return None
            module_name = "_main_branch_rag_document_pdf_runtime_for_docx"
            module = types.ModuleType(module_name)
            module.__file__ = "main:rag/document_docx.py"
            sys.modules[module_name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            return getattr(module, "DocxRAGDocument", None)
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

