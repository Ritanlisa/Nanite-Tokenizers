from __future__ import annotations

import os
import re
import hashlib
import subprocess
import sys
import types
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from rag.document_interface import ImageAsset, MonoPage, PageAssets, RAG_DB_Document
from rag.preprocessor import clean_document

try:
    import pymupdf4llm  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pymupdf4llm = None

_MAIN_PDF_CLASS: Optional[type] = None


class PDFRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        super().__init__(source_document, stable_doc_id)
        self.source_extension = self.source_extension or ".pdf"
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n\f\n"))

    def _extract_with_pymupdf4llm(self) -> Optional[List[Dict[str, Any]]]:
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
        pages: List[Dict[str, Any]] = []
        for item in chunks:
            if isinstance(item, dict):
                pages.append(item)
        return pages or None

    def _asset_output_dir(self) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        digest = hashlib.md5(str(self.base_doc_id).encode("utf-8")).hexdigest()
        out_dir = os.path.join(repo_root, "tmp", "doc_tree_assets", digest, "pdf")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _normalize_toc_title(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"^((?:\d+(?:\.\d+)*)|第[一二三四五六七八九十百千万0-9]+[章节部分篇]|附录[一二三四五六七八九十百千万A-Za-z0-9]+)(?=[A-Za-z\u4e00-\u9fff])",
            r"\1 ",
            text,
        )
        return text.strip()

    def _extract_toc_item_markers(self, tool_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        markers: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for item in tool_pages:
            for row in list(item.get("toc_items") or []):
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                try:
                    level = max(1, min(int(row[0]), 6))
                    title = self._normalize_toc_title(str(row[1] or ""))
                    page = int(row[2])
                except Exception:
                    continue
                if not title or page <= 0:
                    continue
                key = (title, page, level)
                if key in seen:
                    continue
                seen.add(key)
                markers.append({"title": title, "page": page, "level": level})
        return markers

    @staticmethod
    def _group_words_into_lines(words: List[Any]) -> List[Dict[str, Any]]:
        buckets: Dict[tuple[int, int], Dict[str, Any]] = {}
        for item in list(words or []):
            if not isinstance(item, tuple) or len(item) < 8:
                continue
            x0, y0, x1, y1, text, block_no, line_no, _ = item[:8]
            key = (int(block_no), int(line_no))
            bucket = buckets.setdefault(
                key,
                {
                    "x0": float(x0 or 0.0),
                    "y0": float(y0 or 0.0),
                    "x1": float(x1 or 0.0),
                    "y1": float(y1 or 0.0),
                    "parts": [],
                },
            )
            bucket["x0"] = min(float(bucket["x0"]), float(x0 or 0.0))
            bucket["y0"] = min(float(bucket["y0"]), float(y0 or 0.0))
            bucket["x1"] = max(float(bucket["x1"]), float(x1 or 0.0))
            bucket["y1"] = max(float(bucket["y1"]), float(y1 or 0.0))
            word = str(text or "").strip()
            if word:
                bucket["parts"].append(word)

        lines: List[Dict[str, Any]] = []
        for value in buckets.values():
            text = " ".join(str(part or "").strip() for part in value.get("parts") or [] if str(part or "").strip()).strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "x0": float(value.get("x0") or 0.0),
                    "y0": float(value.get("y0") or 0.0),
                    "x1": float(value.get("x1") or 0.0),
                    "y1": float(value.get("y1") or 0.0),
                }
            )
        lines.sort(key=lambda item: (float(item.get("y0") or 0.0), float(item.get("x0") or 0.0)))
        return lines

    def _extract_page_layout_from_tool_page(self, item: Dict[str, Any]) -> Dict[str, Any]:
        lines = self._group_words_into_lines(list(item.get("words") or []))
        if not lines:
            return {"headers": [], "footers": [], "page_number": ""}

        max_y = max(float(line.get("y1") or 0.0) for line in lines) or 1.0
        header_cut = max_y * 0.10
        footer_cut = max_y * 0.90

        headers: List[str] = []
        footers: List[str] = []
        for line in lines:
            text = str(line.get("text") or "").strip()
            if not text:
                continue
            y0 = float(line.get("y0") or 0.0)
            y1 = float(line.get("y1") or 0.0)
            if y1 <= header_cut:
                headers.append(text)
            elif y0 >= footer_cut:
                footers.append(text)

        page_number = ""
        for candidate in reversed(footers):
            if self._parse_page_number_hint(candidate) is not None:
                page_number = str(candidate).strip()
                break

        return {
            "headers": list(dict.fromkeys(headers)),
            "footers": list(dict.fromkeys(footers)),
            "page_number": page_number,
        }

    def _normalize_page_layouts(self, page_layouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header_freq: Dict[str, int] = {}
        footer_freq: Dict[str, int] = {}
        for layout in list(page_layouts or []):
            for text in list(layout.get("headers") or []):
                header_freq[text] = header_freq.get(text, 0) + 1
            for text in list(layout.get("footers") or []):
                footer_freq[text] = footer_freq.get(text, 0) + 1

        normalized: List[Dict[str, Any]] = []
        for layout in list(page_layouts or []):
            page_number = str(layout.get("page_number") or "").strip()
            headers = [
                text
                for text in list(layout.get("headers") or [])
                if header_freq.get(text, 0) >= 2
            ]
            footers: List[str] = []
            for text in list(layout.get("footers") or []):
                if text == page_number or self._parse_page_number_hint(text) is not None:
                    footers.append(text)
                    continue
                if footer_freq.get(text, 0) >= 2:
                    footers.append(text)
            if page_number and page_number not in footers:
                footers.append(page_number)
            normalized.append(
                {
                    "headers": list(dict.fromkeys(headers)),
                    "footers": list(dict.fromkeys(footers)),
                    "page_number": page_number,
                }
            )
        return normalized

    @staticmethod
    def _line_matches_layout_asset(line: str, asset_keys: set[str]) -> bool:
        line_key = RAG_DB_Document._normalized_heading_text(str(line or ""))
        if not line_key or not asset_keys:
            return False
        if line_key in asset_keys:
            return True
        if len(line_key) > 48:
            return False
        match_count = sum(1 for key in asset_keys if key and key in line_key)
        return match_count >= 2

    def _strip_layout_lines_from_page_text(
        self,
        page_text: str,
        headers: List[str],
        footers: List[str],
    ) -> str:
        lines = [str(line or "").rstrip() for line in str(page_text or "").splitlines()]
        header_keys = {
            self._normalized_heading_text(line)
            for line in headers
            if self._normalized_heading_text(line)
        }
        footer_keys = {
            self._normalized_heading_text(line)
            for line in footers
            if self._normalized_heading_text(line)
        }
        while lines:
            if not str(lines[0] or "").strip():
                lines.pop(0)
                continue
            if self._line_matches_layout_asset(lines[0], header_keys):
                lines.pop(0)
                continue
            break
        while lines:
            if not str(lines[-1] or "").strip():
                lines.pop()
                continue
            if self._line_matches_layout_asset(lines[-1], footer_keys):
                lines.pop()
                continue
            break
        return "\n".join(lines).strip()

    def _extract_pdf_page_images(self, page_count: int) -> List[List[Any]]:
        results: List[List[Any]] = [[] for _ in range(max(0, int(page_count or 0)))]
        file_path = str(self.metadata.get("file_name") or "").strip()
        if not file_path or not os.path.isfile(file_path):
            return results
        try:
            import fitz  # type: ignore
        except Exception:
            return results

        try:
            with fitz.open(file_path) as pdf_doc:
                for page_idx in range(min(int(pdf_doc.page_count or 0), len(results))):
                    page = pdf_doc.load_page(page_idx)
                    seen_xrefs: set[int] = set()
                    for image_meta in page.get_images(full=True):
                        if not image_meta:
                            continue
                        xref = int(image_meta[0] or 0)
                        if xref <= 0 or xref in seen_xrefs:
                            continue
                        seen_xrefs.add(xref)
                        image_info = pdf_doc.extract_image(xref)
                        if not isinstance(image_info, dict):
                            continue
                        image_bytes = image_info.get("image")
                        ext = str(image_info.get("ext") or "png").strip().lower() or "png"
                        if not image_bytes:
                            continue
                        filename = f"page-{page_idx + 1:04d}-xref-{xref}.{ext}"
                        results[page_idx].append(
                            ImageAsset.from_bytes(
                                data=bytes(image_bytes),
                                filename=filename,
                                media_type=f"image/{ext}",
                                width=int(image_info.get("width") or 0),
                                height=int(image_info.get("height") or 0),
                                source=f"{os.path.basename(file_path)}#page={page_idx + 1};xref={xref}",
                                page=page_idx + 1,
                            )
                        )
        except Exception:
            return [[] for _ in range(max(0, int(page_count or 0)))]
        return results

    @staticmethod
    def _load_main_pdf_class() -> Optional[type]:
        global _MAIN_PDF_CLASS
        if _MAIN_PDF_CLASS is not None:
            return _MAIN_PDF_CLASS

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        try:
            proc = subprocess.run(
                ["git", "-C", repo_root, "show", "main:rag/document_pdf.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            source = str(proc.stdout or "")
            if not source.strip():
                return None

            module_name = "_main_branch_rag_document_pdf_runtime"
            module = types.ModuleType(module_name)
            module.__file__ = "main:rag/document_pdf.py"
            sys.modules[module_name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            cls = getattr(module, "PDFRAGDocument", None)
            if cls is None:
                return None
            _MAIN_PDF_CLASS = cls
            return cls
        except Exception:
            return None

    def _extract_main_compatible_markers(self, page_texts: List[str]) -> List[Dict[str, Any]]:
        main_cls = self._load_main_pdf_class()
        if main_cls is None:
            return []
        try:
            # Rebuild cleaned_text from per-page markdown to preserve page boundaries exactly.
            merged_text = "\n\f\n".join(str(item or "") for item in page_texts)
            if not merged_text.strip():
                return []
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
            if not merged_text.strip():
                return {}
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
                for page_idx in range(int(start), int(end) + 1):
                    old = best.get(page_idx)
                    score = (depth, -span)
                    if old is None or score > (old[0], old[1]):
                        best[page_idx] = (depth, -span, section_path)

            return {page: item[2] for page, item in best.items()}
        except Exception:
            return {}

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

            if next_start is None:
                end = max(start, int(total_pages or start))
            else:
                end = max(start, int(next_start) - 1)
            ranges.append(
                {
                    "title": title,
                    "start": int(start),
                    "end": int(end),
                    "level": max(1, min(int(level), 6)),
                }
            )
        return ranges

    def build(self) -> RAG_DB_Document:
        tool_pages = self._extract_with_pymupdf4llm()
        if tool_pages is not None:
            page_texts = [str(item.get("text") or "") for item in tool_pages]
            metadata_markers = self._extract_markers_from_metadata(["native_catalog"])
            toc_markers = self._extract_toc_item_markers(tool_pages)
            marker_source = "native_catalog" if metadata_markers else ("tool_toc" if toc_markers else "main_compatible")
            markers = list(metadata_markers) if metadata_markers else (list(toc_markers) if toc_markers else self._extract_main_compatible_markers(page_texts))
            markers = self._remap_markers_with_text_hits(markers, page_texts, len(page_texts))
            markers = self._prune_unmatched_tail_markers(markers, page_texts)
            ranges = self.derive_catalog_ranges(markers, len(page_texts))
            main_section_map = self._extract_main_page_section_map(page_texts) if not metadata_markers and not toc_markers else {}
            range_source = marker_source
            if not ranges:
                fallback_markers = self.extract_multilevel_catalog_markers(
                    page_texts,
                    metadata_keys=["native_catalog", "style_catalog", "font_catalog"],
                )
                fallback_markers = self._remap_markers_with_text_hits(fallback_markers, page_texts, len(page_texts))
                markers = list(fallback_markers)
                ranges = self.derive_catalog_ranges(fallback_markers, len(page_texts))
                range_source = "fallback_multilevel"

            self.set_build_trace(
                builder="pdf-tool",
                marker_source=marker_source,
                range_source=range_source,
                source_page_count=len(page_texts),
                target_page_count=len(page_texts),
                selected_markers=list(markers),
                tree_ranges=list(ranges),
                main_section_map=dict(main_section_map),
            )

            page_layouts = self._normalize_page_layouts([self._extract_page_layout_from_tool_page(item) for item in tool_pages])
            page_images = self._extract_pdf_page_images(len(tool_pages))

            page_nodes: List[MonoPage] = []
            chunks: List[Document] = []
            for fallback_idx, item in enumerate(tool_pages, start=1):
                page_text = str(item.get("text") or "")
                metadata_raw = item.get("metadata")
                metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
                page_idx = self.coerce_page_number(metadata.get("page")) or fallback_idx

                page_layout = page_layouts[fallback_idx - 1] if fallback_idx - 1 < len(page_layouts) else {}
                headers = [str(x).strip() for x in list(page_layout.get("headers") or []) if str(x).strip()]
                footers = [str(x).strip() for x in list(page_layout.get("footers") or []) if str(x).strip()]
                body_text = self._strip_layout_lines_from_page_text(page_text, headers, footers) or page_text
                page_number_hint = str(page_layout.get("page_number") or "").strip()

                mapped_path = str(main_section_map.get(int(page_idx)) or "").strip()
                if mapped_path:
                    mapped_parts = [part.strip() for part in mapped_path.split(" > ") if part.strip()]
                    section_title = mapped_parts[-1] if mapped_parts else mapped_path
                    section_path = mapped_path
                    section_resolver = "main_section_map"
                else:
                    section = self._pick_catalog_section(page_idx, ranges)
                    section_title = str(section.get("title")) if section else f"PDF Page {page_idx}"
                    section_path = section_title if not section else f"L{section['level']}/{section_title}"
                    section_resolver = "catalog_range" if section else "guessed"
                page_meta: Dict[str, Any] = {
                    "doc_name": self.doc_name,
                    "file_name": self.doc_name,
                    "source_extension": self.source_extension,
                    "section_id": f"pdf-page-{page_idx}",
                    "section_title": section_title,
                    "section_path": section_path,
                    "section_start_page": page_idx,
                    "section_end_page": page_idx,
                    "page": page_idx,
                    "logical_page": self._parse_page_number_hint(page_number_hint) or page_idx,
                    "resolved_section_path": section_path,
                    "section_resolver": section_resolver,
                    "header_text": "\n".join(headers),
                    "footer_text": "\n".join(footers),
                    "page_number_hint": page_number_hint,
                    "physical_page": page_idx,
                    "raw_page_text": body_text,
                }

                images = self.resolve_page_images(
                    body_text,
                    page_images[fallback_idx - 1] if fallback_idx - 1 < len(page_images) else [],
                )
                assets = PageAssets(
                    headers=headers,
                    footers=footers,
                    page_numbers=[page_number_hint] if page_number_hint else [],
                    images=images,
                )
                page_meta["image_count"] = len(images)
                page_markdown = self._render_plaintext_page_to_markdown(body_text, page_number=int(page_idx))

                node = self.create_mono_page_node(
                    page_number=int(page_idx),
                    page_text=body_text,
                    markdown_text=page_markdown,
                    assets=assets,
                    metadata=page_meta,
                )
                node.add_page_number(page_idx)
                page_nodes.append(node)

                for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_markdown), start=1):
                    chunk_meta = dict(page_meta)
                    chunk_meta["chunk_index"] = chunk_idx
                    chunks.append(
                        Document(
                            text=chunk_text,
                            metadata=chunk_meta,
                            doc_id=f"{self.doc_name}::pdf-page::{page_idx}::chunk::{chunk_idx}",
                        )
                    )

            self.set_build_trace(
                page_layout_count=len(page_layouts),
                physical_page_count=len(page_nodes),
            )
            self.set_page_nodes(self.build_catalog_tree(page_nodes, ranges))
            self.chunk_documents = chunks
            self.page_count = len(page_nodes)
            self.pagination_mode = "pdf-page-tree-tool"
            self.catalog = self.catalog_payload()
            return self

        if not self.cleaned_text.strip():
            self.set_page_nodes([])
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            self.pagination_mode = "pdf-page-tree"
            return self

        raw_pages = [part.strip() for part in self.cleaned_text.split("\f")]
        page_texts = [part for part in raw_pages]
        if not page_texts:
            page_texts = [self.cleaned_text.strip()]

        metadata_markers = self._extract_markers_from_metadata(["native_catalog"])
        marker_source = "native_catalog" if metadata_markers else "main_compatible"
        markers = list(metadata_markers) if metadata_markers else self._extract_main_compatible_markers(page_texts)
        markers = self._remap_markers_with_text_hits(markers, page_texts, len(page_texts))
        markers = self._prune_unmatched_tail_markers(markers, page_texts)
        ranges = self.derive_catalog_ranges(markers, len(page_texts))
        main_section_map = self._extract_main_page_section_map(page_texts)
        range_source = marker_source
        if not ranges:
            fallback_markers = self.extract_multilevel_catalog_markers(
                page_texts,
                metadata_keys=["native_catalog", "style_catalog", "font_catalog"],
            )
            markers = list(fallback_markers)
            ranges = self.derive_catalog_ranges(fallback_markers, len(page_texts))
            range_source = "fallback_multilevel"

        self.set_build_trace(
            builder="pdf-text",
            marker_source=marker_source,
            range_source=range_source,
            source_page_count=len(page_texts),
            target_page_count=len(page_texts),
            selected_markers=list(markers),
            tree_ranges=list(ranges),
            main_section_map=dict(main_section_map),
        )

        page_nodes: List[MonoPage] = []
        chunks: List[Document] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            mapped_path = str(main_section_map.get(int(page_idx)) or "").strip()
            if mapped_path:
                mapped_parts = [part.strip() for part in mapped_path.split(" > ") if part.strip()]
                section_title = mapped_parts[-1] if mapped_parts else mapped_path
                section_path = mapped_path
                section_resolver = "main_section_map"
            else:
                section = self._pick_catalog_section(page_idx, ranges)
                section_title = str(section.get("title")) if section else f"PDF Page {page_idx}"
                section_path = section_title if not section else f"L{section['level']}/{section_title}"
                section_resolver = "catalog_range" if section else "guessed"
            page_meta: Dict[str, Any] = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": f"pdf-page-{page_idx}",
                "section_title": section_title,
                "section_path": section_path,
                "section_start_page": page_idx,
                "section_end_page": page_idx,
                "page": page_idx,
                "resolved_section_path": section_path,
                "section_resolver": section_resolver,
                "physical_page": page_idx,
                "raw_page_text": page_text,
            }

            node = self.create_mono_page_node(
                page_number=page_idx,
                page_text=page_text,
                markdown_text=page_text,
                metadata=page_meta,
            )
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_text), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::pdf-page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

            self.set_build_trace(physical_page_count=len(page_nodes))
        self.set_page_nodes(self.build_catalog_tree(page_nodes, ranges))
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "pdf-page-tree"
        self.catalog = self.catalog_payload()
        return self
