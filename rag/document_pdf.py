from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from rag.document_interface import MonoPage, RAG_DB_Document
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
            markers = self._extract_main_compatible_markers(page_texts)
            ranges = self._ranges_from_main_markers(markers, len(page_texts))
            main_section_map = self._extract_main_page_section_map(page_texts)
            if not ranges:
                fallback_markers = self.extract_multilevel_catalog_markers(
                    page_texts,
                    metadata_keys=["native_catalog", "style_catalog", "font_catalog"],
                )
                ranges = self.derive_catalog_ranges(fallback_markers, len(page_texts))

            page_nodes: List[MonoPage] = []
            chunks: List[Document] = []
            for fallback_idx, item in enumerate(tool_pages, start=1):
                page_text = str(item.get("text") or "")
                metadata_raw = item.get("metadata")
                metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
                page_idx = self.coerce_page_number(metadata.get("page")) or fallback_idx

                mapped_path = str(main_section_map.get(int(page_idx)) or "").strip()
                if mapped_path:
                    mapped_parts = [part.strip() for part in mapped_path.split(" > ") if part.strip()]
                    section_title = mapped_parts[-1] if mapped_parts else mapped_path
                    section_path = mapped_path
                else:
                    section = self._pick_catalog_section(page_idx, ranges)
                    section_title = str(section.get("title")) if section else f"PDF Page {page_idx}"
                    section_path = section_title if not section else f"L{section['level']}/{section_title}"
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
                }

                node = self.create_mono_page_node(
                    page_number=int(page_idx),
                    page_text=page_text,
                    markdown_text=page_text,
                    metadata=page_meta,
                )
                node.add_page_number(page_idx)
                for image_item in item.get("images") or []:
                    node.add_image(str(image_item))
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

        markers = self._extract_main_compatible_markers(page_texts)
        ranges = self._ranges_from_main_markers(markers, len(page_texts))
        main_section_map = self._extract_main_page_section_map(page_texts)
        if not ranges:
            fallback_markers = self.extract_multilevel_catalog_markers(
                page_texts,
                metadata_keys=["native_catalog", "style_catalog", "font_catalog"],
            )
            ranges = self.derive_catalog_ranges(fallback_markers, len(page_texts))

        page_nodes: List[MonoPage] = []
        chunks: List[Document] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            mapped_path = str(main_section_map.get(int(page_idx)) or "").strip()
            if mapped_path:
                mapped_parts = [part.strip() for part in mapped_path.split(" > ") if part.strip()]
                section_title = mapped_parts[-1] if mapped_parts else mapped_path
                section_path = mapped_path
            else:
                section = self._pick_catalog_section(page_idx, ranges)
                section_title = str(section.get("title")) if section else f"PDF Page {page_idx}"
                section_path = section_title if not section else f"L{section['level']}/{section_title}"
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

        self.set_page_nodes(self.build_catalog_tree(page_nodes, ranges))
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "pdf-page-tree"
        self.catalog = self.catalog_payload()
        return self
