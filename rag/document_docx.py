from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from rag.document_interface import Content, RAG_DB_Document
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
            text = str(item.get("text") or "").strip()
            if text:
                page_texts.append(text)
        return page_texts or None

    def build(self) -> RAG_DB_Document:
        tool_pages = self._extract_with_pymupdf4llm()
        page_texts = tool_pages
        if page_texts is None:
            if not self.cleaned_text.strip():
                self.set_page_nodes([])
                self.chunk_documents = []
                self.catalog = []
                self.page_count = 0
                self.pagination_mode = "docx-page-tree"
                return self
            raw_pages = [part.strip() for part in self.cleaned_text.split("\f")]
            page_texts = [part for part in raw_pages if part]
            if not page_texts:
                page_texts = [self.cleaned_text.strip()]

        markers = self.extract_multilevel_catalog_markers(page_texts, metadata_keys=["style_catalog", "native_catalog", "font_catalog"])
        ranges = self.derive_catalog_ranges(markers, len(page_texts))

        page_nodes: List[Content] = []
        chunks: List[Document] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            section = self._pick_catalog_section(page_idx, ranges)
            section_title = str(section.get("title")) if section else f"DOCX Page {page_idx}"
            section_path = section_title if not section else f"L{section['level']}/{section_title}"

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
            }
            node = Content(title=section_title, markdown_text=page_text, metadata=page_meta)
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_text), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::docx-page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

        self.set_page_nodes(self.build_catalog_tree(page_nodes, ranges))
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "docx-page-tree-tool" if tool_pages is not None else "docx-page-tree"
        self.catalog = self.catalog_payload()
        return self
