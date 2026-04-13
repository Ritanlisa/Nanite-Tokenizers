from __future__ import annotations

from typing import Any, Dict, List

from llama_index.core import Document

from rag.document_interface import Content, RAG_DB_Document
from rag.preprocessor import clean_document


class TextRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        super().__init__(source_document, stable_doc_id)
        self.source_extension = self.source_extension or ".txt"
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n\f\n"))

    def build(self) -> RAG_DB_Document:
        if not self.cleaned_text.strip():
            self.set_page_nodes([])
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            self.pagination_mode = "text-page-tree"
            return self

        raw_pages = [part.strip() for part in self.cleaned_text.split("\f")]
        page_texts = [part for part in raw_pages]
        if not page_texts:
            page_texts = [self.cleaned_text.strip()]

        page_nodes: List[Content] = []
        chunks: List[Document] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            section_title = "Document"
            section_path = "Document"
            page_meta: Dict[str, Any] = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": f"text-page-{page_idx}",
                "section_title": section_title,
                "section_path": section_path,
                "section_start_page": page_idx,
                "section_end_page": page_idx,
                "page": page_idx,
            }
            node = Content(title="", markdown_text=page_text, metadata=page_meta)
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_text), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::text-page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

        self.set_page_nodes(page_nodes)
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "text-page-tree"
        self.catalog = self.catalog_payload()
        return self
