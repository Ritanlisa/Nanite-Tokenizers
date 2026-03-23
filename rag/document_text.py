from __future__ import annotations

from typing import Any, Dict, List, Optional, Pattern, Sequence, cast

from llama_index.core import Document

from rag.document_interface import RAG_DB_Document
from rag.document_pdf import PDFRAGDocument


class TextRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        PDFRAGDocument.__init__(cast(PDFRAGDocument, self), source_document, stable_doc_id)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(PDFRAGDocument, name)
        if hasattr(attr, "__get__"):
            return attr.__get__(self, self.__class__)
        return attr

    def build(self) -> RAG_DB_Document:
        return PDFRAGDocument.build(cast(PDFRAGDocument, self))

    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        return PDFRAGDocument._build_manual_page_map(lines), "manual"

    def allow_heading_detection(self) -> bool:
        return PDFRAGDocument.allow_heading_detection(cast(PDFRAGDocument, self))

    def retrieve_by_regex(
        self,
        *,
        compiled_regex: Optional[Pattern[str]],
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        chunk: Optional[str],
    ) -> List[Dict[str, Any]]:
        return PDFRAGDocument.retrieve_by_regex(
            cast(PDFRAGDocument, self),
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )

    def retrieve_by_vector(
        self,
        *,
        query_text: str,
        section_scores: Dict[str, float],
        compiled_regex: Optional[Pattern[str]],
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        chunk: Optional[str],
    ) -> List[Dict[str, Any]]:
        return PDFRAGDocument.retrieve_by_vector(
            cast(PDFRAGDocument, self),
            query_text=query_text,
            section_scores=section_scores,
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )

    def list_payload(self) -> Dict[str, Any]:
        return PDFRAGDocument.list_payload(cast(PDFRAGDocument, self))

    def catalog_payload(self) -> List[Dict[str, Any]]:
        return PDFRAGDocument.catalog_payload(cast(PDFRAGDocument, self))
