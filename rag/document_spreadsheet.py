from __future__ import annotations

from typing import Any, Dict, List, Optional, Pattern, Sequence, cast

from llama_index.core import Document

from rag.document_interface import RAG_DB_Document
from rag.document_text import TextRAGDocument


class SpreadsheetRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        TextRAGDocument.__init__(cast(TextRAGDocument, self), source_document, stable_doc_id)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(TextRAGDocument, name)
        if hasattr(attr, "__get__"):
            return attr.__get__(self, self.__class__)
        return attr

    def build(self) -> RAG_DB_Document:
        return TextRAGDocument.build(cast(TextRAGDocument, self))

    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        return TextRAGDocument.resolve_page_map(
            cast(TextRAGDocument, self),
            lines,
            native_page_map,
            has_native_marker,
        )

    def allow_heading_detection(self) -> bool:
        return False

    def retrieve_by_regex(
        self,
        *,
        compiled_regex: Optional[Pattern[str]],
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        chunk: Optional[str],
    ) -> List[Dict[str, Any]]:
        return TextRAGDocument.retrieve_by_regex(
            cast(TextRAGDocument, self),
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
        return TextRAGDocument.retrieve_by_vector(
            cast(TextRAGDocument, self),
            query_text=query_text,
            section_scores=section_scores,
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )

    def list_payload(self) -> Dict[str, Any]:
        return TextRAGDocument.list_payload(cast(TextRAGDocument, self))

    def catalog_payload(self) -> List[Dict[str, Any]]:
        return TextRAGDocument.catalog_payload(cast(TextRAGDocument, self))
