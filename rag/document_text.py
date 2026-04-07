from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Pattern, Sequence

from llama_index.core import Document

import config
from rag.document_interface import RAG_DB_Document
from rag.preprocessor import clean_document

TEXT_CHUNK_CHARS = 1200
TEXT_CHUNK_OVERLAP = 180


def _normalize_doc_path(value: str) -> str:
    return str(value or "").replace("\\", "/").strip()


def _detect_source_extension(metadata: Dict[str, Any]) -> str:
    ext = str(metadata.get("source_extension") or "").strip().lower()
    if ext:
        return ext
    file_name = str(metadata.get("file_name") or "").strip()
    if file_name:
        return os.path.splitext(file_name)[1].lower()
    return ""


def _resolve_doc_name(base_doc_id: str, metadata: Dict[str, Any]) -> str:
    current = str(metadata.get("doc_name") or "").strip()
    file_name = str(metadata.get("file_name") or "").strip()
    if file_name:
        return _normalize_doc_path(os.path.abspath(file_name))

    if current:
        data_dir = str(config.settings.DATA_DIR or "").strip()
        if os.path.isabs(current):
            return _normalize_doc_path(os.path.abspath(current))
        if data_dir:
            return _normalize_doc_path(os.path.abspath(os.path.join(data_dir, current)))
        return _normalize_doc_path(os.path.abspath(current))

    if base_doc_id:
        return _normalize_doc_path(os.path.abspath(str(base_doc_id)))
    return _normalize_doc_path(str(base_doc_id))


def _split_text_chunks(text: str, chunk_chars: int = TEXT_CHUNK_CHARS, overlap_chars: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    total = len(normalized)
    while start < total:
        hard_end = min(total, start + chunk_chars)
        if hard_end >= total:
            piece = normalized[start:].strip()
            if piece:
                chunks.append(piece)
            break

        split_at = hard_end
        for marker in ("\n\n", "\n", "。", "！", "？", ".", "!", "?"):
            idx = normalized.rfind(marker, start, hard_end)
            if idx > start:
                split_at = idx + len(marker)
                break
        if split_at <= start:
            split_at = hard_end

        piece = normalized[start:split_at].strip()
        if piece:
            chunks.append(piece)

        next_start = max(0, split_at - overlap_chars)
        if next_start <= start:
            next_start = split_at
        start = next_start

    return chunks


class TextRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        metadata = dict(source_document.metadata or {})
        self.source_document = source_document
        self.metadata = metadata
        self.base_doc_id = source_document.doc_id or stable_doc_id
        self.source_extension = _detect_source_extension(metadata)
        self.doc_name = _resolve_doc_name(str(self.base_doc_id), metadata)
        self.title = os.path.basename(self.doc_name) or self.doc_name
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n"))

        self.chunk_documents: List[Document] = []
        self.catalog: List[Dict[str, Any]] = []
        self.page_count = 1
        self.pagination_mode = "manual"

    def build(self) -> RAG_DB_Document:
        if not self.cleaned_text:
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            return self

        pieces = _split_text_chunks(self.cleaned_text)
        if not pieces:
            pieces = [self.cleaned_text]

        chunks: List[Document] = []
        for idx, piece in enumerate(pieces, start=1):
            chunk_id = f"{self.doc_name}::chunk::{idx}"
            section_id = f"chunk-{idx}"
            metadata = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": section_id,
                "section_title": f"Chunk {idx}",
                "section_path": f"Chunk {idx}",
                "section_start_page": 1,
                "section_end_page": 1,
                "page": 1,
                "chunk_index": idx,
            }
            chunks.append(Document(text=piece, metadata=metadata, doc_id=chunk_id))

        self.chunk_documents = chunks
        self.catalog = [
            {
                "title": "Document",
                "page": 1,
                "end_page": 1,
            }
        ]
        self.page_count = 1
        return self

    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        return [1 for _ in lines], "manual"

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
        section_norm = (section or "").strip().lower()
        chunk_norm = (chunk or "").strip().lower()
        page_filtered = page_start is not None or page_end is not None
        results: List[Dict[str, Any]] = []

        for chunk_doc in self.chunk_documents:
            metadata = dict(chunk_doc.metadata or {})
            text = (chunk_doc.text or "").strip()
            if not text:
                continue

            section_path = str(metadata.get("section_path") or metadata.get("section_title") or "")
            if section_norm and section_norm not in section_path.lower():
                continue
            if chunk_norm and chunk_norm not in text.lower():
                continue
            if compiled_regex and not compiled_regex.search(text):
                continue

            node_page_start = self.coerce_page_number(metadata.get("section_start_page"))
            node_page_end = self.coerce_page_number(metadata.get("section_end_page"))
            node_page_raw = self.coerce_page_number(metadata.get("page"))
            candidate_start = node_page_start or node_page_raw or node_page_end
            candidate_end = node_page_end or node_page_start or node_page_raw
            node_page = candidate_start

            if page_filtered:
                if candidate_start is None or candidate_end is None:
                    continue
                if page_start is not None and candidate_end < page_start:
                    continue
                if page_end is not None and candidate_start > page_end:
                    continue

            results.append(
                {
                    "score": 0.0,
                    "text": text[:1400],
                    "doc_name": metadata.get("doc_name") or self.doc_name,
                    "section_path": section_path or None,
                    "page": node_page,
                    "page_start": candidate_start,
                    "page_end": candidate_end,
                    "metadata": metadata,
                }
            )
        return results

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
        section_norm = (section or "").strip().lower()
        chunk_norm = (chunk or "").strip().lower()
        page_filtered = page_start is not None or page_end is not None
        results: List[Dict[str, Any]] = []

        for chunk_doc in self.chunk_documents:
            metadata = dict(chunk_doc.metadata or {})
            text = (chunk_doc.text or "").strip()
            if not text:
                continue

            section_id = str(metadata.get("section_id") or "").strip()
            if not section_id or section_id not in section_scores:
                continue

            section_path = str(metadata.get("section_path") or metadata.get("section_title") or "")
            if section_norm and section_norm not in section_path.lower():
                continue
            if chunk_norm and chunk_norm not in text.lower():
                continue
            if compiled_regex and not compiled_regex.search(text):
                continue

            node_page_start = self.coerce_page_number(metadata.get("section_start_page"))
            node_page_end = self.coerce_page_number(metadata.get("section_end_page"))
            node_page_raw = self.coerce_page_number(metadata.get("page"))
            candidate_start = node_page_start or node_page_raw or node_page_end
            candidate_end = node_page_end or node_page_start or node_page_raw
            node_page = candidate_start

            if page_filtered:
                if candidate_start is None or candidate_end is None:
                    continue
                if page_start is not None and candidate_end < page_start:
                    continue
                if page_end is not None and candidate_start > page_end:
                    continue

            results.append(
                {
                    "score": float(section_scores.get(section_id, 0.0)),
                    "text": text[:1400],
                    "doc_name": metadata.get("doc_name") or self.doc_name,
                    "section_path": section_path or None,
                    "page": node_page,
                    "page_start": candidate_start,
                    "page_end": candidate_end,
                    "metadata": metadata,
                }
            )
        return results

    def list_payload(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "doc_name": self.doc_name,
            "page_count": self.page_count,
            "source_extension": self.source_extension,
            "chunk_count": len(self.chunk_documents),
            "pagination_mode": self.pagination_mode,
        }

    def catalog_payload(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.catalog]
