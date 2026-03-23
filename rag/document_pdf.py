from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    fitz = None

from llama_index.core import Document

import config
from rag.document_interface import RAG_DB_Document
from rag.preprocessor import clean_document

MANUAL_PAGE_PARAGRAPH_UNITS = 20
MANUAL_WRAP_CHARS = 90
MAX_CHUNK_METADATA_VALUE_LENGTH = 320

logger = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    title: str
    page: int
    end_page: int


@dataclass
class _CatalogLineHit:
    idx: int
    title: str
    page: int
    level: int


class PDFRAGDocument(RAG_DB_Document):
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        metadata = dict(source_document.metadata or {})
        self.source_document = source_document
        self.metadata = metadata
        self.base_doc_id = source_document.doc_id or stable_doc_id
        self.source_extension = self._detect_source_extension(metadata)
        self.doc_name = self._resolve_doc_name(metadata)
        self.title = os.path.basename(self.doc_name) or self.doc_name
        self.cleaned_text = clean_document((source_document.text or "").replace("\f", "\n\f\n"))

        self.chunk_documents: List[Document] = []
        self.catalog: List[CatalogEntry] = []
        self.page_count = 0
        self.pagination_mode = "manual"
        self._source_pdf_title_page_map_cache: Optional[Dict[str, int]] = None
        self._source_pdf_page_norms_cache: Optional[List[str]] = None
        self._office_pdf_title_page_map_cache: Optional[Dict[str, int]] = None
        self._office_pdf_page_norms_cache: Optional[List[str]] = None
        self._office_pdf_page_texts_cache: Optional[List[str]] = None
        self._office_pdf_toc_end_page_cache: Optional[int] = None
        self._office_pdf_page_number_offset_cache: Optional[int] = None
        self._native_split_page_norms_cache: Optional[Dict[int, str]] = None

    def build(self) -> RAG_DB_Document:
        if not self.cleaned_text:
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            return self

        lines = self.cleaned_text.split("\n")
        native_page_map, has_native_marker, has_explicit_page_number = self._build_native_page_map(lines)
        page_by_line, self.pagination_mode = self.resolve_page_map(lines, native_page_map, has_native_marker)

        markers = self._build_markers(lines, page_by_line, has_explicit_page_number)
        if markers:
            self.chunk_documents = self._build_section_chunks(lines, markers)
            self.catalog = self._build_catalog_from_markers(markers)
        else:
            self.chunk_documents = self._build_fixed_chunks(self.cleaned_text)
            self.catalog = self._build_catalog_from_chunks(self.chunk_documents)

        if not self.chunk_documents and self.cleaned_text:
            self.chunk_documents = self._build_fixed_chunks(self.cleaned_text)
            if not self.chunk_documents:
                self.chunk_documents = [self._build_single_chunk(self.cleaned_text)]
            self.catalog = self._build_catalog_from_chunks(self.chunk_documents)

        max_page_from_map = max(page_by_line) if page_by_line else 0
        max_page_from_catalog = max((entry.end_page for entry in self.catalog), default=0)
        self.page_count = max(max_page_from_map, max_page_from_catalog)
        return self

    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        if has_native_marker:
            return list(native_page_map), "native"
        return self._build_manual_page_map(lines), "manual"

    def allow_heading_detection(self) -> bool:
        return True

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
        regex_used = compiled_regex is not None
        results: List[Dict[str, Any]] = []

        for chunk_doc in self.chunk_documents:
            metadata = dict(chunk_doc.metadata or {})
            text = (chunk_doc.text or "").strip()
            if not text:
                continue

            section_path = str(metadata.get("section_path") or metadata.get("section_title") or "")
            if section_norm and section_norm not in section_path.lower():
                continue

            node_page_start = self.coerce_page_number(metadata.get("section_start_page"))
            node_page_end = self.coerce_page_number(metadata.get("section_end_page"))
            node_page_raw = self.coerce_page_number(metadata.get("page"))
            node_page = node_page_start or node_page_raw or node_page_end
            candidate_start = node_page_start or node_page_raw or node_page_end
            candidate_end = node_page_end or node_page_start or node_page_raw
            if page_filtered:
                if candidate_start is None or candidate_end is None:
                    continue
                if page_start is not None and candidate_end < page_start:
                    continue
                if page_end is not None and candidate_start > page_end:
                    continue

            if chunk_norm and chunk_norm not in text.lower():
                continue
            if compiled_regex and not compiled_regex.search(text):
                continue

            score = 0.0
            if section_norm:
                score += 0.15
            if page_filtered:
                score += 0.1
            if chunk_norm:
                score += 0.2
            if regex_used:
                score += 0.2

            results.append(
                {
                    "score": score,
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
        regex_used = compiled_regex is not None
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

            node_page_start = self.coerce_page_number(metadata.get("section_start_page"))
            node_page_end = self.coerce_page_number(metadata.get("section_end_page"))
            node_page_raw = self.coerce_page_number(metadata.get("page"))
            node_page = node_page_start or node_page_raw or node_page_end
            candidate_start = node_page_start or node_page_raw or node_page_end
            candidate_end = node_page_end or node_page_start or node_page_raw
            if page_filtered:
                if candidate_start is None or candidate_end is None:
                    continue
                if page_start is not None and candidate_end < page_start:
                    continue
                if page_end is not None and candidate_start > page_end:
                    continue

            if chunk_norm and chunk_norm not in text.lower():
                continue
            if compiled_regex and not compiled_regex.search(text):
                continue

            score = float(section_scores.get(section_id, 0.0))
            if section_norm:
                score += 0.15
            if page_filtered:
                score += 0.1
            if chunk_norm:
                score += 0.2
            if regex_used:
                score += 0.2

            results.append(
                {
                    "score": score,
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
        }

    def catalog_payload(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": item.title,
                "page": item.page,
                "end_page": item.end_page,
            }
            for item in sorted(self.catalog, key=lambda item: (item.page, item.title))
        ]

    def _build_chunk_metadata_base(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key, value in self.metadata.items():
            if key in {"native_catalog", "style_catalog", "font_catalog"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
                continue
            text = str(value)
            if text and len(text) <= MAX_CHUNK_METADATA_VALUE_LENGTH:
                metadata[key] = text
        return metadata

    @staticmethod
    def _chunk_char_sizes() -> tuple[int, int]:
        chunk_size = max(100, int(getattr(config.settings, "CHUNK_SIZE", 512) or 512))
        chunk_overlap = max(0, int(getattr(config.settings, "CHUNK_OVERLAP", 50) or 50))
        target_chars = max(700, chunk_size * 3)
        overlap_chars = min(target_chars // 3, max(80, chunk_overlap * 3))
        return target_chars, overlap_chars

    def _split_text_for_chunks(self, text: str) -> List[str]:
        normalized = (text or "").strip()
        if not normalized:
            return []

        target_chars, overlap_chars = self._chunk_char_sizes()
        if len(normalized) <= target_chars:
            return [normalized]

        chunks: List[str] = []
        start = 0
        total = len(normalized)
        min_mid = max(120, target_chars // 2)

        while start < total:
            hard_end = min(total, start + target_chars)
            if hard_end >= total:
                piece = normalized[start:total].strip()
                if piece:
                    chunks.append(piece)
                break

            split_at = normalized.rfind("\n", start + min_mid, hard_end)
            if split_at == -1:
                split_at = normalized.rfind(" ", start + min_mid, hard_end)
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

    @staticmethod
    def _trim_metadata_value(value: Any) -> Any:
        if isinstance(value, str) and len(value) > MAX_CHUNK_METADATA_VALUE_LENGTH:
            return value[:MAX_CHUNK_METADATA_VALUE_LENGTH]
        return value

    @staticmethod
    def _normalize_doc_path(value: str) -> str:
        return value.replace("\\", "/").strip()

    @staticmethod
    def coerce_page_number(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else 0
        if isinstance(value, float):
            return int(value) if value >= 0 else 0
        text = str(value).strip()
        if not text:
            return None
        try:
            number = int(float(text))
            return number if number >= 0 else 0
        except Exception:
            match = re.search(r"\d+", text)
            if not match:
                return None
            number = int(match.group(0))
            return number if number >= 0 else 0

    @staticmethod
    def _detect_source_extension(metadata: Dict[str, Any]) -> str:
        ext = str(metadata.get("source_extension") or "").strip().lower()
        if ext:
            return ext
        file_name = str(metadata.get("file_name") or "").strip()
        if file_name:
            return os.path.splitext(file_name)[1].lower()
        return ""

    def _resolve_doc_name(self, metadata: Dict[str, Any]) -> str:
        current = str(metadata.get("doc_name") or "").strip()
        file_name = str(metadata.get("file_name") or "").strip()
        if file_name:
            return self._normalize_doc_path(os.path.abspath(file_name))

        if current:
            data_dir = str(config.settings.DATA_DIR or "").strip()
            if os.path.isabs(current):
                return self._normalize_doc_path(os.path.abspath(current))
            if data_dir:
                return self._normalize_doc_path(os.path.abspath(os.path.join(data_dir, current)))
            return self._normalize_doc_path(os.path.abspath(current))

        if self.base_doc_id:
            return self._normalize_doc_path(os.path.abspath(str(self.base_doc_id)))
        return self._normalize_doc_path(str(self.base_doc_id))

    @staticmethod
    def _is_form_feed_line(raw_line: str) -> bool:
        text = raw_line or ""
        if "\f" not in text:
            return False
        return text.replace("\f", "").strip() == ""

    @staticmethod
    def _build_native_page_map(lines: Sequence[str]) -> tuple[List[int], bool, bool]:
        page_by_line: List[int] = []
        current_page = 1
        has_native_marker = False
        has_explicit_page_number = False
        for raw_line in lines:
            if PDFRAGDocument._is_form_feed_line(raw_line or ""):
                has_native_marker = True
                current_page += 1
                page_by_line.append(current_page)
                continue
            line = (raw_line or "").strip()

            match = re.match(r"^Page\s+(\d+)\b", line, re.IGNORECASE)
            if not match:
                match = re.match(r"^第\s*(\d+)\s*页\b", line)
            if not match:
                match = re.match(r"^[-—\s]*\[?\s*(\d{1,4})\s*/\s*\d{1,4}\s*\]?[-—\s]*$", line)

            if match:
                has_native_marker = True
                has_explicit_page_number = True
                marker_page = int(match.group(1))
                if marker_page > 0:
                    current_page = marker_page
            page_by_line.append(current_page)
        return page_by_line, has_native_marker, has_explicit_page_number

    @staticmethod
    def _build_manual_page_map(
        lines: Sequence[str],
        *,
        paragraph_units: int = MANUAL_PAGE_PARAGRAPH_UNITS,
        wrap_chars: int = MANUAL_WRAP_CHARS,
    ) -> List[int]:
        if not lines:
            return []
        units_per_page = max(1, int(paragraph_units))
        line_chars = max(20, int(wrap_chars))
        page_by_line = [1] * len(lines)

        paragraphs: List[tuple[int, int, int]] = []
        start_idx: Optional[int] = None
        char_count = 0

        for idx, raw_line in enumerate(lines):
            line = (raw_line or "").strip()
            if line:
                if start_idx is None:
                    start_idx = idx
                char_count += len(line)
            elif start_idx is not None:
                paragraphs.append((start_idx, idx - 1, char_count))
                start_idx = None
                char_count = 0
        if start_idx is not None:
            paragraphs.append((start_idx, len(lines) - 1, char_count))

        used_units = 0
        for start, end, chars in paragraphs:
            estimated_units = max(1, (max(chars, 1) + line_chars - 1) // line_chars)
            current_page = (used_units // units_per_page) + 1
            for index in range(start, end + 1):
                page_by_line[index] = current_page
            used_units += estimated_units

        for idx in range(1, len(page_by_line)):
            if not (lines[idx] or "").strip():
                page_by_line[idx] = page_by_line[idx - 1]

        return page_by_line

    @staticmethod
    def _is_noise_heading_line(line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return True
        if re.match(r"^[a-z]\)\s+", text):
            return True
        if text.startswith(("#", "$", ">")):
            return True
        if "|" in text and re.search(r"\b(grep|awk|sed|lspci|python|bash|sh|inm_[a-z_]+)\b", text, re.IGNORECASE):
            return True
        if text.count("/") >= 2 and re.search(r"\b(sh|py|log|txt|cfg|yaml)\b", text, re.IGNORECASE):
            return True
        return False

    @classmethod
    def _extract_heading_page_number(cls, line: str) -> Optional[int]:
        text = (line or "").strip()
        if not text:
            return None
        match = re.search(r"(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
        if not match:
            match = re.search(r"\((\d{1,4})\)\s*$", text)
        if not match:
            return None
        try:
            value = int(match.group(1))
            if value <= 0:
                return None
            return value
        except Exception:
            return None

    @staticmethod
    def _clean_heading_title(line: str) -> str:
        text = (line or "").strip()
        if not text:
            return ""
        text = re.sub(r"(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", "", text)
        text = re.sub(r"\((\d{1,4})\)\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -—:：·•")
        return text[:120]

    @classmethod
    def _heading_level(cls, line: str) -> Optional[int]:
        if cls._is_noise_heading_line(line):
            return None
        markdown = re.match(r"^(#{1,6})\s+", line)
        if markdown:
            return len(markdown.group(1))
        if re.match(r"^(chapter|part|section)\s+[ivxlcdm]+\b", line, re.IGNORECASE):
            return 1
        if re.match(r"^[ivxlcdm]+(?:\.|\)|:|\s+)\s+", line, re.IGNORECASE):
            level = 1
            dot_match = re.match(r"^([ivxlcdm]+(?:\.[ivxlcdm]+){0,5})\s+", line, re.IGNORECASE)
            if dot_match:
                level = min(dot_match.group(1).count(".") + 1, 6)
            return level
        if re.match(r"^\d+(?:\.\d+){0,5}\s+", line):
            return min(line.count(".") + 1, 6)
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇].{0,40}$", line):
            return 1
        if re.match(r"^[A-Z][A-Z0-9\s\-_:]{4,80}$", line):
            return 2
        return None

    def _build_markers(
        self,
        lines: Sequence[str],
        page_by_line: Sequence[int],
        has_explicit_page_number: bool,
    ) -> List[Dict[str, Any]]:
        if not self.allow_heading_detection():
            return []

        layered_builders = [
            lambda: self._build_markers_from_style_headings(lines, page_by_line),
            lambda: self._build_markers_from_metadata_catalog(
                lines, page_by_line, key="native_catalog", kind="native-catalog"
            ),
            lambda: self._build_markers_from_manual_toc(lines, page_by_line),
            lambda: self._build_markers_from_metadata_catalog(
                lines, page_by_line, key="style_catalog", kind="style-catalog"
            ),
            lambda: self._build_markers_from_metadata_catalog(
                lines, page_by_line, key="font_catalog", kind="font-catalog"
            ),
            lambda: self._build_markers_from_text_pattern(lines, page_by_line, has_explicit_page_number),
        ]

        for builder in layered_builders:
            markers = builder()
            if markers:
                markers.sort(key=lambda item: item["idx"])
                return markers

        return []

    @staticmethod
    def _normalized_heading_text(value: str) -> str:
        text = (value or "").strip().lower()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    @classmethod
    def _infer_level_from_title(cls, title: str, fallback: int = 1) -> int:
        text = (title or "").strip()
        if not text:
            return max(1, fallback)
        level = cls._heading_level(text)
        if level is None:
            return max(1, fallback)
        return max(1, min(level, 6))

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        try:
            number = int(float(str(value).strip()))
        except Exception:
            return None
        if number <= 0:
            return None
        return number

    @classmethod
    def _find_title_line_idx(
        cls,
        lines: Sequence[str],
        title: str,
        page_by_line: Sequence[int],
        page_hint: Optional[int],
        used_indices: set[int],
        start_from: int,
        toc_end_idx: Optional[int] = None,
    ) -> Optional[int]:
        title_norm = cls._normalized_heading_text(title)
        if not title_norm:
            return None

        candidates: List[int] = []
        for idx in range(max(0, start_from), len(lines)):
            if idx in used_indices:
                continue
            if toc_end_idx is not None and idx <= toc_end_idx:
                continue
            raw_line = lines[idx] or ""
            line = raw_line.strip()
            if not line or cls._is_form_feed_line(raw_line):
                continue
            line_norm = cls._normalized_heading_text(line)
            if not line_norm:
                continue
            if line_norm == title_norm or title_norm in line_norm or line_norm in title_norm:
                candidates.append(idx)

        if candidates:
            return min(candidates)

        if page_hint is not None and page_by_line:
            best_idx: Optional[int] = None
            best_gap: Optional[int] = None
            for idx in range(max(0, start_from), min(len(lines), len(page_by_line))):
                if idx in used_indices:
                    continue
                if toc_end_idx is not None and idx <= toc_end_idx:
                    continue
                raw_line = lines[idx] or ""
                line = raw_line.strip()
                if not line or cls._is_form_feed_line(raw_line):
                    continue
                page_value = cls.coerce_page_number(page_by_line[idx])
                if page_value is None:
                    continue
                gap = abs(page_value - page_hint)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_idx = idx
            if best_idx is not None:
                return best_idx

        return None

    def _native_catalog_logical_page_map(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        raw = self.metadata.get("native_catalog")
        if not isinstance(raw, list):
            return result
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = self._clean_heading_title(str(item.get("title") or ""))
            if not title:
                continue
            page = self._coerce_positive_int(item.get("page"))
            if page is None:
                continue
            norm = self._normalized_heading_text(title)
            if not norm:
                continue
            old = result.get(norm)
            result[norm] = page if old is None else min(old, page)
        return result

    def _build_markers_from_style_headings(
        self,
        lines: Sequence[str],
        page_by_line: Sequence[int],
    ) -> List[Dict[str, Any]]:
        raw = self.metadata.get("style_catalog")
        if not isinstance(raw, list) or not raw:
            return []

        entries: List[tuple[str, int]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = self._clean_heading_title(str(item.get("title") or ""))
            if not title:
                continue
            level = self._coerce_positive_int(item.get("level")) or self._infer_level_from_title(title, fallback=1)
            entries.append((title, max(1, min(level, 6))))
        if not entries:
            return []

        logical_page_map = self._native_catalog_logical_page_map()
        title_physical_pages = self._resolve_title_physical_pages([title for title, _ in entries])
        toc_end_idx = self._detect_toc_end_idx(lines)

        markers: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        for title, level in entries:
            norm = self._normalized_heading_text(title)
            logical_page = logical_page_map.get(norm)
            search_page_hint = logical_page if self.source_extension == ".pdf" else None
            search_toc_end_idx = None if self.source_extension == ".pdf" else toc_end_idx
            idx = self._find_title_line_idx(
                lines,
                title=title,
                page_by_line=page_by_line,
                page_hint=search_page_hint,
                used_indices=used_indices,
                start_from=cursor,
                toc_end_idx=search_toc_end_idx,
            )
            if idx is None:
                continue
            used_indices.add(idx)
            cursor = idx

            fallback_page = logical_page or 1
            physical_page = self._physical_page_from_line_idx(page_by_line, idx, fallback=fallback_page)
            calibrated_page = title_physical_pages.get(norm)
            bounded_page = self._find_title_page_from_source(norm, min_page=logical_page or 1)
            if bounded_page is not None and bounded_page > 0:
                calibrated_page = bounded_page
            if calibrated_page is not None and calibrated_page > 0:
                physical_page = calibrated_page

            markers.append(
                {
                    "idx": idx,
                    "title": title,
                    "level": level,
                    "page": physical_page,
                    "logical_page": logical_page,
                    "kind": "style-catalog",
                }
            )
        return markers

    @staticmethod
    def _physical_page_from_line_idx(
        page_by_line: Sequence[int],
        idx: int,
        fallback: int = 1,
    ) -> int:
        if not page_by_line:
            return max(1, int(fallback or 1))
        if idx < 0:
            idx = 0
        if idx >= len(page_by_line):
            idx = len(page_by_line) - 1
        value = RAG_DB_Document.coerce_page_number(page_by_line[idx])
        if value is None or value <= 0:
            return max(1, int(fallback or 1))
        return value

    def _build_markers_from_metadata_catalog(
        self,
        lines: Sequence[str],
        page_by_line: Sequence[int],
        *,
        key: str,
        kind: str,
    ) -> List[Dict[str, Any]]:
        raw = self.metadata.get(key)
        if not isinstance(raw, list) or not raw:
            return []

        entries: List[_CatalogLineHit] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = self._clean_heading_title(str(item.get("title") or ""))
            if not title:
                continue
            page = self._coerce_positive_int(item.get("page")) or 1
            level = self._coerce_positive_int(item.get("level")) or self._infer_level_from_title(title, fallback=1)
            entries.append(_CatalogLineHit(idx=-1, title=title, page=page, level=max(1, min(level, 6))))

        if not entries:
            return []

        title_physical_pages = self._resolve_title_physical_pages([entry.title for entry in entries])

        markers: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        toc_end_idx = self._detect_toc_end_idx(lines)
        for entry in entries:
            search_page_hint = entry.page if self.source_extension == ".pdf" else None
            search_toc_end_idx = None if self.source_extension == ".pdf" else toc_end_idx
            idx = self._find_title_line_idx(
                lines,
                title=entry.title,
                page_by_line=page_by_line,
                page_hint=search_page_hint,
                used_indices=used_indices,
                start_from=cursor,
                toc_end_idx=search_toc_end_idx,
            )
            if idx is None:
                continue
            used_indices.add(idx)
            cursor = idx
            physical_page = self._physical_page_from_line_idx(page_by_line, idx, fallback=entry.page)
            title_norm = self._normalized_heading_text(entry.title)
            calibrated_page = title_physical_pages.get(title_norm)
            bounded_page = self._find_title_page_from_source(title_norm, min_page=entry.page)
            if bounded_page is not None and bounded_page > 0:
                calibrated_page = bounded_page
            if calibrated_page is not None and calibrated_page > 0:
                physical_page = calibrated_page
            markers.append(
                {
                    "idx": idx,
                    "title": entry.title,
                    "level": entry.level,
                    "page": physical_page,
                    "logical_page": entry.page,
                    "kind": kind,
                }
            )
        return markers

    def _build_markers_from_manual_toc(
        self,
        lines: Sequence[str],
        page_by_line: Sequence[int],
    ) -> List[Dict[str, Any]]:
        if not lines:
            return []

        toc_anchor, toc_end_idx = self._detect_toc_bounds(lines)

        begin = toc_anchor if toc_anchor is not None else 0
        end = min(len(lines), begin + 260)
        toc_hits: List[_CatalogLineHit] = []
        for idx in range(begin, end):
            raw_line = lines[idx] or ""
            text = raw_line.strip()
            if not text or self._is_form_feed_line(raw_line):
                continue
            match = re.match(r"^\s*([\u4e00-\u9fffA-Za-z0-9][^\n]{1,120}?)(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
            if not match:
                continue
            title = self._clean_heading_title(match.group(1))
            page = self._coerce_positive_int(match.group(2))
            if not title or page is None:
                continue
            if self._is_noise_heading_line(title):
                continue
            leading_spaces = len(text) - len(text.lstrip(" "))
            level = 1 + min(leading_spaces // 4, 5)
            level = max(level, self._infer_level_from_title(title, fallback=level))
            toc_hits.append(_CatalogLineHit(idx=idx, title=title, page=page, level=level))

        if len(toc_hits) < 2:
            return []

        title_physical_pages = self._resolve_title_physical_pages([hit.title for hit in toc_hits])

        markers: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        dynamic_toc_end_idx = max(hit.idx for hit in toc_hits)
        search_toc_end_idx = toc_end_idx if toc_end_idx is not None else dynamic_toc_end_idx
        if self.source_extension == ".pdf":
            search_toc_end_idx = None
        for hit in toc_hits:
            search_page_hint = hit.page if self.source_extension == ".pdf" else None
            idx = self._find_title_line_idx(
                lines,
                title=hit.title,
                page_by_line=page_by_line,
                page_hint=search_page_hint,
                used_indices=used_indices,
                start_from=cursor,
                toc_end_idx=search_toc_end_idx,
            )
            if idx is None:
                continue
            used_indices.add(idx)
            cursor = idx
            physical_page = self._physical_page_from_line_idx(page_by_line, idx, fallback=hit.page)
            title_norm = self._normalized_heading_text(hit.title)
            calibrated_page = title_physical_pages.get(title_norm)
            bounded_page = self._find_title_page_from_source(title_norm, min_page=hit.page)
            if bounded_page is not None and bounded_page > 0:
                calibrated_page = bounded_page
            if calibrated_page is not None and calibrated_page > 0:
                physical_page = calibrated_page
            markers.append(
                {
                    "idx": idx,
                    "title": hit.title,
                    "level": hit.level,
                    "page": physical_page,
                    "logical_page": hit.page,
                    "kind": "manual-toc",
                }
            )
        return markers

    def _build_markers_from_text_pattern(
        self,
        lines: Sequence[str],
        page_by_line: Sequence[int],
        has_explicit_page_number: bool,
    ) -> List[Dict[str, Any]]:

        markers: List[Dict[str, Any]] = []
        for index, raw_line in enumerate(lines):
            line = (raw_line or "").strip()
            if not line or self._is_form_feed_line(raw_line or ""):
                continue

            level = self._heading_level(line)
            if level is None:
                continue

            title = self._clean_heading_title(line)
            if not title:
                continue

            heading_page = self._extract_heading_page_number(line)
            line_page = page_by_line[index] if index < len(page_by_line) else 1
            marker_page = line_page

            markers.append(
                {
                    "idx": index,
                    "title": title,
                    "level": level,
                    "page": marker_page,
                    "logical_page": heading_page if has_explicit_page_number else None,
                    "kind": "heading",
                }
            )
        return markers

    @staticmethod
    def _is_toc_anchor_line(line: str) -> bool:
        lowered = (line or "").strip().lower()
        if not lowered:
            return False

        compact = re.sub(r"\s+", "", lowered)
        compact = compact.strip("-—_:：.·•")
        if compact in {"目录", "contents", "tableofcontents"}:
            return True
        if re.fullmatch(r"目录(?:[ivxlcdm]+|\d+)?", compact):
            return True
        if bool(re.search(r"\btable\s*of\s*contents\b", lowered)):
            return True
        if bool(re.fullmatch(r"contents(?:[ivxlcdm]+|\d+)?", compact)):
            return True
        return False

    @staticmethod
    def _is_toc_entry_line(line: str) -> bool:
        text = (line or "").strip()
        if not text or PDFRAGDocument._is_form_feed_line(line or ""):
            return False
        return bool(
            re.match(r"^\s*([\u4e00-\u9fffA-Za-z0-9][^\n]{1,120}?)(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
        )

    @classmethod
    def _detect_toc_bounds(cls, lines: Sequence[str]) -> tuple[Optional[int], Optional[int]]:
        if not lines:
            return None, None

        toc_anchor: Optional[int] = None
        scan_limit = min(len(lines), 500)
        for idx, raw in enumerate(lines[:scan_limit]):
            if cls._is_toc_anchor_line(raw):
                toc_anchor = idx
                break

        if toc_anchor is None:
            return None, None

        end_limit = min(len(lines), toc_anchor + 320)
        toc_end = toc_anchor
        toc_entries = 0
        non_toc_after_entries = 0
        for idx in range(toc_anchor, end_limit):
            raw_line = lines[idx] or ""
            text = raw_line.strip()
            if not text or cls._is_form_feed_line(raw_line):
                continue
            if cls._is_toc_entry_line(text):
                toc_entries += 1
                toc_end = idx
                non_toc_after_entries = 0
                continue
            if toc_entries >= 2:
                non_toc_after_entries += 1
                if non_toc_after_entries >= 3:
                    break

        if toc_entries < 2:
            return toc_anchor, toc_anchor
        return toc_anchor, toc_end

    @classmethod
    def _detect_toc_end_idx(cls, lines: Sequence[str]) -> Optional[int]:
        _, toc_end = cls._detect_toc_bounds(lines)
        return toc_end

    def _resolve_pdf_title_physical_pages(self, titles: Sequence[str]) -> Dict[str, int]:
        if self.source_extension != ".pdf":
            return {}
        if not titles:
            return {}

        if self._source_pdf_title_page_map_cache is None:
            self._source_pdf_title_page_map_cache = {}

        cache = self._source_pdf_title_page_map_cache
        requested_norms: Set[str] = set()
        for title in titles:
            norm = self._normalized_heading_text(title)
            if norm:
                requested_norms.add(norm)

        missing_norms = {norm for norm in requested_norms if norm not in cache}
        if missing_norms:
            cache.update(self._match_titles_to_source_pdf_pages(missing_norms))

        result: Dict[str, int] = {}
        for norm in requested_norms:
            page = cache.get(norm)
            if page is not None and page > 0:
                result[norm] = page
        return result

    def _resolve_office_title_physical_pages(self, titles: Sequence[str]) -> Dict[str, int]:
        if self.source_extension not in {".doc", ".docx"}:
            return {}
        if not titles:
            return {}

        if self._office_pdf_title_page_map_cache is None:
            self._office_pdf_title_page_map_cache = {}

        cache = self._office_pdf_title_page_map_cache
        requested_norms: Set[str] = set()
        for title in titles:
            norm = self._normalized_heading_text(title)
            if norm:
                requested_norms.add(norm)

        missing_norms = {norm for norm in requested_norms if norm not in cache}
        if missing_norms:
            cache.update(self._match_titles_to_office_pdf_pages(missing_norms))

        result: Dict[str, int] = {}
        for norm in requested_norms:
            page = cache.get(norm)
            if page is not None and page > 0:
                result[norm] = page
        return result

    def _resolve_title_physical_pages(self, titles: Sequence[str]) -> Dict[str, int]:
        if self.source_extension in {".doc", ".docx"}:
            return self._resolve_office_title_physical_pages(titles)
        if self.source_extension == ".pdf":
            return self._resolve_pdf_title_physical_pages(titles)
        return {}

    @staticmethod
    def _derive_section_end_page(start_page: int, next_start_page: Any) -> int:
        start = max(1, int(start_page or 1))
        next_start = RAG_DB_Document.coerce_page_number(next_start_page)
        if next_start is None or next_start <= start:
            return start
        return max(start, next_start - 1)

    @classmethod
    def _next_boundary_start_page(
        cls,
        markers: Sequence[Dict[str, Any]],
        index: int,
    ) -> Optional[int]:
        if index < 0 or index >= len(markers):
            return None
        current_level = cls.coerce_page_number(markers[index].get("level")) or 1
        for idx in range(index + 1, len(markers)):
            candidate_level = cls.coerce_page_number(markers[idx].get("level")) or current_level
            if candidate_level > current_level:
                continue
            candidate_page = cls.coerce_page_number(markers[idx].get("page"))
            if candidate_page is None or candidate_page <= 0:
                continue
            return candidate_page
        return None

    def _load_source_pdf_page_norms(self) -> List[str]:
        if self._source_pdf_page_norms_cache is not None:
            return self._source_pdf_page_norms_cache or []

        file_path = str(self.metadata.get("file_name") or "").strip()
        if not file_path or not os.path.isfile(file_path) or fitz is None:
            self._source_pdf_page_norms_cache = []
            return []

        try:
            page_texts: List[str] = []
            with fitz.open(file_path) as document:
                for page_index in range(int(document.page_count or 0)):
                    page = document.load_page(page_index)
                    page_texts.append(str(page.get_text("text") or "").strip())

            self._source_pdf_page_norms_cache = [self._normalized_heading_text(text) for text in page_texts]
            return self._source_pdf_page_norms_cache
        except Exception as exc:
            logger.debug("source pdf page map failed for %s: %s", file_path, exc)
            self._source_pdf_page_norms_cache = []
            return []

    def _match_titles_to_source_pdf_pages(self, normalized_titles: Set[str]) -> Dict[str, int]:
        if not normalized_titles:
            return {}
        matched: Dict[str, int] = {}
        for norm in sorted(normalized_titles, key=len, reverse=True):
            if len(norm) < 2:
                continue
            page = self._find_source_pdf_title_page(norm, min_page=1)
            if page is not None and page > 0:
                matched[norm] = page
        return matched

    def _load_office_pdf_page_norms(self) -> List[str]:
        if self._office_pdf_page_norms_cache is not None:
            return self._office_pdf_page_norms_cache or []

        if self.source_extension not in {".doc", ".docx"} or fitz is None:
            self._office_pdf_page_norms_cache = []
            self._office_pdf_page_texts_cache = []
            return []

        office = shutil.which("soffice") or shutil.which("libreoffice")
        file_path = str(self.metadata.get("file_name") or "").strip()
        if not office or not file_path or not os.path.isfile(file_path):
            self._office_pdf_page_norms_cache = []
            self._office_pdf_page_texts_cache = []
            return []

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                command = [
                    office,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmp_dir,
                    file_path,
                ]
                subprocess.run(command, capture_output=True, timeout=120, check=False)

                expected_pdf = os.path.join(
                    tmp_dir,
                    f"{os.path.splitext(os.path.basename(file_path))[0]}.pdf",
                )
                pdf_path = expected_pdf
                if not os.path.isfile(pdf_path):
                    candidates = [
                        os.path.join(tmp_dir, name)
                        for name in os.listdir(tmp_dir)
                        if name.lower().endswith(".pdf")
                    ]
                    if not candidates:
                        self._office_pdf_page_norms_cache = []
                        self._office_pdf_page_texts_cache = []
                        return []
                    pdf_path = sorted(candidates)[0]

                page_texts: List[str] = []
                with fitz.open(pdf_path) as document:
                    for page_index in range(int(document.page_count or 0)):
                        page = document.load_page(page_index)
                        page_texts.append(str(page.get_text("text") or "").strip())

                self._office_pdf_page_texts_cache = page_texts
                self._office_pdf_page_norms_cache = [
                    self._normalized_heading_text(text)
                    for text in page_texts
                ]
                return self._office_pdf_page_norms_cache
        except Exception as exc:
            logger.debug("office pdf page map failed for %s: %s", file_path, exc)
            self._office_pdf_page_norms_cache = []
            self._office_pdf_page_texts_cache = []
            return []

    def _detect_office_pdf_toc_end_page(self) -> int:
        if self._office_pdf_toc_end_page_cache is not None:
            return max(0, int(self._office_pdf_toc_end_page_cache))

        _ = self._load_office_pdf_page_norms()
        page_texts = self._office_pdf_page_texts_cache or []
        if not page_texts:
            self._office_pdf_toc_end_page_cache = 0
            return 0

        toc_end = 0
        in_toc = False
        blank_or_non_toc_streak = 0
        for page_idx, page_text in enumerate(page_texts, start=1):
            lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
            if not lines:
                if in_toc:
                    blank_or_non_toc_streak += 1
                    if blank_or_non_toc_streak >= 2:
                        break
                continue

            has_anchor = any(self._is_toc_anchor_line(line) for line in lines)
            toc_entries = sum(1 for line in lines if self._is_toc_entry_line(line))

            if has_anchor:
                in_toc = True
                toc_end = page_idx
                blank_or_non_toc_streak = 0
                continue

            if in_toc:
                if toc_entries >= 2:
                    toc_end = page_idx
                    blank_or_non_toc_streak = 0
                    continue
                blank_or_non_toc_streak += 1
                if blank_or_non_toc_streak >= 2:
                    break
                continue

            if page_idx <= 8 and toc_entries >= 10:
                in_toc = True
                toc_end = page_idx
                blank_or_non_toc_streak = 0

        self._office_pdf_toc_end_page_cache = max(0, int(toc_end))
        return max(0, int(toc_end))

    def _detect_office_pdf_page_number_offset(self) -> int:
        if self._office_pdf_page_number_offset_cache is not None:
            return int(self._office_pdf_page_number_offset_cache)

        _ = self._load_office_pdf_page_norms()
        page_texts = self._office_pdf_page_texts_cache or []
        if not page_texts:
            self._office_pdf_page_number_offset_cache = 0
            return 0

        for page_idx, page_text in enumerate(page_texts[:12], start=1):
            lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
            if not lines:
                continue
            for line in lines:
                if "目录" not in line and "contents" not in line.lower():
                    continue
                match = re.search(r"(\d{1,4})\s*$", line)
                if not match:
                    continue
                marker = self._coerce_positive_int(match.group(1))
                if marker is None:
                    continue
                delta = marker - page_idx
                if abs(delta) <= 30:
                    self._office_pdf_page_number_offset_cache = int(delta)
                    return int(delta)

        self._office_pdf_page_number_offset_cache = 1
        return 1

    def _match_titles_to_office_pdf_pages(self, normalized_titles: Set[str]) -> Dict[str, int]:
        if not normalized_titles:
            return {}
        matched: Dict[str, int] = {}
        for norm in sorted(normalized_titles, key=len, reverse=True):
            if len(norm) < 2:
                continue
            page = self._find_office_pdf_title_page(norm, min_page=1)
            if page is not None and page > 0:
                matched[norm] = page
        return matched

    def _load_native_split_page_norms(self) -> Dict[int, str]:
        if self._native_split_page_norms_cache is not None:
            return self._native_split_page_norms_cache

        page_lines: Dict[int, List[str]] = {}
        page_no = 1
        for raw_line in self.cleaned_text.split("\n"):
            if self._is_form_feed_line(raw_line or ""):
                page_no += 1
                continue
            text = (raw_line or "").strip()
            if not text:
                continue
            page_lines.setdefault(page_no, []).append(text)

        norms: Dict[int, str] = {}
        for pno, lines in page_lines.items():
            norms[pno] = self._normalized_heading_text("\n".join(lines))

        self._native_split_page_norms_cache = norms
        return norms

    def _find_source_pdf_title_page(self, normalized_title: str, min_page: int = 1) -> Optional[int]:
        if self.source_extension != ".pdf":
            return None
        norm = (normalized_title or "").strip()
        if not norm:
            return None

        native_split_norms = self._load_native_split_page_norms()
        if native_split_norms:
            max_page = max(native_split_norms)
            start_page = max(1, int(min_page or 1))
            if start_page > max_page:
                start_page = max_page
            for page_idx in range(start_page, max_page + 1):
                page_norm = native_split_norms.get(page_idx, "")
                if not page_norm:
                    continue
                if norm in page_norm or page_norm in norm:
                    return page_idx

        page_norms = self._load_source_pdf_page_norms()
        if not page_norms:
            return None

        start_page = max(1, int(min_page or 1))
        if start_page > len(page_norms):
            start_page = len(page_norms)
        for page_idx in range(start_page, len(page_norms) + 1):
            page_norm = page_norms[page_idx - 1]
            if not page_norm:
                continue
            if norm in page_norm or page_norm in norm:
                return page_idx
        return None

    def _find_office_pdf_title_page(self, normalized_title: str, min_page: int = 1) -> Optional[int]:
        if self.source_extension not in {".doc", ".docx"}:
            return None
        norm = (normalized_title or "").strip()
        if not norm:
            return None

        page_norms = self._load_office_pdf_page_norms()
        if not page_norms:
            return None

        toc_end_page = self._detect_office_pdf_toc_end_page()
        start_page = max(1, int(min_page or 1), toc_end_page + 1)
        if start_page > len(page_norms):
            start_page = len(page_norms)
        for page_idx in range(start_page, len(page_norms) + 1):
            page_norm = page_norms[page_idx - 1]
            if not page_norm:
                continue
            if norm in page_norm or page_norm in norm:
                offset = self._detect_office_pdf_page_number_offset()
                calibrated = page_idx + offset
                return calibrated if calibrated > 0 else page_idx
        return None

    def _find_title_page_from_source(self, normalized_title: str, min_page: int = 1) -> Optional[int]:
        if self.source_extension == ".pdf":
            return self._find_source_pdf_title_page(normalized_title, min_page=min_page)
        if self.source_extension in {".doc", ".docx"}:
            return self._find_office_pdf_title_page(normalized_title, min_page=min_page)
        return None

    def _build_section_chunks(self, lines: Sequence[str], markers: Sequence[Dict[str, Any]]) -> List[Document]:
        hierarchy: List[str] = []
        section_docs: List[Document] = []

        for section_index, marker in enumerate(markers, start=1):
            start = marker["idx"]
            end = markers[section_index]["idx"] if section_index < len(markers) else len(lines)
            section_text = "\n".join(lines[start:end]).strip()
            if len(section_text) < 40:
                continue

            level = marker["level"]
            while len(hierarchy) >= level:
                hierarchy.pop()
            hierarchy.append(marker["title"])
            section_path = " > ".join(hierarchy)

            boundary_page = self._next_boundary_start_page(markers, section_index - 1)
            section_end_page = self._derive_section_end_page(marker["page"], boundary_page)

            metadata = self._build_chunk_metadata_base()
            metadata.update(
                {
                    "doc_id": self.base_doc_id,
                    "doc_name": self.doc_name,
                    "section_id": f"{self.base_doc_id}::sec:{section_index}",
                    "section_title": marker["title"],
                    "section_path": section_path,
                    "section_level": level,
                    "section_index": section_index,
                    "section_start_page": marker["page"],
                    "section_end_page": section_end_page,
                    "structure_generated": marker["kind"],
                    "pagination_mode": self.pagination_mode,
                }
            )
            for key in list(metadata.keys()):
                metadata[key] = self._trim_metadata_value(metadata[key])
            text_chunks = self._split_text_for_chunks(section_text)
            if not text_chunks:
                continue

            total_parts = len(text_chunks)
            for part_index, chunk_text in enumerate(text_chunks, start=1):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_part"] = part_index
                chunk_metadata["chunk_parts"] = total_parts
                if total_parts > 1:
                    chunk_metadata["section_id"] = f"{metadata['section_id']}::chunk:{part_index}"
                chunk_doc_id = (
                    f"{self.base_doc_id}::sec:{section_index}::chunk:{part_index}"
                    if total_parts > 1
                    else f"{self.base_doc_id}::sec:{section_index}"
                )
                section_docs.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        doc_id=chunk_doc_id,
                    )
                )

        return section_docs

    def _build_fixed_chunks(self, text: str) -> List[Document]:
        target_chars, _ = self._chunk_char_sizes()
        page_size = max(900, target_chars)
        sections: List[Document] = []
        for section_index, start in enumerate(range(0, len(text), page_size), start=1):
            section_text = text[start : start + page_size].strip()
            if len(section_text) < 80:
                continue
            metadata = self._build_chunk_metadata_base()
            metadata.update(
                {
                    "doc_id": self.base_doc_id,
                    "doc_name": self.doc_name,
                    "section_id": f"{self.base_doc_id}::part:{section_index}",
                    "section_title": f"Part {section_index}",
                    "section_path": f"Part {section_index}",
                    "section_level": 1,
                    "section_index": section_index,
                    "section_start_page": section_index,
                    "section_end_page": section_index,
                    "structure_generated": "fixed-window",
                    "pagination_mode": self.pagination_mode,
                }
            )
            for key in list(metadata.keys()):
                metadata[key] = self._trim_metadata_value(metadata[key])
            text_chunks = self._split_text_for_chunks(section_text)
            if not text_chunks:
                continue
            total_parts = len(text_chunks)
            for part_index, chunk_text in enumerate(text_chunks, start=1):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_part"] = part_index
                chunk_metadata["chunk_parts"] = total_parts
                if total_parts > 1:
                    chunk_metadata["section_id"] = f"{metadata['section_id']}::chunk:{part_index}"
                chunk_doc_id = (
                    f"{self.base_doc_id}::part:{section_index}::chunk:{part_index}"
                    if total_parts > 1
                    else f"{self.base_doc_id}::part:{section_index}"
                )
                sections.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        doc_id=chunk_doc_id,
                    )
                )
        return sections

    def _build_single_chunk(self, text: str) -> Document:
        metadata = self._build_chunk_metadata_base()
        metadata.update(
            {
                "doc_id": self.base_doc_id,
                "doc_name": self.doc_name,
                "section_id": f"{self.base_doc_id}::full:1",
                "section_title": "Full Document",
                "section_path": "Full Document",
                "section_level": 1,
                "section_index": 1,
                "section_start_page": 1,
                "section_end_page": 1,
                "structure_generated": "single-chunk",
                "pagination_mode": self.pagination_mode,
            }
        )
        for key in list(metadata.keys()):
            metadata[key] = self._trim_metadata_value(metadata[key])
        return Document(
            text=text,
            metadata=metadata,
            doc_id=f"{self.base_doc_id}::full:1",
        )

    def _build_catalog_from_markers(self, markers: Sequence[Dict[str, Any]]) -> List[CatalogEntry]:
        catalog: Dict[str, CatalogEntry] = {}
        for index, marker in enumerate(markers):
            title = str(marker.get("title") or "").strip()
            if not title:
                continue
            page = int(marker.get("page") or 1)
            boundary_page = self._next_boundary_start_page(markers, index)
            end_page = self._derive_section_end_page(page, boundary_page)
            existing = catalog.get(title)
            if existing is None:
                catalog[title] = CatalogEntry(title=title, page=page, end_page=end_page)
                continue
            existing.page = min(existing.page, page)
            existing.end_page = max(existing.end_page, end_page)
        return list(catalog.values())

    @staticmethod
    def _build_catalog_from_chunks(chunks: Sequence[Document]) -> List[CatalogEntry]:
        catalog: List[CatalogEntry] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            title = str(metadata.get("section_path") or metadata.get("section_title") or "").strip()
            if not title:
                continue
            start_page = int(metadata.get("section_start_page") or 1)
            end_page = int(metadata.get("section_end_page") or start_page)
            catalog.append(CatalogEntry(title=title, page=start_page, end_page=max(start_page, end_page)))
        return catalog
