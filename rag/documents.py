from __future__ import annotations

import os
import re
import logging
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set

from llama_index.core import Document

import config
from rag.preprocessor import clean_document
from rag.pdf_reader import get_pdf_reader

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


class RAG_DB_Document(ABC):
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

    @abstractmethod
    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        raise NotImplementedError

    def allow_heading_detection(self) -> bool:
        return True

    @abstractmethod
    def retrieve_by_regex(
        self,
        *,
        compiled_regex: Optional[Pattern[str]],
        section: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        chunk: Optional[str],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

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

    @classmethod
    def resolve_doc_name_matches(
        cls,
        doc_name_query: str,
        available_doc_names: Set[str],
        *,
        data_dir: str = "",
    ) -> Set[str]:
        query = cls._normalize_doc_path(doc_name_query)
        if not query or not available_doc_names:
            return set()

        normalized_names = {
            raw: cls._normalize_doc_path(raw)
            for raw in available_doc_names
            if isinstance(raw, str) and raw.strip()
        }
        if not normalized_names:
            return set()

        query_abs = cls._normalize_doc_path(os.path.abspath(query))
        full_path_matches = {
            raw
            for raw, normalized in normalized_names.items()
            if cls._normalize_doc_path(os.path.abspath(normalized)) == query_abs
            or (
                bool(data_dir)
                and cls._normalize_doc_path(
                    os.path.abspath(os.path.join(data_dir, normalized))
                )
                == query_abs
            )
        }
        if full_path_matches:
            return full_path_matches

        relative_matches = {
            raw for raw, normalized in normalized_names.items() if normalized == query
        }
        if relative_matches:
            return relative_matches

        query_basename = os.path.basename(query)
        filename_matches = {
            raw
            for raw, normalized in normalized_names.items()
            if os.path.basename(normalized) == query_basename
        }
        if filename_matches:
            return filename_matches

        query_stem = os.path.splitext(query_basename)[0]
        if not query_stem:
            return set()
        return {
            raw
            for raw, normalized in normalized_names.items()
            if os.path.splitext(os.path.basename(normalized))[0] == query_stem
        }

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
        if current:
            return self._normalize_doc_path(current)

        file_name = str(metadata.get("file_name") or "").strip()
        if not file_name:
            return self.base_doc_id

        data_dir = str(config.settings.DATA_DIR or "").strip()
        if data_dir:
            abs_file = os.path.abspath(file_name)
            abs_data_dir = os.path.abspath(data_dir)
            if abs_file.startswith(abs_data_dir):
                rel = os.path.relpath(abs_file, abs_data_dir)
                return self._normalize_doc_path(rel)

        return self._normalize_doc_path(os.path.basename(file_name) or file_name)

    @staticmethod
    def _build_native_page_map(lines: Sequence[str]) -> tuple[List[int], bool, bool]:
        page_by_line: List[int] = []
        current_page = 1
        has_native_marker = False
        has_explicit_page_number = False
        for raw_line in lines:
            line = (raw_line or "").strip()
            if line == "\f":
                has_native_marker = True
                current_page += 1
                page_by_line.append(current_page)
                continue

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
    ) -> Optional[int]:
        title_norm = cls._normalized_heading_text(title)
        if not title_norm:
            return None

        candidates: List[int] = []
        for idx in range(max(0, start_from), len(lines)):
            if idx in used_indices:
                continue
            line = (lines[idx] or "").strip()
            if not line or line == "\f":
                continue
            line_norm = cls._normalized_heading_text(line)
            if not line_norm:
                continue
            if line_norm == title_norm or title_norm in line_norm or line_norm in title_norm:
                candidates.append(idx)

        if candidates:
            return min(candidates)

        if page_hint is not None and page_by_line:
            for idx in range(max(0, start_from), min(len(lines), len(page_by_line))):
                if idx in used_indices:
                    continue
                line = (lines[idx] or "").strip()
                if not line or line == "\f":
                    continue
                if page_by_line[idx] >= page_hint:
                    return idx

        return None

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

        markers: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        for entry in entries:
            idx = self._find_title_line_idx(
                lines,
                title=entry.title,
                page_by_line=page_by_line,
                page_hint=entry.page,
                used_indices=used_indices,
                start_from=cursor,
            )
            if idx is None:
                continue
            used_indices.add(idx)
            cursor = idx
            markers.append(
                {
                    "idx": idx,
                    "title": entry.title,
                    "level": entry.level,
                    "page": entry.page,
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

        toc_anchor: Optional[int] = None
        for idx, raw in enumerate(lines[: min(len(lines), 500)]):
            line = (raw or "").strip().lower()
            if line in {"目录", "contents", "table of contents"} or re.search(r"\bcontents\b", line):
                toc_anchor = idx
                break

        begin = toc_anchor if toc_anchor is not None else 0
        end = min(len(lines), begin + 260)
        toc_hits: List[_CatalogLineHit] = []
        for idx in range(begin, end):
            text = (lines[idx] or "").strip()
            if not text or text == "\f":
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
            level = min(level, self._infer_level_from_title(title, fallback=level))
            toc_hits.append(_CatalogLineHit(idx=idx, title=title, page=page, level=level))

        if len(toc_hits) < 2:
            return []

        markers: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        for hit in toc_hits:
            idx = self._find_title_line_idx(
                lines,
                title=hit.title,
                page_by_line=page_by_line,
                page_hint=hit.page,
                used_indices=used_indices,
                start_from=cursor,
            )
            if idx is None:
                continue
            used_indices.add(idx)
            cursor = idx
            markers.append(
                {
                    "idx": idx,
                    "title": hit.title,
                    "level": hit.level,
                    "page": hit.page,
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
            if not line or line == "\f":
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
            if self.pagination_mode != "native" and heading_page is not None:
                marker_page = heading_page
            if self.pagination_mode == "native" and has_explicit_page_number and heading_page is not None:
                marker_page = heading_page

            markers.append(
                {
                    "idx": index,
                    "title": title,
                    "level": level,
                    "page": marker_page,
                    "kind": "heading",
                }
            )
        return markers

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

            next_page = marker["page"]
            if section_index < len(markers):
                candidate_next_page = markers[section_index].get("page", next_page)
                if isinstance(candidate_next_page, int) and candidate_next_page >= next_page:
                    next_page = candidate_next_page

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
                    "section_end_page": max(marker["page"], next_page),
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
            end_page = page
            if index + 1 < len(markers):
                next_page = markers[index + 1].get("page")
                if isinstance(next_page, int):
                    end_page = max(page, next_page)
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


class NativePreferredRAGDocument(RAG_DB_Document):
    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        if has_native_marker:
            return list(native_page_map), "native"
        return self._build_manual_page_map(lines), "manual"

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


class ManualPreferredRAGDocument(RAG_DB_Document):
    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        return self._build_manual_page_map(lines), "manual"

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


class SpreadsheetRAGDocument(ManualPreferredRAGDocument):
    def allow_heading_detection(self) -> bool:
        return False


def stable_doc_id(doc: Document) -> str:
    import hashlib

    source = (doc.metadata or {}).get("file_name") or (doc.text or "")[:200]
    return hashlib.md5(str(source).encode()).hexdigest()


def _extract_doc_with_command(command: List[str]) -> str:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        return ""
    return (process.stdout or "").strip()


def _extract_office_text_with_page_breaks(file_path: str) -> tuple[str, bool]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return "", False
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            office,
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            tmp_dir,
            file_path,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=60, check=False)
            txt_name = f"{os.path.splitext(os.path.basename(file_path))[0]}.txt"
            txt_path = os.path.join(tmp_dir, txt_name)
            if not os.path.isfile(txt_path):
                return "", False
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            if not text.strip():
                return "", False
            has_native_breaks = "\f" in text
            return text.strip(), has_native_breaks
        except Exception as exc:
            logger.warning("libreoffice conversion failed for %s: %s", file_path, exc)
            return "", False


def _extract_legacy_doc_text(file_path: str) -> str:
    office_text, _ = _extract_office_text_with_page_breaks(file_path)
    if office_text:
        return office_text

    antiword = shutil.which("antiword")
    if antiword:
        text = _extract_doc_with_command([antiword, file_path])
        if text:
            return text

    catdoc = shutil.which("catdoc")
    if catdoc:
        text = _extract_doc_with_command([catdoc, file_path])
        if text:
            return text

    return ""


def _extract_docx_text_from_paragraph(paragraph: ET.Element, ns: Dict[str, str]) -> str:
    parts: List[str] = []
    for node in paragraph.findall('.//w:t', ns):
        if node.text:
            parts.append(node.text)
    return "".join(parts).strip()


def _docx_heading_level_from_style(style_id: str, style_name: str) -> Optional[int]:
    source = f"{style_id} {style_name}".lower()
    match = re.search(r"(?:heading|标题|toc)\s*([1-9])", source)
    if match:
        return max(1, min(int(match.group(1)), 6))
    return None


def _extract_docx_catalog_metadata(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    native_catalog: List[Dict[str, Any]] = []
    style_catalog: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            document_xml = archive.read("word/document.xml")
            styles_xml: Optional[bytes] = None
            try:
                styles_xml = archive.read("word/styles.xml")
            except Exception:
                styles_xml = None

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        style_name_by_id: Dict[str, str] = {}
        if styles_xml:
            styles_root = ET.fromstring(styles_xml)
            for style in styles_root.findall('.//w:style', ns):
                style_id = str(style.attrib.get(f"{{{ns['w']}}}styleId") or "").strip()
                if not style_id:
                    continue
                name_node = style.find('w:name', ns)
                if name_node is not None:
                    style_name = str(name_node.attrib.get(f"{{{ns['w']}}}val") or "").strip()
                    if style_name:
                        style_name_by_id[style_id] = style_name

        doc_root = ET.fromstring(document_xml)
        for paragraph in doc_root.findall('.//w:body/w:p', ns):
            text = _extract_docx_text_from_paragraph(paragraph, ns)
            if not text:
                continue
            p_style = paragraph.find('w:pPr/w:pStyle', ns)
            style_id = ""
            if p_style is not None:
                style_id = str(p_style.attrib.get(f"{{{ns['w']}}}val") or "").strip()
            if not style_id:
                continue
            style_name = style_name_by_id.get(style_id, "")
            lower_style = f"{style_id} {style_name}".lower()

            toc_match = re.match(r"^(.{1,160}?)(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
            if ("toc" in lower_style or "目录" in lower_style) and toc_match:
                page = int(toc_match.group(2))
                if page > 0:
                    native_catalog.append(
                        {
                            "title": toc_match.group(1).strip()[:160],
                            "page": page,
                            "level": _docx_heading_level_from_style(style_id, style_name) or 1,
                        }
                    )
                continue

            heading_level = _docx_heading_level_from_style(style_id, style_name)
            if heading_level is not None:
                style_catalog.append(
                    {
                        "title": text[:160],
                        "page": 1,
                        "level": heading_level,
                    }
                )
    except Exception as exc:
        logger.debug("Failed to extract DOCX structured catalog for %s: %s", file_path, exc)

    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[tuple[str, int, int]] = set()
        deduped: List[Dict[str, Any]] = []
        for item in items:
            title = str(item.get("title") or "").strip()
            page = int(item.get("page") or 1)
            level = int(item.get("level") or 1)
            if not title:
                continue
            key = (title.lower(), page, level)
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"title": title, "page": page, "level": level})
        return deduped

    return {
        "native_catalog": _dedupe(native_catalog),
        "style_catalog": _dedupe(style_catalog),
    }


def load_single_file_document(file_path: str, supported_extensions: Set[str]) -> Optional[Document]:
    from llama_index.readers.file import DocxReader, PandasExcelReader
    from llama_index.core import SimpleDirectoryReader

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext not in supported_extensions:
            logger.warning("Skipping unsupported extension %s for %s", ext, file_path)
            return None

        if ext in {".txt", ".md", ".markdown", ".csv"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            if not text.strip():
                return None
            return Document(
                text=text,
                metadata={"file_name": file_path, "source_extension": ext},
                doc_id=file_path,
            )

        if ext == ".doc":
            text = _extract_legacy_doc_text(file_path)
            if not text:
                logger.warning("Skipping .doc file with unsupported parser: %s", file_path)
                return None
            return Document(
                text=text,
                metadata={
                    "file_name": file_path,
                    "source_extension": ".doc",
                    "native_pagination": ("\f" in text),
                },
                doc_id=file_path,
            )

        if ext == ".docx":
            office_text, has_native_breaks = _extract_office_text_with_page_breaks(file_path)
            docx_catalogs = _extract_docx_catalog_metadata(file_path)
            if office_text:
                return Document(
                    text=office_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".docx",
                        "native_pagination": has_native_breaks,
                        "docx_parser": "libreoffice-txt",
                        "native_catalog": docx_catalogs.get("native_catalog") or [],
                        "style_catalog": docx_catalogs.get("style_catalog") or [],
                    },
                    doc_id=file_path,
                )

        file_extractors = {
            ".pdf": get_pdf_reader(),
            ".docx": DocxReader(),
            ".xlsx": PandasExcelReader(),
            ".xls": PandasExcelReader(),
        }
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            filename_as_id=True,
            file_extractor=file_extractors,
        )
        docs = reader.load_data()
        if not docs:
            return None
        first = docs[0]
        metadata = dict(first.metadata or {})
        metadata.setdefault("file_name", file_path)
        metadata.setdefault("source_extension", ext)
        if ext == ".docx":
            docx_catalogs = _extract_docx_catalog_metadata(file_path)
            metadata.setdefault("native_catalog", docx_catalogs.get("native_catalog") or [])
            metadata.setdefault("style_catalog", docx_catalogs.get("style_catalog") or [])
        return Document(
            text=first.text,
            metadata=metadata,
            doc_id=first.doc_id or file_path,
        )
    except Exception as exc:
        logger.warning("Failed to load document %s: %s", file_path, exc)
        return None


def build_rag_db_documents(loaded_docs: Sequence[Document]) -> List[RAG_DB_Document]:
    rag_docs: List[RAG_DB_Document] = []
    for loaded_doc in loaded_docs:
        try:
            doc_id = stable_doc_id(loaded_doc)
            rag_doc = create_rag_db_document(loaded_doc, stable_doc_id=doc_id).build()
        except Exception as exc:
            logger.warning("Failed to build RAG_DB_Document for %s: %s", loaded_doc.doc_id, exc)
            continue
        if not rag_doc.chunk_documents and not rag_doc.cleaned_text:
            continue
        rag_docs.append(rag_doc)
    return rag_docs


def chunk_documents_from_rag_documents(rag_docs: Sequence[RAG_DB_Document]) -> List[Document]:
    docs: List[Document] = []
    for rag_doc in rag_docs:
        docs.extend(rag_doc.chunk_documents)
    return docs


def prepare_documents_for_indexing_from_loaded_docs(loaded_docs: Sequence[Document]) -> List[Document]:
    rag_docs = build_rag_db_documents(loaded_docs)
    return chunk_documents_from_rag_documents(rag_docs)


def load_rag_documents_from_paths(paths: Sequence[str], supported_extensions: Set[str]) -> List[RAG_DB_Document]:
    loaded_docs: List[Document] = []
    for file_path in paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_extensions:
            logger.warning("Skipping unsupported upload file %s", file_path)
            continue
        loaded_doc = load_single_file_document(file_path, supported_extensions)
        if loaded_doc is None:
            continue
        loaded_docs.append(loaded_doc)
    return build_rag_db_documents(loaded_docs)


def load_chunk_documents_from_paths(paths: Sequence[str], supported_extensions: Set[str]) -> List[Document]:
    rag_docs = load_rag_documents_from_paths(paths, supported_extensions)
    return chunk_documents_from_rag_documents(rag_docs)


def collect_supported_document_paths(root_dir: str, supported_extensions: Set[str]) -> List[str]:
    if not root_dir or not os.path.isdir(root_dir):
        return []
    paths: List[str] = []
    for base, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in supported_extensions:
                continue
            paths.append(os.path.join(base, name))
    paths.sort()
    return paths


def load_chunk_documents_from_data_dir(
    data_dir: str,
    supported_extensions: Set[str],
) -> List[Document]:
    paths = collect_supported_document_paths(data_dir, supported_extensions)
    if not paths:
        return []
    return load_chunk_documents_from_paths(paths, supported_extensions)


def load_chunk_documents_from_persist_dir(
    persist_dir: str,
    supported_extensions: Set[str],
) -> List[Document]:
    docs_dir = os.path.join(persist_dir, "docs")
    paths = collect_supported_document_paths(docs_dir, supported_extensions)
    if not paths:
        return []
    return load_chunk_documents_from_paths(paths, supported_extensions)


def load_rag_documents_from_persist_dir(
    persist_dir: str,
    supported_extensions: Set[str],
) -> List[RAG_DB_Document]:
    docs_dir = os.path.join(persist_dir, "docs")
    paths = collect_supported_document_paths(docs_dir, supported_extensions)
    if not paths:
        return []
    return load_rag_documents_from_paths(paths, supported_extensions)


def create_rag_db_document(source_document: Document, stable_doc_id: str) -> RAG_DB_Document:
    metadata = dict(source_document.metadata or {})
    ext = str(metadata.get("source_extension") or "").strip().lower()
    if not ext:
        file_name = str(metadata.get("file_name") or "").strip()
        ext = os.path.splitext(file_name)[1].lower() if file_name else ""

    if ext in {".pdf", ".doc", ".docx"}:
        return NativePreferredRAGDocument(source_document, stable_doc_id)
    if ext in {".xlsx", ".xls"}:
        return SpreadsheetRAGDocument(source_document, stable_doc_id)
    return ManualPreferredRAGDocument(source_document, stable_doc_id)
