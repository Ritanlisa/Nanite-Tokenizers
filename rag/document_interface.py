from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set

from llama_index.core import Document


class RAG_DB_Document(ABC):
    @abstractmethod
    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        ...

    @abstractmethod
    def build(self) -> "RAG_DB_Document":
        ...

    @abstractmethod
    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        ...

    @abstractmethod
    def allow_heading_detection(self) -> bool:
        ...

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
        ...

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
        ...

    @abstractmethod
    def list_payload(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def catalog_payload(self) -> List[Dict[str, Any]]:
        ...

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
