from __future__ import annotations

import logging
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional

import fitz
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from rag.ocr import OCRPDFReader, ocr_enabled

logger = logging.getLogger(__name__)


def _resolve_layout_engine() -> Optional[str]:
    for dist_name in ("pymupdf-layout", "pymupdf_layout"):
        try:
            importlib_metadata.version(dist_name)
            return "pymupdf-layout"
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


class PyMuPDF4LLMPDFReader(BaseReader):
    def __init__(self, ocr_fallback: Optional[BaseReader] = None) -> None:
        self.ocr_fallback = ocr_fallback
        self.layout_engine = _resolve_layout_engine()

    def _fallback_text_extract(self, file: Path) -> str:
        parts: List[str] = []
        with fitz.open(file) as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_text = (page.get_text("text") or "").strip()
                if page_text:
                    parts.append(page_text)
        return "\n\f\n".join(parts).strip()

    def _extract_text_by_native_pages(self, file: Path) -> tuple[str, int]:
        parts: List[str] = []
        page_count = 0
        with fitz.open(file) as document:
            page_count = int(document.page_count or 0)
            for page_index in range(page_count):
                page = document.load_page(page_index)
                page_text = (page.get_text("text") or "").strip()
                parts.append(page_text)
        return "\n\f\n".join(parts).strip(), page_count

    def _extract_native_catalog(self, file: Path) -> List[Dict[str, int | str]]:
        catalog: List[Dict[str, int | str]] = []
        try:
            with fitz.open(file) as document:
                toc_items = document.get_toc(simple=True) or []
            for item in toc_items:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                level = int(item[0] or 1)
                title = str(item[1] or "").strip()
                page = int(item[2] or 0)
                if not title or page <= 0:
                    continue
                catalog.append({"title": title[:160], "page": page, "level": max(1, min(level, 6))})
        except Exception as exc:
            logger.debug("Failed to extract PDF native catalog for %s: %s", file, exc)
        return catalog

    @staticmethod
    def _normalize_heading_text(value: str) -> str:
        text = (value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\u4e00-\u9fff\s]", "", text)
        return text

    def _extract_font_catalog(self, file: Path, *, max_items: int = 120) -> List[Dict[str, int | str]]:
        candidates: List[tuple[float, str, int]] = []
        try:
            with fitz.open(file) as document:
                page_count = int(document.page_count or 0)
                for page_index in range(page_count):
                    page = document.load_page(page_index)
                    text_dict = page.get_text("dict") or {}
                    blocks = text_dict.get("blocks") if isinstance(text_dict, dict) else []
                    if not isinstance(blocks, list):
                        continue
                    for block in blocks:
                        lines = block.get("lines") if isinstance(block, dict) else []
                        if not isinstance(lines, list):
                            continue
                        for line_obj in lines:
                            spans = line_obj.get("spans") if isinstance(line_obj, dict) else []
                            if not isinstance(spans, list) or not spans:
                                continue
                            text_parts: List[str] = []
                            max_size = 0.0
                            for span in spans:
                                if not isinstance(span, dict):
                                    continue
                                text_parts.append(str(span.get("text") or ""))
                                try:
                                    max_size = max(max_size, float(span.get("size") or 0.0))
                                except Exception:
                                    continue
                            line_text = "".join(text_parts).strip()
                            if not line_text:
                                continue
                            if len(line_text) > 90:
                                continue
                            if re.match(r"^\d+$", line_text):
                                continue
                            if max_size <= 0:
                                continue
                            candidates.append((max_size, line_text, page_index + 1))
        except Exception as exc:
            logger.debug("Failed to extract PDF font catalog for %s: %s", file, exc)
            return []

        if not candidates:
            return []

        sorted_sizes = sorted(size for size, _, _ in candidates)
        q_index = max(0, int(len(sorted_sizes) * 0.85) - 1)
        threshold = sorted_sizes[q_index]

        seen: set[tuple[str, int]] = set()
        font_catalog: List[Dict[str, int | str]] = []
        for size, text, page in sorted(candidates, key=lambda row: (row[2], -row[0])):
            if size < threshold:
                continue
            normalized = self._normalize_heading_text(text)
            if not normalized:
                continue
            key = (normalized, page)
            if key in seen:
                continue
            seen.add(key)
            font_catalog.append({"title": text[:160], "page": int(page), "level": 1})
            if len(font_catalog) >= max_items:
                break
        return font_catalog

    def load_data(
        self, file: Path, extra_info: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        try:
            if hasattr(fitz.TOOLS, "mupdf_display_errors"):
                fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        parser = "pymupdf-native-pages"
        text = ""
        page_count = 0
        try:
            text, page_count = self._extract_text_by_native_pages(file)
        except Exception as exc:
            logger.warning(
                "PyMuPDF native page extraction failed for %s, fallback to parser chain: %s",
                file,
                exc,
            )
            if self.ocr_fallback is not None:
                docs = self.ocr_fallback.load_data(file, extra_info=extra_info)
                if docs:
                    return docs
            parser = "pymupdf"
            try:
                text = self._fallback_text_extract(file)
                with fitz.open(file) as document:
                    page_count = int(document.page_count or 0)
            except Exception as fallback_exc:
                logger.warning("PyMuPDF fallback parse failed for %s: %s", file, fallback_exc)
                return []

        if not text:
            return []

        metadata = {
            "file_name": str(file),
            "pdf_parser": parser,
            "layout_engine": self.layout_engine or "none",
            "native_pagination": True,
            "native_page_count": page_count,
            "source_extension": ".pdf",
            "native_catalog": self._extract_native_catalog(file),
            "font_catalog": self._extract_font_catalog(file),
        }
        if extra_info:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata)]


def get_pdf_reader() -> BaseReader:
    return PyMuPDF4LLMPDFReader(ocr_fallback=OCRPDFReader() if ocr_enabled() else None)
