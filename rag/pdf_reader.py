from __future__ import annotations

import logging
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

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
    def __init__(self) -> None:
        self.layout_engine = _resolve_layout_engine()

    @staticmethod
    def _clean_heading_title(line: str) -> str:
        text = (line or "").strip()
        if not text:
            return ""
        text = re.sub(r"(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", "", text)
        text = re.sub(r"\((\d{1,4})\)\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -—:：·•")
        return text[:160]

    @staticmethod
    def _extract_heading_page_number(line: str) -> Optional[int]:
        text = (line or "").strip()
        if not text:
            return None
        match = re.search(r"(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
        if not match:
            match = re.search(r"\((\d{1,4})\)\s*$", text)
        if not match:
            return None
        try:
            page = int(match.group(1))
            return page if page > 0 else None
        except Exception:
            return None

    @staticmethod
    def _is_toc_anchor_line(line: str) -> bool:
        lowered = (line or "").strip().lower()
        if not lowered:
            return False
        compact = re.sub(r"\s+", "", lowered).strip("-—_:：.·•")
        if compact in {"目录", "contents", "tableofcontents"}:
            return True
        if bool(re.search(r"\btable\s*of\s*contents\b", lowered)):
            return True
        if bool(re.fullmatch(r"目录(?:[ivxlcdm]+|\d+)?", compact)):
            return True
        if bool(re.fullmatch(r"contents(?:[ivxlcdm]+|\d+)?", compact)):
            return True
        # "CONTENTS" (all caps) — common in Intel-style manuals
        if lowered == "contents" or (len(lowered) <= 19 and lowered.startswith("contents")):
            return True
        return False

    @classmethod
    def _is_toc_entry_line(cls, line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return False
        return bool(
            re.match(
                r"^\s*([\u4e00-\u9fffA-Za-z0-9][^\n]{1,160}?)(?:\t+|[·•.\s]{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*$",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _roman_to_int(roman: str) -> int:
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        prev = 0
        for ch in reversed(str(roman or "").upper().strip()):
            v = values.get(ch, 0)
            if v >= prev:
                total += v
            else:
                total -= v
            prev = v
        return total

    @classmethod
    def _infer_heading_level(cls, line: str) -> Optional[int]:
        text = (line or "").strip()
        if not text:
            return None
        markdown = re.match(r"^(#{1,6})\s+", text)
        if markdown:
            return len(markdown.group(1))
        num_match = re.match(r"^\s*(\d+(?:\.\d+){0,5})\s+", text)
        if num_match:
            return min(num_match.group(1).count(".") + 1, 6)
        roman_match = re.match(r"^\s*([ivxlcdm]+(?:\.[ivxlcdm]+){0,5})(?:\.|\)|:|\s+)\s+", text, re.IGNORECASE)
        if roman_match:
            return min(roman_match.group(1).count(".") + 1, 6)
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇].{0,40}$", text):
            return 1
        if re.match(r"^[A-Z][A-Z0-9\s\-_:]{4,80}$", text):
            return 2
        return None

    @classmethod
    def _infer_toc_level(cls, line: str) -> int:
        text = (line or "").rstrip("\n")
        explicit = cls._infer_heading_level(text)
        if explicit is not None:
            return max(1, min(explicit, 6))
        indent = len(text) - len(text.lstrip(" "))
        return max(1, min(indent // 4 + 1, 6))

    @classmethod
    def _dedupe_catalog(cls, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[tuple[str, int, int]] = set()
        deduped: List[Dict[str, Any]] = []
        for item in items:
            title = cls._clean_heading_title(str(item.get("title") or ""))
            page = int(item.get("page") or 1)
            level = int(item.get("level") or 1)
            if not title or page <= 0:
                continue
            key = (cls._normalize_heading_text(title), page, max(1, min(level, 6)))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            deduped.append({"title": title, "page": page, "level": key[2]})
        return deduped

    @classmethod
    def _collect_toc_entries_from_range(
        cls,
        page_texts: Sequence[str],
        toc_start: int,
        window: int,
    ) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        non_toc_streak = 0
        for page_idx in range(toc_start, min(len(page_texts), toc_start + window)):
            lines = [line for line in str(page_texts[page_idx] or "").splitlines() if line.strip()]
            page_hits = 0
            for line in lines:
                stripped = line.strip()
                match = re.match(
                    r"^\s*([\u4e00-\u9fffA-Za-z0-9][^\n]{1,160}?)(?:\t+|[·•.\s]{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*$",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if not match:
                    continue
                title = cls._clean_heading_title(match.group(1))
                if not title:
                    continue
                # Filter out page-number-only labels ("iv", "vi", "1-2" etc.)
                if re.match(r"^[ivxlcdm\d\-\s\.]{1,12}$", title, flags=re.IGNORECASE):
                    continue
                page_str = match.group(2).upper()
                try:
                    page = int(page_str)
                except ValueError:
                    page = cls._roman_to_int(page_str)
                if page <= 0:
                    continue
                level = cls._infer_toc_level(match.group(1))
                catalog.append({"title": title, "page": page, "level": level})
                page_hits += 1

            if page_hits == 0:
                non_toc_streak += 1
                if non_toc_streak >= 4 and catalog:
                    break
            else:
                non_toc_streak = 0
        return catalog

    @classmethod
    def _extract_toc_catalog_from_pages(cls, page_texts: Sequence[str]) -> List[Dict[str, Any]]:
        if not page_texts:
            return []

        total_pages = len(page_texts)
        if total_pages <= 100:
            scan_limit = min(total_pages, 24)
            collect_window = 12
        elif total_pages <= 1000:
            scan_limit = min(total_pages, 60)
            collect_window = 30
        else:
            scan_limit = min(total_pages, 200)
            collect_window = 60

        all_catalogs: List[Dict[str, Any]] = []

        def _find_and_collect(start_offset: int, scan_range: int) -> bool:
            toc_start: Optional[int] = None
            for page_idx in range(start_offset, min(total_pages, start_offset + scan_range)):
                lines = [line.strip() for line in str(page_texts[page_idx] or "").splitlines() if line.strip()]
                if any(cls._is_toc_anchor_line(line) for line in lines):
                    toc_start = page_idx
                    break
                toc_count = sum(1 for line in lines if cls._is_toc_entry_line(line))
                if page_idx - start_offset <= max(5, scan_range // 20) and toc_count >= 8:
                    toc_start = page_idx
                    break
            if toc_start is None:
                return False
            all_catalogs.extend(cls._collect_toc_entries_from_range(page_texts, toc_start, collect_window))
            return True

        # Primary pass: front of document
        _find_and_collect(0, scan_limit)

        # Multi-pass for large multi-volume PDFs
        if total_pages > 500:
            num_passes = 3 if total_pages > 2000 else 2
            step = max(1, total_pages // num_passes)
            for pass_idx in range(1, num_passes):
                offset = pass_idx * step
                _find_and_collect(offset, scan_limit // 2)

        return cls._dedupe_catalog(all_catalogs)

    @classmethod
    def _extract_style_catalog_from_pages(
        cls,
        page_texts: Sequence[str],
        *,
        max_items: int = 240,
    ) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            for line in str(page_text or "").splitlines():
                text = line.strip()
                if not text:
                    continue
                if cls._is_toc_anchor_line(text) or cls._is_toc_entry_line(text):
                    continue
                level = cls._infer_heading_level(text)
                if level is None:
                    continue
                title = cls._clean_heading_title(re.sub(r"^#{1,6}\s+", "", text))
                if not title:
                    continue
                if len(title) > 120:
                    continue
                catalog.append({"title": title, "page": page_idx, "level": max(1, min(level, 6))})
                if len(catalog) >= max_items:
                    break
            if len(catalog) >= max_items:
                break
        return cls._dedupe_catalog(catalog)

    @classmethod
    def _build_catalogs_from_pages(
        cls,
        page_texts: Sequence[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        total_pages = len(page_texts)
        native_catalog = cls._extract_toc_catalog_from_pages(page_texts)

        # Scale style catalog cap with document size
        if total_pages <= 500:
            max_style = 240
        elif total_pages <= 2000:
            max_style = 800
        else:
            max_style = 2000

        style_catalog = cls._extract_style_catalog_from_pages(page_texts, max_items=max_style)
        font_catalog = style_catalog[:min(200, max_style // 2)]
        return native_catalog, style_catalog, font_catalog

    def _extract_text_by_pymupdf4llm(self, file: Path) -> tuple[str, int, List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        import json
        import subprocess as _sp
        import sys as _sys

        try:
            import fitz
            doc = fitz.open(str(file))
            page_count = doc.page_count
            doc.close()
        except Exception:
            return "", 0, [], [], [], []

        if page_count <= 0:
            return "", 0, [], [], [], []

        # For single-chunk PDFs, try pymupdf4llm in a subprocess for richer markdown.
        if page_count <= 800:
            script = r"""
import json, sys
try:
    import pymupdf4llm
    chunks = pymupdf4llm.to_markdown(sys.argv[1], page_chunks=True)
    if not isinstance(chunks, list):
        sys.exit(1)
    parts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("text") or chunk.get("md") or chunk.get("markdown") or "").strip())
        else:
            parts.append(str(chunk or "").strip())
    print(json.dumps({"parts": parts}, ensure_ascii=False))
except Exception:
    sys.exit(1)
""".strip()
            try:
                proc = _sp.run(
                    [_sys.executable, "-c", script, str(file)],
                    capture_output=True, text=True, timeout=120, check=False,
                )
                if proc.returncode == 0:
                    result = json.loads(proc.stdout.strip())
                    enriched = list(result.get("parts") or [])
                    if enriched and len(enriched) >= page_count:
                        text = "\n\f\n".join(enriched).strip()
                        nc, sc, fc = self._build_catalogs_from_pages(enriched)
                        return text, len(enriched), enriched, nc, sc, fc
            except Exception:
                pass

        # For medium-large PDFs, try chunked pymupdf4llm for better quality.
        # Very large PDFs (>3000 pages) use fitz direct for speed.
        elif page_count <= 3000:
            try:
                from rag.document_pdf import _pymupdf4llm_extract_chunked
                all_pages = _pymupdf4llm_extract_chunked(str(file), page_count)
                if all_pages is not None:
                    page_parts = [
                        str(p.get("text") or p.get("md") or p.get("markdown") or "").strip()
                        for p in all_pages
                    ]
                    if page_parts and len(page_parts) >= page_count:
                        text = "\n\f\n".join(page_parts).strip()
                        nc, sc, fc = self._build_catalogs_from_pages(page_parts)
                        return text, len(page_parts), page_parts, nc, sc, fc
            except Exception:
                pass

        # Fast fallback: fitz.get_text() page by page
        try:
            doc = fitz.open(str(file))
            page_parts = [doc.load_page(i).get_text() for i in range(page_count)]
            doc.close()
            if page_parts:
                text = "\n\f\n".join(page_parts).strip()
                nc, sc, fc = self._build_catalogs_from_pages(page_parts)
                return text, len(page_parts), page_parts, nc, sc, fc
        except Exception:
            pass

        return "", 0, [], [], [], []

    @staticmethod
    def _normalize_heading_text(value: str) -> str:
        text = (value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\u4e00-\u9fff\s]", "", text)
        return text

    def load_data(
        self, file: Path, extra_info: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        parser = "pymupdf4llm"
        text = ""
        page_count = 0
        native_catalog: List[Dict[str, Any]] = []
        style_catalog: List[Dict[str, Any]] = []
        font_catalog: List[Dict[str, Any]] = []
        try:
            text, page_count, _, native_catalog, style_catalog, font_catalog = self._extract_text_by_pymupdf4llm(file)
        except Exception as exc:
            logger.warning("pymupdf4llm parse failed for %s: %s", file, exc)
            return []

        if not text:
            return []

        metadata = {
            "file_name": str(file),
            "pdf_parser": parser,
            "layout_engine": self.layout_engine or "none",
            "native_pagination": ("\f" in text),
            "native_page_count": page_count,
            "source_extension": ".pdf",
            "native_catalog": native_catalog,
            "style_catalog": style_catalog,
            "font_catalog": font_catalog,
        }
        if extra_info:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata)]


def get_pdf_reader() -> BaseReader:
    return PyMuPDF4LLMPDFReader()
