from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set

from llama_index.core import Document


class PageType:
    COVER = "cover"
    CATALOGUE = "catalogue"
    INTRODUCTION = "introduction"
    CONTENT = "content"
    APPENDIX = "appendix"
    CHAPTER = "chapter"


@dataclass(frozen=True)
class ImageAsset:
    asset_id: str
    filename: str = ""
    media_type: str = ""
    data: bytes = b""
    width: int = 0
    height: int = 0
    caption: str = ""
    source: str = ""
    page: int = 0

    @classmethod
    def from_bytes(
        cls,
        *,
        data: bytes,
        filename: str = "",
        media_type: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        caption: str = "",
        source: str = "",
        page: Optional[int] = None,
    ) -> "ImageAsset":
        payload = bytes(data or b"")
        name = str(filename or "").strip()
        mime = str(media_type or "").strip().lower()
        if (not mime) and name:
            guessed, _ = mimetypes.guess_type(name)
            mime = str(guessed or "").strip().lower()
        digest = hashlib.sha1(payload or name.encode("utf-8") or caption.encode("utf-8") or source.encode("utf-8")).hexdigest()
        return cls(
            asset_id=digest,
            filename=name,
            media_type=mime,
            data=payload,
            width=max(0, int(width or 0)),
            height=max(0, int(height or 0)),
            caption=str(caption or "").strip(),
            source=str(source or "").strip(),
            page=max(0, int(page or 0)),
        )

    @classmethod
    def from_payload(cls, value: Any) -> Optional["ImageAsset"]:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return None

        payload = dict(value)
        raw_data = payload.get("data")
        if isinstance(raw_data, str):
            try:
                raw_data = base64.b64decode(raw_data.encode("ascii"), validate=False)
            except Exception:
                raw_data = b""
        elif not isinstance(raw_data, (bytes, bytearray)):
            raw_data = b""

        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            digest_source = bytes(raw_data or b"")
            if not digest_source:
                digest_source = "|".join(
                    [
                        str(payload.get("filename") or "").strip(),
                        str(payload.get("caption") or "").strip(),
                        str(payload.get("source") or "").strip(),
                        str(payload.get("media_type") or "").strip(),
                    ]
                ).encode("utf-8")
            asset_id = hashlib.sha1(digest_source).hexdigest()

        return cls(
            asset_id=asset_id,
            filename=str(payload.get("filename") or "").strip(),
            media_type=str(payload.get("media_type") or "").strip().lower(),
            data=bytes(raw_data or b""),
            width=max(0, int(payload.get("width") or 0)),
            height=max(0, int(payload.get("height") or 0)),
            caption=str(payload.get("caption") or "").strip(),
            source=str(payload.get("source") or "").strip(),
            page=max(0, int(payload.get("page") or 0)),
        )

    @property
    def byte_size(self) -> int:
        return len(self.data or b"")

    @property
    def has_binary(self) -> bool:
        return bool(self.data)

    def to_payload(self, *, include_data: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
            "source": self.source,
            "page": self.page,
            "has_binary": self.has_binary,
        }
        if include_data and self.data:
            payload["data"] = base64.b64encode(self.data).decode("ascii")
        return payload

    def to_debug_payload(self) -> Dict[str, Any]:
        return self.to_payload(include_data=False)


def _normalize_image_asset(value: Any) -> Optional[Any]:
    if isinstance(value, ImageAsset):
        return value
    asset = ImageAsset.from_payload(value)
    if asset is not None:
        return asset
    text = str(value or "").strip()
    return text or None


def _image_asset_key(value: Any) -> str:
    normalized = _normalize_image_asset(value)
    if normalized is None:
        return ""
    if isinstance(normalized, ImageAsset):
        return f"image:{normalized.asset_id}"
    return f"text:{normalized}"


def _dedupe_text_values(values: Sequence[Any]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _dedupe_image_values(values: Sequence[Any]) -> List[Any]:
    seen: Set[str] = set()
    ordered: List[Any] = []
    for item in values:
        normalized = _normalize_image_asset(item)
        if normalized is None:
            continue
        key = _image_asset_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


@dataclass
class PageAssets:
    headers: List[str] = field(default_factory=list)
    footers: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    page_numbers: List[str] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)

    def merged(self, others: Sequence["PageAssets"]) -> "PageAssets":
        merged_assets = PageAssets(
            headers=list(self.headers),
            footers=list(self.footers),
            annotations=list(self.annotations),
            citations=list(self.citations),
            page_numbers=list(self.page_numbers),
            images=list(self.images),
        )
        for item in others:
            merged_assets.headers.extend(item.headers)
            merged_assets.footers.extend(item.footers)
            merged_assets.annotations.extend(item.annotations)
            merged_assets.citations.extend(item.citations)
            merged_assets.page_numbers.extend(item.page_numbers)
            merged_assets.images.extend(item.images)
        merged_assets.headers = _dedupe_text_values(merged_assets.headers)
        merged_assets.footers = _dedupe_text_values(merged_assets.footers)
        merged_assets.annotations = _dedupe_text_values(merged_assets.annotations)
        merged_assets.citations = _dedupe_text_values(merged_assets.citations)
        merged_assets.page_numbers = _dedupe_text_values(merged_assets.page_numbers)
        merged_assets.images = _dedupe_image_values(merged_assets.images)
        return merged_assets


class Page(ABC):
    def __init__(
        self,
        *,
        title: str,
        markdown_text: str = "",
        assets: Optional[PageAssets] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.title = str(title or "").strip()
        self.markdown_text = str(markdown_text or "")
        self.assets = assets or PageAssets()
        self.metadata = dict(metadata or {})

    @property
    @abstractmethod
    def category(self) -> str:
        ...

    @abstractmethod
    def flatten_mono_pages(self) -> List["MonoPage"]:
        ...

    def iter_pages(self) -> List["Page"]:
        return [self]

    def set_markdown(self, markdown_text: str) -> None:
        self.markdown_text = str(markdown_text or "")

    def append_markdown(self, text: str) -> None:
        part = str(text or "")
        if not part:
            return
        if self.markdown_text:
            self.markdown_text = f"{self.markdown_text}\n{part}"
        else:
            self.markdown_text = part

    def add_header(self, value: str) -> None:
        v = str(value or "").strip()
        if v:
            self.assets.headers.append(v)

    def add_footer(self, value: str) -> None:
        v = str(value or "").strip()
        if v:
            self.assets.footers.append(v)

    def add_annotation(self, value: str) -> None:
        v = str(value or "").strip()
        if v:
            self.assets.annotations.append(v)

    def add_citation(self, value: str) -> None:
        v = str(value or "").strip()
        if v:
            self.assets.citations.append(v)

    def add_page_number(self, value: Any) -> None:
        v = str(value).strip()
        if v:
            self.assets.page_numbers.append(v)

    def add_image(self, value: Any) -> None:
        normalized = _normalize_image_asset(value)
        if normalized is None:
            return
        self.assets.images.append(normalized)

    def get_headers(self) -> List[str]:
        return list(self.assets.headers)

    def get_footers(self) -> List[str]:
        return list(self.assets.footers)

    def get_annotations(self) -> List[str]:
        return list(self.assets.annotations)

    def get_citations(self) -> List[str]:
        return list(self.assets.citations)

    def get_page_numbers(self) -> List[str]:
        return list(self.assets.page_numbers)

    def get_images(self) -> List[Any]:
        return list(self.assets.images)

    def collect_assets(self) -> PageAssets:
        return PageAssets(
            headers=list(self.assets.headers),
            footers=list(self.assets.footers),
            annotations=list(self.assets.annotations),
            citations=list(self.assets.citations),
            page_numbers=list(self.assets.page_numbers),
            images=list(self.assets.images),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "markdown_text": self.markdown_text,
            "headers": self.get_headers(),
            "footers": self.get_footers(),
            "annotations": self.get_annotations(),
            "citations": self.get_citations(),
            "page_numbers": self.get_page_numbers(),
            "images": [item.to_payload() if isinstance(item, ImageAsset) else str(item or "") for item in self.get_images()],
            "metadata": dict(self.metadata),
        }


class MonoPage(Page):
    def __init__(
        self,
        *,
        title: str,
        page_type: str,
        markdown_text: str = "",
        assets: Optional[PageAssets] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            title=title,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )
        self.page_type = str(page_type or "").strip()

    @property
    def category(self) -> str:
        return self.page_type

    def flatten_mono_pages(self) -> List["MonoPage"]:
        return [self]


class Cover(MonoPage):
    def __init__(self, *, title: str = "Front Page", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            title=title,
            page_type=PageType.COVER,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )


class Catalogue(MonoPage):
    def __init__(self, *, title: str = "Table of Contents", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            title=title,
            page_type=PageType.CATALOGUE,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )


class Introduction(MonoPage):
    def __init__(self, *, title: str = "Preface", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            title=title,
            page_type=PageType.INTRODUCTION,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )


class Content(MonoPage):
    def __init__(self, *, title: str = "", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        normalized_title = str(title or "").strip()
        if normalized_title:
            raise AssertionError("Content title must be empty; use metadata['section_title'] for labels.")
        super().__init__(
            title="",
            page_type=PageType.CONTENT,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )


class SemiPage(Content):
    def __init__(
        self,
        *,
        markdown_text: str = "",
        assets: Optional[PageAssets] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        fragment_meta = dict(metadata or {})
        fragment_meta.setdefault("is_fragment", True)
        super().__init__(
            markdown_text=markdown_text,
            assets=assets,
            metadata=fragment_meta,
        )


class Appendix(MonoPage):
    def __init__(self, *, title: str = "Appendix", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            title=title,
            page_type=PageType.APPENDIX,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )


class Chapter(Page):
    def __init__(
        self,
        *,
        title: str,
        markdown_text: str = "",
        assets: Optional[PageAssets] = None,
        metadata: Optional[Dict[str, Any]] = None,
        SubContent: Optional[Sequence[Page]] = None,
    ) -> None:
        super().__init__(
            title=title,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
        )
        self.SubContent: List[Page] = list(SubContent or [])

    @property
    def category(self) -> str:
        return PageType.CHAPTER

    def add_child(self, page: Page) -> None:
        self.SubContent.append(page)

    def remove_child(self, page: Page) -> None:
        self.SubContent = [item for item in self.SubContent if item is not page]

    def flatten_mono_pages(self) -> List[MonoPage]:
        leaves: List[MonoPage] = []
        for item in self.SubContent:
            leaves.extend(item.flatten_mono_pages())
        return leaves

    def iter_pages(self) -> List[Page]:
        pages: List[Page] = [self]
        for item in self.SubContent:
            pages.extend(item.iter_pages())
        return pages

    def merged_markdown(self) -> str:
        parts: List[str] = []
        for item in self.SubContent:
            content = item.markdown_text.strip()
            if content:
                parts.append(content)
        own_text = self.markdown_text.strip()
        if own_text and not parts:
            parts.append(own_text)
        return "\n\n".join(parts)

    def collect_assets(self) -> PageAssets:
        # Avoid virtual getter recursion by reading local asset fields directly.
        local_assets = PageAssets(
            headers=list(self.assets.headers),
            footers=list(self.assets.footers),
            annotations=list(self.assets.annotations),
            citations=list(self.assets.citations),
            page_numbers=list(self.assets.page_numbers),
            images=list(self.assets.images),
        )
        child_assets = [item.collect_assets() for item in self.SubContent]
        return local_assets.merged(child_assets)

    def get_headers(self) -> List[str]:
        return self.collect_assets().headers

    def get_footers(self) -> List[str]:
        return self.collect_assets().footers

    def get_annotations(self) -> List[str]:
        return self.collect_assets().annotations

    def get_citations(self) -> List[str]:
        return self.collect_assets().citations

    def get_page_numbers(self) -> List[str]:
        return self.collect_assets().page_numbers

    def get_images(self) -> List[Any]:
        return self.collect_assets().images

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload["markdown_text"] = self.merged_markdown()
        payload["SubContent"] = [item.to_payload() for item in self.SubContent]
        return payload


class RAG_DB_Document(Chapter, ABC):
    page_nodes: List[Page]
    default_chunk_chars = 1200
    default_chunk_overlap = 180

    def __init__(self, source_document: Document, stable_doc_id: str) -> None:
        metadata = dict(source_document.metadata or {})
        base_doc_id = source_document.doc_id or stable_doc_id
        doc_name = self._resolve_doc_name(str(base_doc_id), metadata)
        title = os.path.basename(doc_name) or doc_name or "Document"
        super().__init__(title=title, metadata=metadata, SubContent=[])

        self.source_document = source_document
        self.base_doc_id = base_doc_id
        self.doc_name = self._normalize_doc_path(str(doc_name))
        self.source_extension = self._detect_source_extension(metadata)
        self.cleaned_text = ""
        self.chunk_documents: List[Document] = []
        self.catalog: List[Any] = []
        self.page_count = 0
        self.pagination_mode = "page-tree"

    def set_build_trace(self, **kwargs: Any) -> None:
        trace = dict(self.metadata.get("_doc_tree_build_trace") or {})
        for key, value in kwargs.items():
            trace[str(key)] = value
        self.metadata["_doc_tree_build_trace"] = trace

    def get_build_trace(self) -> Dict[str, Any]:
        return dict(self.metadata.get("_doc_tree_build_trace") or {})

    @abstractmethod
    def build(self) -> "RAG_DB_Document":
        ...

    def resolve_page_map(
        self,
        lines: Sequence[str],
        native_page_map: Sequence[int],
        has_native_marker: bool,
    ) -> tuple[List[int], str]:
        return [idx + 1 for idx, _ in enumerate(lines)], "page-tree"

    def allow_heading_detection(self) -> bool:
        return False

    @staticmethod
    def _cn_numeral_to_int(token: str) -> Optional[int]:
        text = str(token or "").strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)

        digits = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        units = {"十": 10, "百": 100, "千": 1000, "万": 10000}

        total = 0
        section = 0
        number = 0
        has_token = False
        for ch in text:
            if ch in digits:
                number = digits[ch]
                has_token = True
                continue
            unit = units.get(ch)
            if unit is None:
                return None
            has_token = True
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        value = total + section + number
        if not has_token:
            return None
        return value if value > 0 else None

    @classmethod
    def _top_level_key_from_title(cls, title: str) -> Optional[tuple[str, str]]:
        text = str(title or "").strip()
        if not text:
            return None

        chapter = re.search(r"第\s*([一二三四五六七八九十百千万两0-9]+)\s*章", text)
        if chapter:
            number = cls._cn_numeral_to_int(chapter.group(1))
            if number is not None:
                return ("chapter", str(number))

        appendix = re.search(r"附录\s*([一二三四五六七八九十百千万两A-Za-z0-9]+)", text)
        if appendix:
            token = str(appendix.group(1) or "").strip()
            number = cls._cn_numeral_to_int(token)
            if number is not None:
                return ("appendix", str(number))
            return ("appendix", token.lower())

        return None

    @classmethod
    def _top_level_key_from_page_text(cls, page_text: str) -> Optional[tuple[str, str]]:
        lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
        for line in lines[:12]:
            chapter = re.search(r"第\s*([一二三四五六七八九十百千万两0-9]+)\s*章", line)
            if chapter:
                number = cls._cn_numeral_to_int(chapter.group(1))
                if number is not None:
                    return ("chapter", str(number))

            appendix = re.search(r"附录\s*([一二三四五六七八九十百千万两A-Za-z0-9]+)", line)
            if appendix:
                token = str(appendix.group(1) or "").strip()
                number = cls._cn_numeral_to_int(token)
                if number is not None:
                    return ("appendix", str(number))
                return ("appendix", token.lower())

            sec = re.match(r"^(\d{1,2})\s*[\.、．]\s*", line)
            if sec:
                num = int(sec.group(1))
                if 1 <= num <= 50:
                    return ("chapter", str(num))
        return None

    def extend_top_level_ranges_with_page_signals(
        self,
        page_texts: Sequence[str],
        ranges: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        base_ranges = [dict(item) for item in list(ranges or [])]
        if not base_ranges or not page_texts:
            return base_ranges

        top_level = [item for item in base_ranges if int(self._coerce_positive_int(item.get("level")) or 1) == 1]
        if not top_level:
            return base_ranges

        title_by_key: Dict[tuple[str, str], str] = {}
        for item in top_level:
            title = str(item.get("title") or "").strip()
            key = self._top_level_key_from_title(title)
            if key is not None and key not in title_by_key:
                title_by_key[key] = title

        if not title_by_key:
            return base_ranges

        page_to_title: Dict[int, str] = {}
        current_title = ""
        for idx, page_text in enumerate(page_texts, start=1):
            key = self._top_level_key_from_page_text(page_text)
            if key is not None and key in title_by_key:
                current_title = title_by_key[key]
            if current_title:
                page_to_title[idx] = current_title

        if not page_to_title:
            return base_ranges

        spans: Dict[str, tuple[int, int]] = {}
        for page_idx in sorted(page_to_title.keys()):
            title = page_to_title[page_idx]
            old = spans.get(title)
            if old is None:
                spans[title] = (page_idx, page_idx)
            else:
                spans[title] = (min(old[0], page_idx), max(old[1], page_idx))

        for item in base_ranges:
            level = int(self._coerce_positive_int(item.get("level")) or 1)
            if level != 1:
                continue
            title = str(item.get("title") or "").strip()
            span = spans.get(title)
            if span is None:
                continue
            start, end = span
            old_start = int(self._coerce_positive_int(item.get("start")) or start)
            old_end = int(self._coerce_positive_int(item.get("end")) or old_start)
            item["start"] = min(old_start, start)
            item["end"] = max(old_end, end)

        return base_ranges

    def enforce_native_page_count(self, page_texts: Sequence[str]) -> List[str]:
        pages = [str(item or "").strip() for item in list(page_texts)]
        if not pages:
            pages = [""]

        target = self.coerce_page_number(self.metadata.get("native_page_count"))
        if target is None or target <= 0:
            return pages

        target = int(target)
        if len(pages) == target:
            return pages

        if target == 1:
            merged = "\n".join(part for part in pages if part).strip()
            return [merged]

        merged_lines: List[str] = []
        for part in pages:
            text = str(part or "").strip()
            if not text:
                merged_lines.append("")
                continue
            lines = text.splitlines()
            if lines:
                merged_lines.extend(lines)
            else:
                merged_lines.append(text)

        if not merged_lines:
            return [""] * target

        total = len(merged_lines)
        normalized_pages: List[str] = []
        for idx in range(target):
            start = int(round(idx * total / target))
            end = int(round((idx + 1) * total / target))
            if start >= total:
                chunk = ""
            else:
                chunk = "\n".join(merged_lines[start:end]).strip()
            normalized_pages.append(chunk)

        if normalized_pages and not normalized_pages[0].strip():
            normalized_pages[0] = "\n".join(line for line in merged_lines[: max(1, total // target)]).strip()
        return normalized_pages

    def _split_text_chunks(self, text: str) -> List[str]:
        normalized = str(text or "").strip()
        if not normalized:
            return []

        chunk_chars = self.default_chunk_chars
        overlap_chars = self.default_chunk_overlap
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

    @staticmethod
    def _parse_page_number_hint(line: str) -> Optional[int]:
        text = str(line or "").strip()
        if not text:
            return None

        explicit_patterns = [
            r"(?:第\s*)(\d{1,4})(?:\s*页)",
            r"(?:page\s*)(\d{1,4})",
            r"(?:p\.?\s*)(\d{1,4})",
        ]
        for pat in explicit_patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = int(match.group(1))
            if 1 <= value <= 5000:
                return value

        compact = re.sub(r"\s+", "", text)
        if len(compact) <= 6 and re.fullmatch(r"\d{1,4}", compact):
            value = int(compact)
            if 1 <= value <= 5000:
                return value
        roman = RAG_DB_Document._roman_numeral_to_int(compact)
        if roman is not None and 1 <= roman <= 5000:
            return roman
        return None

    @staticmethod
    def _median_int(values: Sequence[int]) -> int:
        seq = sorted(int(v) for v in values)
        if not seq:
            return 0
        mid = len(seq) // 2
        if len(seq) % 2 == 1:
            return seq[mid]
        return int(round((seq[mid - 1] + seq[mid]) / 2.0))

    def build_page_signals(self, page_texts: Sequence[str], ranges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pages = [str(text or "") for text in list(page_texts)]
        if not pages:
            return []

        line_sets: List[List[str]] = []
        first_lines: List[str] = []
        last_lines: List[str] = []
        for text in pages:
            lines = [line.strip() for line in str(text).splitlines() if line.strip()]
            line_sets.append(lines)
            first_lines.append(lines[0] if lines else "")
            last_lines.append(lines[-1] if lines else "")

        head_freq: Dict[str, int] = {}
        tail_freq: Dict[str, int] = {}
        for value in first_lines:
            if value and len(value) <= 120:
                head_freq[value] = head_freq.get(value, 0) + 1
        for value in last_lines:
            if value and len(value) <= 120:
                tail_freq[value] = tail_freq.get(value, 0) + 1

        signals: List[Dict[str, Any]] = []
        hint_anchors: List[tuple[int, int]] = []
        for idx, lines in enumerate(line_sets, start=1):
            header_text = ""
            footer_text = ""
            first_line = lines[0] if lines else ""
            last_line = lines[-1] if lines else ""
            if lines:
                if head_freq.get(first_line, 0) >= 2:
                    header_text = first_line
                if tail_freq.get(last_line, 0) >= 2:
                    footer_text = last_line

            candidates: List[str] = []
            if footer_text:
                candidates.append(footer_text)
            if header_text:
                candidates.append(header_text)
            # Strict policy: only inspect likely header/footer lines, never正文区间。
            if not footer_text and last_line and len(last_line) <= 24:
                candidates.append(last_line)
            if not header_text and first_line and len(first_line) <= 24:
                candidates.append(first_line)

            hint_value: Optional[int] = None
            for candidate in candidates:
                hint = self._parse_page_number_hint(candidate)
                if hint is not None:
                    hint_value = hint
                    break

            if hint_value is not None:
                hint_anchors.append((idx, int(hint_value)))

            signals.append(
                {
                    "header_text": header_text,
                    "footer_text": footer_text,
                    "page_number_hint": str(hint_value) if hint_value is not None else "",
                    "hint_page_number": hint_value,
                    "logical_page_number": idx,
                }
            )

        offsets: List[int] = [hint - physical for physical, hint in hint_anchors]
        offset = 0
        if offsets and len(offsets) >= 5:
            min_off = min(offsets)
            max_off = max(offsets)
            if max_off - min_off <= 6:
                offset = self._median_int(offsets)

        for idx, item in enumerate(signals, start=1):
            hint = item.get("hint_page_number")
            logical = int(hint) if hint is not None else int(idx + offset)
            if logical <= 0:
                logical = idx
            item["logical_page_number"] = logical

        return signals

    @staticmethod
    def _clean_heading_title(line: str) -> str:
        text = str(line or "").strip()
        text = re.sub(r"^\s*#{1,6}\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?:\t+|[·•.]{2,}|\s{2,})(?:\d{1,4}|[IVXLCDMivxlcdm]{1,8})\s*$", "", text)
        return text.strip()

    @staticmethod
    def _normalized_heading_text(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    @staticmethod
    def _is_noise_heading_line(line: str) -> bool:
        text = re.sub(r"\s+", "", str(line or "").strip().lower())
        if not text:
            return True
        noise = {"目录", "目錄", "contents", "tableofcontents", "toc"}
        return text in noise

    @staticmethod
    def _is_catalogue_keyword(line: str) -> bool:
        text = re.sub(r"\s+", "", str(line or "").strip().lower())
        if not text:
            return False
        return text in {"目录", "目錄", "contents", "tableofcontents", "toc"}

    @classmethod
    def _parse_toc_entry_line(cls, line: str) -> Optional[Dict[str, Any]]:
        text = str(line or "").strip()
        if not text:
            return None

        normalized = text.strip("|").strip()
        patterns = [
            r"^(.{1,200}?)(?:\s*\|\s*|\t+|[·•.]{2,}|\s{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*\|?\s*$",
            r"^(.{1,200}?)\s+(\d{1,4}|[IVXLCDM]{1,8})\s*$",
        ]
        match = None
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                break
        if not match:
            return None

        title = cls._clean_heading_title(match.group(1))
        page = cls.coerce_page_number(match.group(2))
        if not title or page is None or cls._is_noise_heading_line(title):
            return None

        level = cls._heading_level(title) or 1
        return {
            "title": title,
            "page": int(page),
            "level": max(1, min(int(level), 6)),
        }

    @classmethod
    def _looks_like_toc_entry_line(cls, line: str) -> bool:
        row = cls._parse_toc_entry_line(line)
        if row is None:
            return False
        title = str(row.get("title") or "")
        if cls._heading_level(title) is not None:
            return True
        return bool(re.match(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节部分篇]|附录|\d+(?:\.\d+){0,5})", title))

    def _looks_like_catalogue_page(self, page_text: str) -> bool:
        lines = [line.strip() for line in str(page_text or "").splitlines() if line and line.strip()]
        if not lines:
            return False

        first_line = lines[0]
        toc_hits = sum(1 for line in lines[:48] if self._parse_toc_entry_line(line) is not None)
        if self._is_catalogue_keyword(first_line):
            return True
        if toc_hits >= 4:
            return True
        return toc_hits >= 2 and toc_hits * 2 >= min(len(lines), 12)

    @classmethod
    def _heading_level(cls, line: str) -> Optional[int]:
        text = str(line or "").strip()
        if not text or cls._is_noise_heading_line(text):
            return None
        markdown = re.match(r"^(#{1,6})\s+", text)
        if markdown:
            return len(markdown.group(1))
        if re.match(r"^(chapter|part|section)\s+[ivxlcdm\d]+\b", text, re.IGNORECASE):
            return 1
        if re.match(r"^\d+(?:\.\d+){0,5}\s+", text):
            return min(text.count(".") + 1, 6)
        compact_numbered = re.match(r"^(\d+(?:\.\d+){0,5})\s*[\u4e00-\u9fffA-Za-z（(]", text)
        if compact_numbered:
            return min(compact_numbered.group(1).count(".") + 1, 6)
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇]", text):
            return 1
        if re.match(r"^附录[一二三四五六七八九十百千万0-9A-Za-z（(]", text):
            return 1
        return None

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        number = RAG_DB_Document.coerce_page_number(value)
        if number is None or number <= 0:
            return None
        return number

    def _extract_markers_from_metadata(self, keys: Sequence[str]) -> List[Dict[str, Any]]:
        markers: List[Dict[str, Any]] = []
        for key in keys:
            raw = self.metadata.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                title = self._clean_heading_title(str(item.get("title") or ""))
                page = self._coerce_positive_int(item.get("page"))
                if not title or page is None:
                    continue
                level = self._coerce_positive_int(item.get("level")) or self._heading_level(title) or 1
                markers.append(
                    {
                        "title": title,
                        "page": page,
                        "level": max(1, min(level, 6)),
                        "kind": key,
                        "order": len(markers),
                    }
                )
        return markers

    def _extract_markers_from_manual_toc(self, page_texts: Sequence[str]) -> List[Dict[str, Any]]:
        lines: List[str] = []
        for page_text in page_texts[:12]:
            lines.extend(str(page_text or "").splitlines())

        markers: List[Dict[str, Any]] = []
        for raw in lines:
            row = self._parse_toc_entry_line(raw)
            if row is None:
                continue
            markers.append(
                {
                    "title": str(row.get("title") or ""),
                    "page": int(row.get("page") or 1),
                    "level": int(row.get("level") or 1),
                    "kind": "manual-toc",
                }
            )
        markers.sort(key=lambda item: (int(item["page"]), int(item["level"])))
        return markers

    def _extract_markers_from_text_pattern(self, page_texts: Sequence[str]) -> List[Dict[str, Any]]:
        markers: List[Dict[str, Any]] = []
        for page_idx, page_text in enumerate(page_texts, start=1):
            first_lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
            if not first_lines:
                continue
            seen_titles: Set[str] = set()
            for line in first_lines[:24]:
                if self._looks_like_toc_entry_line(line):
                    continue
                level = self._heading_level(line)
                if level is None:
                    continue
                title = self._clean_heading_title(line)
                if not title or self._is_noise_heading_line(title):
                    continue
                key = self._normalized_heading_text(title)
                if not key or key in seen_titles:
                    continue
                seen_titles.add(key)
                markers.append(
                    {
                        "title": title,
                        "page": page_idx,
                        "level": max(1, min(level, 6)),
                        "kind": "text-pattern",
                    }
                )
        markers.sort(key=lambda item: (int(item["page"]), int(item["level"])))
        return markers

    def _remap_markers_with_text_hits(
        self,
        markers: Sequence[Dict[str, Any]],
        page_texts: Sequence[str],
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        if not markers or not page_texts:
            return list(markers)

        text_hits = list(self._extract_markers_from_text_pattern(page_texts) or [])
        if not text_hits:
            return list(markers)

        normalized_hits: List[tuple[str, int]] = []
        for item in text_hits:
            key = self._normalized_heading_text(self._clean_heading_title(str(item.get("title") or "")))
            page = self._coerce_positive_int(item.get("page"))
            if not key or page is None:
                continue
            normalized_hits.append((key, int(page)))

        if not normalized_hits:
            return list(markers)

        top_level_bounds: Dict[int, tuple[int, int]] = {}
        top_level_indices = [
            (idx, self._coerce_positive_int(item.get("page")) or 1)
            for idx, item in enumerate(markers)
            if (self._coerce_positive_int(item.get("level")) or 1) == 1
        ]
        for pos, (idx, start) in enumerate(top_level_indices):
            next_start = top_level_indices[pos + 1][1] if pos + 1 < len(top_level_indices) else int(total_pages)
            top_level_bounds[idx] = (
                int(start),
                max(int(start), int(next_start) - 1 if pos + 1 < len(top_level_indices) else int(total_pages)),
            )

        remapped: List[Dict[str, Any]] = []
        current_top_span = (1, int(total_pages))
        for idx, item in enumerate(markers):
            row = dict(item)
            level = self._coerce_positive_int(row.get("level")) or 1
            if level == 1 and idx in top_level_bounds:
                current_top_span = top_level_bounds[idx]

            key = self._normalized_heading_text(self._clean_heading_title(str(row.get("title") or "")))
            if not key:
                remapped.append(row)
                continue

            best_hit: Optional[int] = None
            for hit_key, hit_page in normalized_hits:
                if hit_key != key:
                    continue
                if not (int(current_top_span[0]) <= int(hit_page) <= int(current_top_span[1])):
                    continue
                if best_hit is None or int(hit_page) < int(best_hit):
                    best_hit = int(hit_page)

            if best_hit is not None:
                row["page"] = int(best_hit)
            remapped.append(row)
        return remapped

    def _prune_unmatched_tail_markers(
        self,
        markers: Sequence[Dict[str, Any]],
        page_texts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        rows = [dict(item) for item in list(markers or [])]
        if not rows or not page_texts:
            return rows

        text_hits = list(self._extract_markers_from_text_pattern(page_texts) or [])
        if not text_hits:
            return rows

        hit_keys: Set[str] = set()
        for item in text_hits:
            key = self._normalized_heading_text(self._clean_heading_title(str(item.get("title") or "")))
            if key:
                hit_keys.add(key)
        if not hit_keys:
            return rows

        top_level_positions = [
            idx
            for idx, item in enumerate(rows)
            if (self._coerce_positive_int(item.get("level")) or 1) == 1
        ]
        if len(top_level_positions) < 2:
            return rows

        supported_top_positions: List[int] = []
        for pos, start_idx in enumerate(top_level_positions):
            end_idx = top_level_positions[pos + 1] if pos + 1 < len(top_level_positions) else len(rows)
            segment_supported = False
            for item in rows[start_idx:end_idx]:
                key = self._normalized_heading_text(self._clean_heading_title(str(item.get("title") or "")))
                if key and key in hit_keys:
                    segment_supported = True
                    break
            if segment_supported:
                supported_top_positions.append(pos)

        if len(supported_top_positions) < 2:
            return rows

        last_supported_pos = supported_top_positions[-1]
        unsupported_tail_count = len(top_level_positions) - (last_supported_pos + 1)
        if unsupported_tail_count < 2:
            return rows

        cutoff = top_level_positions[last_supported_pos + 1]
        return rows[:cutoff]

    def extract_multilevel_catalog_markers(
        self,
        page_texts: Sequence[str],
        *,
        metadata_keys: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        keys = list(metadata_keys or ["native_catalog", "style_catalog", "font_catalog"])
        layered: List[List[Dict[str, Any]]] = [
            self._extract_markers_from_metadata(keys),
            self._extract_markers_from_manual_toc(page_texts),
            self._extract_markers_from_text_pattern(page_texts),
        ]
        for markers in layered:
            if markers:
                return markers
        return []

    @staticmethod
    def _next_boundary_start_page(markers: Sequence[Dict[str, Any]], index: int) -> Optional[int]:
        if index < 0 or index >= len(markers):
            return None
        current_level = RAG_DB_Document.coerce_page_number(markers[index].get("level")) or 1
        for idx in range(index + 1, len(markers)):
            candidate_level = RAG_DB_Document.coerce_page_number(markers[idx].get("level")) or current_level
            if candidate_level > current_level:
                continue
            candidate_page = RAG_DB_Document.coerce_page_number(markers[idx].get("page"))
            if candidate_page is None:
                continue
            return candidate_page
        return None

    def derive_catalog_ranges(self, markers: Sequence[Dict[str, Any]], total_pages: int) -> List[Dict[str, Any]]:
        if not markers:
            return []
        ranges: List[Dict[str, Any]] = []
        for idx, marker in enumerate(markers):
            title = self._clean_heading_title(str(marker.get("title") or ""))
            start_page = self._coerce_positive_int(marker.get("page"))
            if not title or start_page is None:
                continue
            level = self._coerce_positive_int(marker.get("level")) or 1
            next_start = self._next_boundary_start_page(markers, idx)
            if next_start is None:
                end_page = max(start_page, int(total_pages or start_page))
            else:
                end_page = max(start_page, next_start - 1)
            ranges.append(
                {
                    "title": title,
                    "start": start_page,
                    "end": end_page,
                    "level": max(1, min(level, 6)),
                }
            )
        return ranges

    @staticmethod
    def _pick_catalog_section(page_idx: int, ranges: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates: List[tuple[int, int, int, int, Dict[str, Any]]] = []
        for order, item in enumerate(ranges):
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", 0) or 0)
            if not (start <= page_idx <= end):
                continue
            level = int(RAG_DB_Document._coerce_positive_int(item.get("level")) or 1)
            span = max(1, end - start + 1)
            catalog_order = int(item.get("order", order) or order)
            candidates.append((-level, span, start, catalog_order, dict(item)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return candidates[0][4]

    def _build_target_page_start_marker_map(
        self,
        ranges: Sequence[Dict[str, Any]],
        source_page_map: Sequence[int],
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not ranges:
            return {}

        target_by_source: Dict[int, int] = {}
        source_pages = [int(self._coerce_positive_int(value) or (idx + 1)) for idx, value in enumerate(source_page_map)]
        for target_page, source_page in enumerate(source_pages, start=1):
            target_by_source.setdefault(int(source_page), int(target_page))

        markers_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for order, item in enumerate(ranges):
            source_start = int(self._coerce_positive_int(item.get("start")) or 0)
            if source_start <= 0:
                continue
            target_page = target_by_source.get(source_start)
            if target_page is None:
                for idx, source_page in enumerate(source_pages, start=1):
                    if int(source_page) >= source_start:
                        target_page = idx
                        break
            if target_page is None:
                target_page = source_start
            row = dict(item)
            row.setdefault("order", order)
            markers_by_page.setdefault(int(target_page), []).append(row)

        for items in markers_by_page.values():
            items.sort(
                key=lambda row: (
                    int(self._coerce_positive_int(row.get("level")) or 1),
                    int(row.get("order", 0) or 0),
                    str(row.get("title") or ""),
                )
            )
        return markers_by_page

    @staticmethod
    def _normalize_list_markdown_line(line: str) -> str:
        text = str(line or "").rstrip()
        if not text.strip():
            return ""

        bullet = re.match(r"^\s*[•●▪■◆◇·]\s*(.+)$", text)
        if bullet:
            return f"- {bullet.group(1).strip()}"

        return text

    @classmethod
    def _extract_figure_captions(cls, text: str) -> List[str]:
        captions: List[str] = []
        seen: Set[str] = set()
        for raw in str(text or "").splitlines():
            line = str(raw or "").strip()
            if not line or len(line) > 160:
                continue
            if not re.match(r"^(?:图|figure)\s*[A-Za-z0-9一二三四五六七八九十百千万\-‐‑–—.]*\s*[:：]?\s*.+$", line, flags=re.IGNORECASE):
                continue
            key = cls._normalized_heading_text(line)
            if not key or key in seen:
                continue
            seen.add(key)
            captions.append(line)
        return captions

    def resolve_page_images(self, page_text: str, existing_images: Optional[Sequence[Any]] = None) -> List[Any]:
        resolved = _dedupe_image_values(list(existing_images or []))

        if any(self._looks_like_real_image_asset(item) for item in resolved):
            return resolved

        for caption in self._extract_figure_captions(page_text):
            marker = f"figure: {caption}"
            resolved.append(marker)
        return _dedupe_image_values(resolved)

    @staticmethod
    def _looks_like_real_image_asset(value: Any) -> bool:
        normalized = _normalize_image_asset(value)
        if normalized is None:
            return False
        if isinstance(normalized, ImageAsset):
            return bool(normalized.has_binary or normalized.media_type or normalized.filename)
        text = str(normalized or "").strip()
        if text.startswith("data:image/"):
            return True
        return bool(re.search(r"(?:^|/|\\)([^/\\]+)\.(?:png|jpe?g|gif|bmp|webp|tiff?|svg|wmf|emf)$", text, flags=re.IGNORECASE))

    def get_page_layouts(self) -> List[Dict[str, Any]]:
        raw_layouts = self.metadata.get("page_layout")
        if not isinstance(raw_layouts, list) or not raw_layouts:
            raw_layouts = self.metadata.get("structured_page_assets")
        if not isinstance(raw_layouts, list):
            return []

        normalized_layouts: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_layouts, start=1):
            row = dict(item) if isinstance(item, dict) else {}
            normalized_layouts.append(
                {
                    "page": int(self._coerce_positive_int(row.get("page")) or index),
                    "headers": _dedupe_text_values(list(row.get("headers") or [])),
                    "footers": _dedupe_text_values(list(row.get("footers") or [])),
                    "page_number": str(row.get("page_number") or "").strip(),
                    "images": _dedupe_image_values(list(row.get("images") or [])),
                }
            )
        return normalized_layouts

    @staticmethod
    def _group_items_evenly(values: Sequence[Any], slots: int) -> List[List[Any]]:
        if slots <= 0:
            return []
        groups: List[List[Any]] = [[] for _ in range(slots)]
        items = [value for value in list(values or []) if _normalize_image_asset(value) is not None or str(value or "").strip()]
        for idx, item in enumerate(items):
            groups[min(slots - 1, idx % slots)].append(item)
        return groups

    def _annotate_range_paths(self, ranges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []
        ordered = sorted(
            [dict(item) for item in list(ranges or [])],
            key=lambda row: (
                int(self._coerce_positive_int(row.get("start")) or 10**9),
                int(row.get("order", 0) or 0),
                int(self._coerce_positive_int(row.get("level")) or 1),
            ),
        )
        for index, item in enumerate(ordered):
            title = self._clean_heading_title(str(item.get("title") or ""))
            if not title:
                continue
            level = max(1, min(int(self._coerce_positive_int(item.get("level")) or 1), 6))
            while stack and int(stack[-1].get("level") or 1) >= level:
                stack.pop()
            parent_path = str(stack[-1].get("section_path") or "").strip() if stack else ""
            section_path = f"{parent_path}/{title}" if parent_path else title
            row = dict(item)
            row["title"] = title
            row["level"] = level
            row["order"] = int(row.get("order", index) or index)
            row["section_path"] = section_path
            annotated.append(row)
            stack.append({"level": level, "section_path": section_path})
        return annotated

    def _find_page_marker_hits(
        self,
        lines: Sequence[str],
        markers: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_lines = [
            self._normalized_heading_text(self._clean_heading_title(line))
            for line in list(lines or [])
        ]
        hits: List[Dict[str, Any]] = []
        for order, marker in enumerate(markers):
            title = self._clean_heading_title(str(marker.get("title") or ""))
            key = self._normalized_heading_text(title)
            if not title or not key or self._is_noise_heading_line(title):
                continue
            if len(key) < 2 and self._heading_level(title) is None:
                continue
            hit_index: Optional[int] = None
            for line_index, line_key in enumerate(normalized_lines):
                if not line_key:
                    continue
                if line_key == key or line_key.startswith(key) or (len(line_key) >= 6 and key.startswith(line_key)):
                    hit_index = line_index
                    break
            if hit_index is None:
                continue
            row = dict(marker)
            row["title"] = title
            row["line_index"] = int(hit_index)
            row["order"] = int(row.get("order", order) or order)
            hits.append(row)

        deduped_by_line: Dict[int, Dict[str, Any]] = {}
        for item in sorted(
            hits,
            key=lambda row: (
                int(row.get("line_index", 10**9)),
                -len(self._normalized_heading_text(str(row.get("title") or ""))),
                -int(self._coerce_positive_int(row.get("level")) or 1),
                int(row.get("order", 0) or 0),
            ),
        ):
            line_index = int(item.get("line_index", 10**9))
            deduped_by_line.setdefault(line_index, item)
        return [deduped_by_line[key] for key in sorted(deduped_by_line.keys())]

    def _split_monopage_by_markers(
        self,
        page: MonoPage,
        markers: Sequence[Dict[str, Any]],
    ) -> List[MonoPage]:
        if not isinstance(page, MonoPage) or page.category != PageType.CONTENT:
            return [page]

        page_meta = dict(getattr(page, "metadata", {}) or {})
        physical_page = int(self.coerce_page_number(page_meta.get("page")) or 0)
        if physical_page <= 0:
            return [page]

        raw_text = str(page_meta.get("raw_page_text") or page.markdown_text or "")
        lines = [str(line or "") for line in raw_text.splitlines()]
        if not lines:
            return [page]

        page_markers = [
            dict(item)
            for item in list(markers or [])
            if int(self._coerce_positive_int(item.get("start")) or 0) == physical_page
        ]
        if not page_markers:
            return [page]

        hits = self._find_page_marker_hits(lines, page_markers)
        if not hits:
            page.set_markdown(
                self._render_plaintext_page_to_markdown(
                    raw_text,
                    start_markers=page_markers,
                    page_number=physical_page,
                )
            )
            page.metadata.setdefault("physical_page", physical_page)
            page.metadata["fragment_index"] = 1
            page.metadata["fragment_count"] = 1
            return [page]

        if len(hits) == 1 and int(hits[0].get("line_index", 0) or 0) <= 0:
            start_marker = dict(hits[0])
            page.set_markdown(
                self._render_plaintext_page_to_markdown(
                    raw_text,
                    start_markers=[start_marker],
                    page_number=physical_page,
                )
            )
            page.metadata.setdefault("physical_page", physical_page)
            page.metadata["fragment_index"] = 1
            page.metadata["fragment_count"] = 1
            return [page]

        segments: List[Dict[str, Any]] = []
        current_title = str(page_meta.get("section_title") or "").strip()
        current_path = str(page_meta.get("section_path") or current_title).strip()
        first_hit_index = int(hits[0].get("line_index", 0) or 0)
        if first_hit_index > 0:
            prefix_text = "\n".join(lines[:first_hit_index]).strip()
            if prefix_text:
                segments.append(
                    {
                        "text": prefix_text,
                        "section_title": current_title,
                        "section_path": current_path,
                        "marker": None,
                    }
                )

        for index, hit in enumerate(hits):
            start = int(hit.get("line_index", 0) or 0)
            next_start = int(hits[index + 1].get("line_index", len(lines)) or len(lines)) if index + 1 < len(hits) else len(lines)
            fragment_text = "\n".join(lines[start:next_start]).strip()
            if not fragment_text:
                continue
            section_title = str(hit.get("title") or current_title).strip()
            section_path = str(hit.get("section_path") or section_title).strip() or current_path
            segments.append(
                {
                    "text": fragment_text,
                    "section_title": section_title,
                    "section_path": section_path,
                    "marker": dict(hit),
                }
            )

        if len(segments) <= 1:
            return [page]

        page_images = list(page.get_images() or [])
        caption_target_indexes = [
            idx
            for idx, item in enumerate(segments)
            if self._extract_figure_captions(str(item.get("text") or ""))
        ]
        image_target_indexes = caption_target_indexes or list(range(len(segments)))
        image_groups = self._group_items_evenly(page_images, len(image_target_indexes)) if image_target_indexes else []

        fragments: List[MonoPage] = []
        fragment_count = len(segments)
        for index, segment in enumerate(segments, start=1):
            fragment_meta = dict(page_meta)
            fragment_meta["physical_page"] = physical_page
            fragment_meta["page"] = physical_page
            fragment_meta["section_start_page"] = physical_page
            fragment_meta["section_end_page"] = physical_page
            fragment_meta["fragment_index"] = index
            fragment_meta["fragment_count"] = fragment_count
            fragment_meta["section_title"] = str(segment.get("section_title") or current_title).strip()
            fragment_meta["section_path"] = str(segment.get("section_path") or current_path).strip()
            fragment_meta["resolved_section_path"] = fragment_meta["section_path"]
            fragment_meta["section_resolver"] = "intra_page_split"
            fragment_meta["raw_page_text"] = str(segment.get("text") or "")

            assets = PageAssets(
                headers=list(page.assets.headers) if index == 1 else [],
                footers=list(page.assets.footers) if index == fragment_count else [],
                annotations=list(page.assets.annotations) if index == 1 else [],
                citations=list(page.assets.citations) if index == 1 else [],
                page_numbers=list(page.assets.page_numbers) if index == fragment_count else [],
                images=[],
            )
            if image_target_indexes:
                for target_offset, target_index in enumerate(image_target_indexes):
                    if target_index != index - 1:
                        continue
                    assets.images.extend(image_groups[target_offset])

            fragment_text = str(segment.get("text") or "")
            marker = segment.get("marker")
            fragment_markdown = self._render_plaintext_page_to_markdown(
                fragment_text,
                start_markers=[dict(marker)] if isinstance(marker, dict) else None,
                page_number=physical_page,
            )
            fragments.append(
                SemiPage(
                    markdown_text=fragment_markdown,
                    assets=assets,
                    metadata=fragment_meta,
                )
            )

        return fragments or [page]

    def split_mono_pages_by_section_markers(
        self,
        pages: Sequence[MonoPage],
        ranges: Sequence[Dict[str, Any]],
    ) -> List[MonoPage]:
        annotated_ranges = self._annotate_range_paths(ranges)
        split_pages: List[MonoPage] = []
        for page in list(pages or []):
            split_pages.extend(self._split_monopage_by_markers(page, annotated_ranges))
        return split_pages

    def _render_plaintext_page_to_markdown(
        self,
        text: str,
        *,
        start_markers: Optional[Sequence[Dict[str, Any]]] = None,
        page_number: Optional[int] = None,
    ) -> str:
        markers = [dict(item) for item in list(start_markers or []) if str(item.get("title") or "").strip()]
        figure_captions = self._extract_figure_captions(text)

        def _marker_key(item: Dict[str, Any]) -> str:
            return self._normalized_heading_text(self._clean_heading_title(str(item.get("title") or "")))

        marker_keys = {_marker_key(item): item for item in markers if _marker_key(item)}
        figure_keys = {
            self._normalized_heading_text(caption): (idx, caption)
            for idx, caption in enumerate(figure_captions, start=1)
            if self._normalized_heading_text(caption)
        }
        consumed: Set[str] = set()
        normalized_lines: List[str] = []
        for raw in str(text or "").splitlines():
            line = self._normalize_list_markdown_line(raw)
            line_key = self._normalized_heading_text(self._clean_heading_title(line))
            marker = marker_keys.get(line_key)
            if marker is not None and line_key not in consumed:
                level = max(1, min(int(self._coerce_positive_int(marker.get("level")) or 1), 6))
                normalized_lines.append(f"{'#' * level} {str(marker.get('title') or '').strip()}")
                consumed.add(line_key)
                continue
            normalized_lines.append(line)

        prepend: List[str] = []
        for item in markers:
            key = _marker_key(item)
            if not key or key in consumed:
                continue
            level = max(1, min(int(self._coerce_positive_int(item.get("level")) or 1), 6))
            prepend.append(f"{'#' * level} {str(item.get('title') or '').strip()}")

        lines = prepend + normalized_lines if prepend else normalized_lines
        out: List[str] = []
        emitted_figures: Set[str] = set()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = str(line or "").strip()
            figure_key = self._normalized_heading_text(stripped_line)
            figure_info = figure_keys.get(figure_key)
            if figure_info is not None and figure_key not in emitted_figures:
                figure_idx, caption = figure_info
                page_token = int(page_number or 0) if page_number else 0
                out.append(f"![{caption}](image://page-{page_token}/{figure_idx})")
                emitted_figures.add(figure_key)

            cells = [cell.strip() for cell in str(line).split("\t") if cell.strip()]
            if len(cells) >= 2:
                block_lines: List[str] = [str(line)]
                table_rows: List[List[str]] = [cells]
                probe_idx = i + 1
                while probe_idx < len(lines):
                    next_line = str(lines[probe_idx])
                    next_cells = [cell.strip() for cell in next_line.split("\t") if cell.strip()]
                    if len(next_cells) >= 2:
                        block_lines.append(next_line)
                        table_rows.append(next_cells)
                        probe_idx += 1
                    else:
                        break

                toc_like_count = sum(
                    1
                    for row in block_lines
                    if self._looks_like_toc_entry_line(row) or self._is_catalogue_keyword(row)
                )
                if toc_like_count >= max(1, len(block_lines) - 1):
                    for row in block_lines:
                        out.append(re.sub(r"\t+", "    ", str(row).rstrip()))
                    i = probe_idx
                    continue

                width = max(len(row) for row in table_rows)
                norm = [row + [""] * (width - len(row)) for row in table_rows]
                header = "| " + " | ".join(cell.replace("|", "\\|") for cell in norm[0]) + " |"
                sep = "| " + " | ".join(["---"] * width) + " |"
                out.append(header)
                out.append(sep)
                for row in norm[1:]:
                    out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
                i = probe_idx
                continue

            out.append(line)
            i += 1
        return "\n".join(out)

    @staticmethod
    def _is_markdown_table_block(block: str) -> bool:
        lines = [line.rstrip() for line in str(block or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        if not lines[0].lstrip().startswith("|"):
            return False
        return lines[1].lstrip().startswith("|") and "---" in lines[1]

    @classmethod
    def _split_markdown_table_block(cls, block: str, slots: int) -> List[str]:
        lines = [line.rstrip() for line in str(block or "").splitlines() if line.strip()]
        if len(lines) < 3 or slots <= 1:
            return [str(block or "").strip()]
        header = lines[:2]
        rows = lines[2:]
        if not rows:
            return ["\n".join(header)]
        parts: List[str] = []
        total_rows = len(rows)
        chunk_count = max(1, min(int(slots), total_rows))
        for idx in range(chunk_count):
            start = int(round(idx * total_rows / chunk_count))
            end = int(round((idx + 1) * total_rows / chunk_count))
            chunk_rows = rows[start:end]
            if not chunk_rows:
                continue
            parts.append("\n".join(header + chunk_rows).strip())
        return parts or [str(block or "").strip()]

    @staticmethod
    def _split_markdown_text_block(block: str, slots: int) -> List[str]:
        text = str(block or "").strip()
        if not text or slots <= 1:
            return [text]

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        if len(paragraphs) >= 2:
            mid = max(1, len(paragraphs) // 2)
            return ["\n\n".join(paragraphs[:mid]).strip(), "\n\n".join(paragraphs[mid:]).strip()]

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 4:
            mid = max(1, len(lines) // 2)
            return ["\n".join(lines[:mid]).strip(), "\n".join(lines[mid:]).strip()]

        if len(text) < 480:
            return [text]

        split_at = len(text) // 2
        window = text[max(0, split_at - 120): min(len(text), split_at + 120)]
        match = re.search(r"[。！？!?]\s*", window)
        if match:
            split_at = max(1, max(0, split_at - 120) + match.end())
        return [text[:split_at].strip(), text[split_at:].strip()]

    @classmethod
    def _split_markdown_blocks_for_slots(cls, blocks: Sequence[str], slots: int) -> List[str]:
        cleaned_blocks = [str(item or "").strip() for item in list(blocks or []) if str(item or "").strip()]
        if slots <= 0:
            return []
        if not cleaned_blocks:
            return [""] * slots

        fragments = list(cleaned_blocks)
        while len(fragments) < slots:
            best_index = -1
            best_size = 0
            for idx, fragment in enumerate(fragments):
                size = len(fragment)
                if size > best_size:
                    best_index = idx
                    best_size = size
            if best_index < 0:
                break
            fragment = fragments[best_index]
            if cls._is_markdown_table_block(fragment):
                split_parts = cls._split_markdown_table_block(fragment, 2)
            else:
                split_parts = cls._split_markdown_text_block(fragment, 2)
            split_parts = [part for part in split_parts if str(part or "").strip()]
            if len(split_parts) <= 1:
                break
            fragments = fragments[:best_index] + split_parts + fragments[best_index + 1:]

        if len(fragments) > slots:
            merged = [""] * slots
            total = len(fragments)
            for idx, fragment in enumerate(fragments):
                slot = min(slots - 1, int(idx * slots / max(1, total)))
                merged[slot] = f"{merged[slot]}\n\n{fragment}".strip() if merged[slot] else fragment
            fragments = merged

        if len(fragments) < slots:
            fragments.extend([""] * (slots - len(fragments)))
        return [str(item or "").strip() for item in fragments[:slots]]

    @staticmethod
    def _group_values_evenly(values: Sequence[Any], slots: int) -> List[List[Any]]:
        if slots <= 0:
            return []
        groups: List[List[Any]] = [[] for _ in range(slots)]
        items: List[Any] = []
        for value in list(values or []):
            normalized = _normalize_image_asset(value)
            if normalized is not None:
                items.append(normalized)
                continue
            text = str(value or "").strip()
            if text:
                items.append(text)
        for idx, item in enumerate(items):
            groups[min(slots - 1, idx % slots)].append(item)
        return groups

    def _apply_structured_section_fallback(
        self,
        pages: Sequence[MonoPage],
        ranges: Sequence[Dict[str, Any]],
    ) -> None:
        structured_rows = list(self.metadata.get("structured_sections") or [])
        if not structured_rows or not pages or not ranges:
            return

        section_lookup: Dict[str, Dict[str, Any]] = {}
        for row in structured_rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            key = self._normalized_heading_text(self._clean_heading_title(title))
            if not key:
                continue
            blocks = [str(item or "").strip() for item in list(row.get("blocks") or []) if str(item or "").strip()]
            images = [str(item or "").strip() for item in list(row.get("images") or []) if str(item or "").strip()]
            current_score = len(blocks) + len(images)
            old = section_lookup.get(key)
            old_score = 0
            if old is not None:
                old_score = len(list(old.get("blocks") or [])) + len(list(old.get("images") or []))
            if old is None or current_score > old_score:
                section_lookup[key] = {
                    "title": title,
                    "blocks": blocks,
                    "images": images,
                }

        sorted_ranges = sorted(
            list(ranges or []),
            key=lambda row: (
                -(int(self._coerce_positive_int(row.get("level")) or 1)),
                max(1, int(self._coerce_positive_int(row.get("end")) or self._coerce_positive_int(row.get("start")) or 1) - int(self._coerce_positive_int(row.get("start")) or 1) + 1),
                int(self._coerce_positive_int(row.get("start")) or 0),
            ),
        )

        for item in sorted_ranges:
            title = str(item.get("title") or "").strip()
            key = self._normalized_heading_text(self._clean_heading_title(title))
            if not key:
                continue
            section = section_lookup.get(key)
            if section is None:
                continue

            start = int(self._coerce_positive_int(item.get("start")) or 0)
            end = int(self._coerce_positive_int(item.get("end")) or start)
            if start <= 0 or end <= 0:
                continue

            target_indexes: List[int] = []
            for idx, page in enumerate(pages):
                meta = dict(getattr(page, "metadata", {}) or {})
                source_page = self.coerce_page_number(meta.get("source_page"))
                page_no = self.coerce_page_number(meta.get("page"))
                candidate = int(source_page or page_no or 0)
                if start <= candidate <= end:
                    target_indexes.append(idx)
            if not target_indexes:
                continue

            blank_indexes = [
                idx
                for idx in target_indexes
                if not str(getattr(pages[idx], "markdown_text", "") or "").strip()
            ]

            blocks = list(section.get("blocks") or [])
            if blocks and blank_indexes:
                fragments = self._split_markdown_blocks_for_slots(blocks, len(blank_indexes))
                for page_idx, fragment in zip(blank_indexes, fragments):
                    text = str(fragment or "").strip()
                    if not text:
                        continue
                    current = str(getattr(pages[page_idx], "markdown_text", "") or "").strip()
                    if current:
                        pages[page_idx].set_markdown(f"{current}\n\n{text}".strip())
                    else:
                        pages[page_idx].set_markdown(text)

            images = [item for item in list(section.get("images") or []) if self._looks_like_real_image_asset(item)]
            if not images:
                continue
            preferred_indexes = [
                idx
                for idx in target_indexes
                if re.search(r"image://page-|^(?:图|figure)\b", str(getattr(pages[idx], "markdown_text", "") or ""), flags=re.IGNORECASE | re.MULTILINE)
            ]
            image_target_indexes = preferred_indexes or target_indexes
            image_groups = self._group_values_evenly(images, len(image_target_indexes))
            for page_idx, group in zip(image_target_indexes, image_groups):
                if group:
                    pages[page_idx].assets.images = [
                        item
                        for item in list(pages[page_idx].assets.images or [])
                        if self._looks_like_real_image_asset(item)
                    ]
                existing = set(pages[page_idx].get_images())
                for image in group:
                    if image in existing:
                        continue
                    pages[page_idx].add_image(image)
                    existing.add(image)
                pages[page_idx].metadata["image_count"] = len(pages[page_idx].get_images())

    def create_mono_page_node(
        self,
        *,
        page_number: int,
        page_text: str,
        markdown_text: str,
        assets: Optional[PageAssets] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonoPage:
        page_meta = dict(metadata or {})
        section_title = str(page_meta.get("section_title") or "").strip()
        if self._looks_like_catalogue_page(page_text) or self._is_catalogue_keyword(section_title):
            page_meta["section_title"] = "目录"
            if self._is_catalogue_keyword(str(page_meta.get("section_path") or "")) or not str(page_meta.get("section_path") or "").strip():
                page_meta["section_path"] = "目录"
            return Catalogue(title="目录", markdown_text=markdown_text, assets=assets, metadata=page_meta)
        if section_title == "封面":
            return Cover(title="封面", markdown_text=markdown_text, assets=assets, metadata=page_meta)
        return Content(title="", markdown_text=markdown_text, assets=assets, metadata=page_meta)

    def build_catalog_tree(self, pages: Sequence[MonoPage], ranges: Sequence[Dict[str, Any]]) -> List[Page]:
        if not ranges:
            return list(pages)

        indexed_ranges = list(enumerate(list(ranges)))
        indexed_ranges.sort(
            key=lambda pair: (
                int(self._coerce_positive_int(pair[1].get("start")) or 10**9),
                int(pair[1].get("order", pair[0])),
                int(self._coerce_positive_int(pair[1].get("level")) or 1),
            )
        )
        normalized_ranges = [item for _, item in indexed_ranges]

        chapters: List[Chapter] = []
        for idx, item in enumerate(normalized_ranges, start=1):
            chapter_meta = {
                "section_id": f"chapter-{idx}",
                "section_title": item["title"],
                "section_path": item["title"],
                "section_start_page": item["start"],
                "section_end_page": item["end"],
                "level": item["level"],
                "order": item.get("order", idx - 1),
            }
            chapters.append(Chapter(title=item["title"], metadata=chapter_meta, SubContent=[]))

        stack: List[tuple[int, Chapter]] = []
        roots: List[Page] = []
        for chapter in chapters:
            level = self._coerce_positive_int(chapter.metadata.get("level")) or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                stack[-1][1].add_child(chapter)
                parent_path = str(stack[-1][1].metadata.get("section_path") or stack[-1][1].title)
                chapter.metadata["section_path"] = f"{parent_path}/{chapter.title}"
            else:
                roots.append(chapter)
            stack.append((level, chapter))

        def _normalize_section_token(value: Any) -> str:
            text = str(value or "").strip().lower()
            text = re.sub(r"\s+", "", text)
            text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
            return text

        def _split_section_path(value: Any) -> List[str]:
            raw = str(value or "").strip().replace("＞", ">")
            if not raw:
                return []
            parts = re.split(r"\s*(?:>|/)\s*", raw)
            cleaned: List[str] = []
            for part in parts:
                token = _normalize_section_token(part)
                if not token or token == "document" or re.fullmatch(r"l[1-6]", token):
                    continue
                cleaned.append(token)
            return cleaned

        chapter_path_lookup: Dict[tuple[str, ...], Chapter] = {}
        for chapter in chapters:
            path_parts = _split_section_path(chapter.metadata.get("section_path") or chapter.title)
            if path_parts:
                chapter_path_lookup[tuple(path_parts)] = chapter

        def _select_chapter_from_page_path(page: MonoPage) -> Optional[Chapter]:
            resolver = str(page.metadata.get("section_resolver") or "").strip().lower()
            if resolver != "main_section_map":
                return None
            path_parts = _split_section_path(page.metadata.get("resolved_section_path") or page.metadata.get("section_path") or page.title)
            if not path_parts:
                return None
            for width in range(len(path_parts), 0, -1):
                candidate = chapter_path_lookup.get(tuple(path_parts[:width]))
                if candidate is not None:
                    return candidate
            return None

        page_assigned = [False for _ in pages]
        for pidx, page in enumerate(pages, start=1):
            page_number = self._coerce_positive_int(page.metadata.get("page")) or self._coerce_positive_int(page.metadata.get("physical_page")) or pidx
            selected = _select_chapter_from_page_path(page)
            if selected is not None:
                page_assigned[pidx - 1] = True
                page_path = str(selected.metadata.get("section_path") or selected.title)
                page.metadata["section_path"] = page_path
                page.metadata["section_title"] = selected.title
                page.metadata["resolved_section_path"] = page_path
                page.metadata["section_resolver"] = "catalog_tree"
                selected.add_child(page)
                continue

            candidates: List[tuple[int, int, int, int, Chapter]] = []
            for chapter in chapters:
                start = self._coerce_positive_int(chapter.metadata.get("section_start_page")) or 1
                end = self._coerce_positive_int(chapter.metadata.get("section_end_page")) or start
                if not (start <= int(page_number) <= end):
                    continue
                level = self._coerce_positive_int(chapter.metadata.get("level")) or 1
                span = max(1, end - start + 1)
                order = self._coerce_positive_int(chapter.metadata.get("order")) or 0
                # Deterministic policy: deeper level, narrower span, earlier range start, then catalog order.
                candidates.append((-int(level), int(span), int(start), int(order), chapter))

            if not candidates:
                continue

            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            selected = candidates[0][4]
            page_assigned[pidx - 1] = True
            if not selected.metadata.get("section_path"):
                selected.metadata["section_path"] = selected.title
            page_path = str(selected.metadata.get("section_path"))
            page.metadata["section_path"] = page_path
            page.metadata["section_title"] = selected.title
            page.metadata["resolved_section_path"] = page_path
            page.metadata["section_resolver"] = "catalog_tree"
            selected.add_child(page)

        for idx, page in enumerate(pages):
            if not page_assigned[idx]:
                roots.append(page)

        def _chapter_has_local_assets(chapter: Chapter) -> bool:
            return bool(
                chapter.assets.headers
                or chapter.assets.footers
                or chapter.assets.annotations
                or chapter.assets.citations
                or chapter.assets.page_numbers
                or chapter.assets.images
            )

        def _build_synthetic_monopage_from_chapter(chapter: Chapter) -> Content:
            meta = dict(chapter.metadata or {})
            section_title = str(meta.get("section_title") or chapter.title or "Untitled")
            section_path = str(meta.get("section_path") or section_title)
            start_page = self.coerce_page_number(meta.get("section_start_page"))
            end_page = self.coerce_page_number(meta.get("section_end_page"))
            page_no = start_page or end_page or 1
            mono_meta = {
                **meta,
                "section_title": section_title,
                "section_path": section_path,
                "section_start_page": start_page or page_no,
                "section_end_page": end_page or page_no,
                "page": page_no,
            }
            mono = Content(title="", markdown_text=str(chapter.markdown_text or ""), metadata=mono_meta)
            for item in chapter.assets.headers:
                mono.add_header(item)
            for item in chapter.assets.footers:
                mono.add_footer(item)
            for item in chapter.assets.annotations:
                mono.add_annotation(item)
            for item in chapter.assets.citations:
                mono.add_citation(item)
            for item in chapter.assets.page_numbers:
                mono.add_page_number(item)
            for item in chapter.assets.images:
                mono.add_image(item)
            return mono

        def _prune_empty_chapter_nodes(node: Page) -> Optional[Page]:
            if not isinstance(node, Chapter):
                return node

            pruned_children: List[Page] = []
            for child in list(node.SubContent or []):
                pruned = _prune_empty_chapter_nodes(child)
                if pruned is not None:
                    pruned_children.append(pruned)
            node.SubContent = pruned_children

            if node.SubContent:
                return node

            if str(node.markdown_text or "").strip() or _chapter_has_local_assets(node):
                node.SubContent = [_build_synthetic_monopage_from_chapter(node)]
                return node

            return None

        def _refresh_chapter_page_span(node: Page) -> None:
            if not isinstance(node, Chapter):
                return
            for child in node.SubContent:
                _refresh_chapter_page_span(child)
            page_numbers: List[int] = []
            for leaf in node.flatten_mono_pages():
                metadata = dict(getattr(leaf, "metadata", {}) or {})
                page_no = self.coerce_page_number(metadata.get("page"))
                if page_no is None:
                    page_no = self.coerce_page_number(metadata.get("section_start_page"))
                if page_no is not None:
                    page_numbers.append(int(page_no))
            if page_numbers:
                node.metadata["section_start_page"] = min(page_numbers)
                node.metadata["section_end_page"] = max(page_numbers)

        normalized_roots: List[Page] = []
        for root in roots:
            pruned_root = _prune_empty_chapter_nodes(root)
            if pruned_root is not None:
                normalized_roots.append(pruned_root)
        roots = normalized_roots
        for root in roots:
            _refresh_chapter_page_span(root)

        def _node_start_page(node: Page) -> int:
            if isinstance(node, MonoPage):
                metadata = dict(getattr(node, "metadata", {}) or {})
                value = self.coerce_page_number(metadata.get("page"))
                if value is None:
                    value = self.coerce_page_number(metadata.get("section_start_page"))
                return int(value or 10**9)
            if isinstance(node, Chapter):
                metadata = dict(getattr(node, "metadata", {}) or {})
                value = self.coerce_page_number(metadata.get("section_start_page"))
                if value is not None:
                    return int(value)
            leaves = node.flatten_mono_pages()
            best = 10**9
            for leaf in leaves:
                metadata = dict(getattr(leaf, "metadata", {}) or {})
                value = self.coerce_page_number(metadata.get("page"))
                if value is None:
                    value = self.coerce_page_number(metadata.get("section_start_page"))
                if value is not None:
                    best = min(best, int(value))
            return best

        def _node_order_value(node: Page) -> int:
            metadata = dict(getattr(node, "metadata", {}) or {})
            return int(self._coerce_positive_int(metadata.get("order")) or 0)

        def _node_sort_key(node: Page) -> tuple[int, int, int, str]:
            title = str(getattr(node, "title", "") or (getattr(node, "metadata", {}) or {}).get("section_title") or "").strip()
            mono_rank = 0 if isinstance(node, MonoPage) else 1
            return (_node_start_page(node), mono_rank, _node_order_value(node), title)

        def _sort_tree_children(node: Page) -> None:
            if not isinstance(node, Chapter):
                return
            for child in node.SubContent:
                _sort_tree_children(child)
            node.SubContent.sort(key=_node_sort_key)

        def _wrap_top_level_catalogue_pages(nodes: Sequence[Page]) -> List[Page]:
            wrapped: List[Page] = []
            idx = 0
            ordered = list(nodes)
            while idx < len(ordered):
                item = ordered[idx]
                if isinstance(item, MonoPage) and item.category == PageType.CATALOGUE:
                    group: List[MonoPage] = []
                    while idx < len(ordered):
                        candidate = ordered[idx]
                        if not isinstance(candidate, MonoPage) or candidate.category != PageType.CATALOGUE:
                            break
                        group.append(candidate)
                        idx += 1
                    page_numbers = [
                        page_no
                        for page in group
                        for page_no in self._coerce_page_numbers_from_page(page)
                    ]
                    start_page = min(page_numbers) if page_numbers else _node_start_page(group[0])
                    end_page = max(page_numbers) if page_numbers else _node_start_page(group[-1])
                    wrapped.append(
                        Chapter(
                            title="目录",
                            metadata={
                                "section_id": f"front-catalogue-{start_page}-{end_page}",
                                "section_title": "目录",
                                "section_path": "目录",
                                "section_start_page": start_page,
                                "section_end_page": end_page,
                                "level": 1,
                                "order": -1,
                            },
                            SubContent=group,
                        )
                    )
                    continue
                wrapped.append(item)
                idx += 1
            return wrapped

        roots.sort(key=_node_sort_key)
        roots = _wrap_top_level_catalogue_pages(roots)
        for root in roots:
            _refresh_chapter_page_span(root)
            _sort_tree_children(root)
        roots.sort(key=_node_sort_key)
        return roots

    def _build_from_cleaned_text(self) -> "RAG_DB_Document":
        cleaned_text = str(getattr(self, "cleaned_text", "") or "").strip()
        if not cleaned_text:
            self.set_page_nodes([])
            self.chunk_documents = []
            self.catalog = []
            self.page_count = 0
            self.pagination_mode = "page-tree"
            return self

        raw_pages = [part.strip() for part in cleaned_text.split("\f")]
        page_texts = [part for part in raw_pages if part]
        if not page_texts:
            page_texts = [cleaned_text]

        page_nodes: List[MonoPage] = []
        chunks: List[Document] = []

        for page_idx, page_text in enumerate(page_texts, start=1):
            section_title = f"Page {page_idx}"
            page_meta: Dict[str, Any] = {
                "doc_name": self.doc_name,
                "file_name": self.doc_name,
                "source_extension": self.source_extension,
                "section_id": f"page-{page_idx}",
                "section_title": section_title,
                "section_path": section_title,
                "section_start_page": page_idx,
                "section_end_page": page_idx,
                "page": page_idx,
            }

            node = self.create_mono_page_node(
                page_number=page_idx,
                page_text=page_text,
                markdown_text=page_text,
                metadata=page_meta,
            )
            node.add_page_number(page_idx)
            page_nodes.append(node)

            for chunk_idx, chunk_text in enumerate(self._split_text_chunks(page_text), start=1):
                chunk_meta = dict(page_meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=chunk_meta,
                        doc_id=f"{self.doc_name}::page::{page_idx}::chunk::{chunk_idx}",
                    )
                )

        self.set_page_nodes(page_nodes)
        self.chunk_documents = chunks
        self.page_count = len(page_nodes)
        self.pagination_mode = "page-tree"
        self.catalog = self.catalog_payload()
        return self

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

        for mono_page in self.get_mono_pages():
            text = (mono_page.markdown_text or "").strip()
            if not text:
                continue

            metadata = dict(mono_page.metadata or {})
            section_path = str(metadata.get("section_path") or mono_page.title or "")
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
        query_norm = (query_text or "").strip().lower()
        regex_matches = self.retrieve_by_regex(
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )
        enriched: List[Dict[str, Any]] = []
        for row in regex_matches:
            metadata = dict(row.get("metadata") or {})
            section_id = str(metadata.get("section_id") or "").strip()
            section_path = str(row.get("section_path") or "")
            text = str(row.get("text") or "")
            score = float(section_scores.get(section_id, 0.0))
            if query_norm and query_norm in text.lower():
                score += 0.2
            if section_path:
                score += float(section_scores.get(section_path, 0.0))

            enriched_row = dict(row)
            enriched_row["score"] = score
            enriched.append(enriched_row)
        return enriched

    def list_payload(self) -> Dict[str, Any]:
        payload = self.to_payload()
        payload["doc_name"] = self.doc_name
        payload["page_count"] = self.page_count
        payload["pagination_mode"] = self.pagination_mode
        payload["catalog"] = self.catalog_payload()
        return payload

    def query(
        self,
        *,
        query_text: str = "",
        section_scores: Optional[Dict[str, float]] = None,
        compiled_regex: Optional[Pattern[str]] = None,
        section: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        chunk: Optional[str] = None,
        use_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        if use_vector:
            return self.retrieve_by_vector(
                query_text=query_text,
                section_scores=dict(section_scores or {}),
                compiled_regex=compiled_regex,
                section=section,
                page_start=page_start,
                page_end=page_end,
                chunk=chunk,
            )
        return self.retrieve_by_regex(
            compiled_regex=compiled_regex,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk=chunk,
        )

    def extract_catalog(self) -> List[Dict[str, Any]]:
        return self.catalog_payload()

    def catalog_payload(self) -> List[Dict[str, Any]]:
        pages = self.get_mono_pages()
        if not pages:
            return list(getattr(self, "catalog", []))

        catalog: List[Dict[str, Any]] = []
        for idx, mono_page in enumerate(pages, start=1):
            metadata = dict(mono_page.metadata or {})
            start_page = self.coerce_page_number(metadata.get("section_start_page"))
            end_page = self.coerce_page_number(metadata.get("section_end_page"))
            page = self.coerce_page_number(metadata.get("page"))
            resolved_start = start_page or page or idx
            resolved_end = end_page or resolved_start
            display_title = str(metadata.get("section_title") or mono_page.title or "").strip() or f"Page {idx}"
            catalog.append(
                {
                    "title": display_title,
                    "page": resolved_start,
                    "end_page": resolved_end,
                    "category": mono_page.category,
                }
            )
        return catalog

    def set_page_nodes(self, page_nodes: Sequence[Page]) -> None:
        self.page_nodes = list(page_nodes)
        self.SubContent = list(page_nodes)
        self._hydrate_chapter_markdown()

    def _hydrate_chapter_markdown(self) -> None:
        for node in self.page_nodes:
            if isinstance(node, Chapter):
                self._hydrate_chapter_markdown_recursive(node)

    def _hydrate_chapter_markdown_recursive(self, chapter: Chapter) -> str:
        parts: List[str] = []
        own_text = str(chapter.markdown_text or "").strip()

        for child in chapter.SubContent:
            if isinstance(child, Chapter):
                child_text = self._hydrate_chapter_markdown_recursive(child)
            else:
                child_text = str(child.markdown_text or "").strip()
            if child_text:
                parts.append(child_text)

        if own_text and not parts:
            parts.append(own_text)

        merged = "\n\n".join(parts).strip()
        chapter.markdown_text = merged
        return merged

    def get_page_nodes(self) -> List[Page]:
        if hasattr(self, "page_nodes"):
            return list(self.page_nodes)
        return list(getattr(self, "SubContent", []))

    def get_mono_pages(self) -> List[MonoPage]:
        leaves: List[MonoPage] = []
        for node in self.get_page_nodes():
            leaves.extend(node.flatten_mono_pages())
        return leaves

    def get_single_pages(self) -> List[MonoPage]:
        unique_pages: List[MonoPage] = []
        seen: Set[int] = set()
        for page in self.get_mono_pages():
            metadata = dict(getattr(page, "metadata", {}) or {})
            physical_page = self.coerce_page_number(metadata.get("physical_page"))
            page_no = self.coerce_page_number(metadata.get("page"))
            key = int(physical_page or page_no or 0)
            if key <= 0 or key in seen:
                continue
            seen.add(key)
            unique_pages.append(page)
        return unique_pages

    @staticmethod
    def _unique_nonempty(values: Sequence[Any]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in values:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    @classmethod
    def _coerce_page_numbers_from_page(cls, page: Page) -> List[int]:
        values: List[int] = []

        for item in page.get_page_numbers():
            number = cls.coerce_page_number(item)
            if number is not None and number > 0:
                values.append(int(number))

        metadata = dict(getattr(page, "metadata", {}) or {})
        for key in ("section_start_page", "section_end_page", "page"):
            number = cls.coerce_page_number(metadata.get(key))
            if number is not None and number > 0:
                values.append(int(number))

        if not values:
            return []
        return sorted(set(values))

    @staticmethod
    def _format_page_range_label(page_numbers: Sequence[int]) -> str:
        numbers = sorted({int(item) for item in page_numbers if int(item) > 0})
        if not numbers:
            return ""
        if len(numbers) == 1:
            return str(numbers[0])
        return f"{numbers[0]}-{numbers[-1]}"

    def _compose_chapter_markdown(
        self,
        chapter: Chapter,
        *,
        heading_level: int = 2,
        visited: Optional[Set[int]] = None,
    ) -> str:
        visited_ids = visited if visited is not None else set()
        chapter_id = id(chapter)
        if chapter_id in visited_ids:
            level = min(max(int(heading_level), 1), 6)
            title = str(chapter.title or "").strip() or "Untitled"
            return f"{'#' * level} {title}\n\n[递归保护] 检测到循环引用，已停止继续展开。"
        visited_ids.add(chapter_id)

        title = str(chapter.title or "").strip() or "Untitled"
        level = min(max(int(heading_level), 1), 6)
        heading = f"{'#' * level} {title}"

        chapter_pages = chapter.flatten_mono_pages()
        page_numbers: List[int] = []
        for mono_page in chapter_pages:
            page_numbers.extend(self._coerce_page_numbers_from_page(mono_page))
        page_range = self._format_page_range_label(page_numbers)

        notes = self._unique_nonempty(chapter.get_annotations())
        own_markdown = str(getattr(chapter, "markdown_text", "") or "").strip()
        children_markdown = [
            self._compose_page_markdown(child, heading_level=min(level + 1, 6), visited=visited_ids)
            for child in chapter.SubContent
        ]
        children_markdown = [item for item in children_markdown if item.strip()]

        parts: List[str] = [heading]
        if page_range:
            parts.append(f"页码范围: {page_range}")
        if notes:
            parts.append("注释:")
            parts.extend([f"- {item}" for item in notes])
        if own_markdown:
            parts.append("正文:")
            parts.append(own_markdown)
        if children_markdown:
            parts.extend(children_markdown)
        result = "\n\n".join(parts).strip()
        visited_ids.discard(chapter_id)
        return result

    def _compose_monopage_markdown(self, mono_page: MonoPage, *, heading_level: int = 3) -> str:
        metadata = dict(getattr(mono_page, "metadata", {}) or {})
        title = str(mono_page.title or metadata.get("section_title") or "").strip() or "Untitled Page"
        level = min(max(int(heading_level), 1), 6)
        heading = f"{'#' * level} {title}"

        page_numbers = self._coerce_page_numbers_from_page(mono_page)
        page_range = self._format_page_range_label(page_numbers)
        notes = self._unique_nonempty(mono_page.get_annotations())
        body_markdown = str(mono_page.markdown_text or "").strip()

        parts: List[str] = [heading]
        if page_range:
            parts.append(f"页码: {page_range}")
        if notes:
            parts.append("注释:")
            parts.extend([f"- {item}" for item in notes])
        if body_markdown:
            parts.append("正文:")
            parts.append(body_markdown)
        return "\n\n".join(parts).strip()

    def _compose_page_markdown(
        self,
        page: Page,
        *,
        heading_level: int = 2,
        visited: Optional[Set[int]] = None,
    ) -> str:
        if isinstance(page, Chapter):
            return self._compose_chapter_markdown(page, heading_level=heading_level, visited=visited)
        if isinstance(page, MonoPage):
            return self._compose_monopage_markdown(page, heading_level=heading_level)

        # Unknown Page subtype fallback: keep markdown text to avoid data loss.
        metadata = dict(getattr(page, "metadata", {}) or {})
        title = str(getattr(page, "title", "") or metadata.get("section_title") or "").strip() or "Untitled"
        level = min(max(int(heading_level), 1), 6)
        heading = f"{'#' * level} {title}"
        body_markdown = str(getattr(page, "markdown_text", "") or "").strip()
        if body_markdown:
            return f"{heading}\n\n{body_markdown}".strip()
        return heading

    def export_markdown_from_tree(self) -> str:
        visited: Set[int] = set()
        parts = [
            self._compose_page_markdown(node, heading_level=2, visited=visited)
            for node in self.get_page_nodes()
        ]
        parts = [item for item in parts if item.strip()]
        return "\n\n".join(parts).strip()

    def build_chapter(self, title: str, pages: Sequence[Page]) -> Chapter:
        return Chapter(title=title, SubContent=list(pages))

    @staticmethod
    def _normalize_doc_path(value: str) -> str:
        return value.replace("\\", "/").strip()

    @staticmethod
    def _detect_source_extension(metadata: Dict[str, Any]) -> str:
        ext = str(metadata.get("source_extension") or "").strip().lower()
        if ext:
            return ext
        file_name = str(metadata.get("file_name") or "").strip()
        if file_name:
            return os.path.splitext(file_name)[1].lower()
        return ""

    @classmethod
    def _resolve_doc_name(cls, base_doc_id: str, metadata: Dict[str, Any]) -> str:
        current = str(metadata.get("doc_name") or "").strip()
        file_name = str(metadata.get("file_name") or "").strip()
        if file_name:
            return cls._normalize_doc_path(os.path.abspath(file_name))

        if current:
            data_dir = ""
            try:
                import config  # type: ignore

                data_dir = str(getattr(config.settings, "DATA_DIR", "") or "").strip()
            except Exception:
                data_dir = ""

            if os.path.isabs(current):
                return cls._normalize_doc_path(os.path.abspath(current))
            if data_dir:
                return cls._normalize_doc_path(os.path.abspath(os.path.join(data_dir, current)))
            return cls._normalize_doc_path(os.path.abspath(current))

        if base_doc_id:
            return cls._normalize_doc_path(os.path.abspath(str(base_doc_id)))
        return cls._normalize_doc_path(str(base_doc_id))

    @staticmethod
    def _roman_numeral_to_int(token: str) -> Optional[int]:
        text = str(token or "").strip().upper()
        if not text or not re.fullmatch(r"[IVXLCDM]{1,8}", text):
            return None
        mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        prev = 0
        for ch in reversed(text):
            value = mapping.get(ch, 0)
            if value < prev:
                total -= value
            else:
                total += value
                prev = value
        return total if total > 0 else None

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
            roman = RAG_DB_Document._roman_numeral_to_int(text)
            if roman is not None:
                return roman
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
