from __future__ import annotations

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


@dataclass
class PageAssets:
    headers: List[str] = field(default_factory=list)
    footers: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    page_numbers: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

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

    def add_image(self, value: str) -> None:
        v = str(value or "").strip()
        if v:
            self.assets.images.append(v)

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

    def get_images(self) -> List[str]:
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
            "images": self.get_images(),
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
    def __init__(self, *, title: str = "Body", markdown_text: str = "", assets: Optional[PageAssets] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            title=title,
            page_type=PageType.CONTENT,
            markdown_text=markdown_text,
            assets=assets,
            metadata=metadata,
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
        if self.markdown_text.strip():
            parts.append(self.markdown_text.strip())
        for item in self.SubContent:
            content = item.markdown_text.strip()
            if content:
                parts.append(content)
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

    def get_images(self) -> List[str]:
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
    def _clean_heading_title(line: str) -> str:
        text = str(line or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^[#\-\*\d\.\)\s]+", "", text)
        text = re.sub(r"[·•.\s]+\d{1,4}\s*$", "", text)
        return text.strip()

    @staticmethod
    def _normalized_heading_text(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    @staticmethod
    def _is_noise_heading_line(line: str) -> bool:
        text = str(line or "").strip().lower()
        if not text:
            return True
        noise = {"目录", "contents", "table of contents", "toc"}
        return text in noise

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
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇]", text):
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
                    }
                )
        markers.sort(key=lambda item: (int(item["page"]), int(item["level"])))
        return markers

    def _extract_markers_from_manual_toc(self, page_texts: Sequence[str]) -> List[Dict[str, Any]]:
        lines: List[str] = []
        for page_text in page_texts[:12]:
            lines.extend(str(page_text or "").splitlines())

        markers: List[Dict[str, Any]] = []
        for raw in lines:
            text = str(raw or "").strip()
            if not text:
                continue
            match = re.match(r"^\s*([\u4e00-\u9fffA-Za-z0-9][^\n]{1,140}?)(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
            if not match:
                continue
            title = self._clean_heading_title(match.group(1))
            page = self._coerce_positive_int(match.group(2))
            if not title or page is None or self._is_noise_heading_line(title):
                continue
            level = self._heading_level(title) or 1
            markers.append(
                {
                    "title": title,
                    "page": page,
                    "level": max(1, min(level, 6)),
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
            line = first_lines[0]
            level = self._heading_level(line)
            if level is None:
                continue
            title = self._clean_heading_title(line)
            if not title or self._is_noise_heading_line(title):
                continue
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
        for item in ranges:
            if int(item.get("start", 0)) <= page_idx <= int(item.get("end", 0)):
                return dict(item)
        return None

    def build_catalog_tree(self, pages: Sequence[Content], ranges: Sequence[Dict[str, Any]]) -> List[Page]:
        if not ranges:
            return list(pages)

        chapters: List[Chapter] = []
        for idx, item in enumerate(ranges, start=1):
            chapter_meta = {
                "section_id": f"chapter-{idx}",
                "section_title": item["title"],
                "section_path": item["title"],
                "section_start_page": item["start"],
                "section_end_page": item["end"],
                "level": item["level"],
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

        page_assigned = [False for _ in pages]
        for chapter in chapters:
            start = self._coerce_positive_int(chapter.metadata.get("section_start_page")) or 1
            end = self._coerce_positive_int(chapter.metadata.get("section_end_page")) or start
            for pidx, page in enumerate(pages, start=1):
                if not (start <= pidx <= end):
                    continue
                page_assigned[pidx - 1] = True
                if not chapter.metadata.get("section_path"):
                    chapter.metadata["section_path"] = chapter.title
                page_path = str(chapter.metadata.get("section_path"))
                page.metadata["section_path"] = page_path
                page.metadata["section_title"] = chapter.title
                chapter.add_child(page)

        for idx, page in enumerate(pages):
            if not page_assigned[idx]:
                roots.append(page)
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

        page_nodes: List[Content] = []
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

            node = Content(title=section_title, markdown_text=page_text, metadata=page_meta)
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
            catalog.append(
                {
                    "title": mono_page.title or f"Page {idx}",
                    "page": resolved_start,
                    "end_page": resolved_end,
                    "category": mono_page.category,
                }
            )
        return catalog

    def set_page_nodes(self, page_nodes: Sequence[Page]) -> None:
        self.page_nodes = list(page_nodes)
        self.SubContent = list(page_nodes)

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
        return self.get_mono_pages()

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
        title = str(mono_page.title or "").strip() or "Untitled Page"
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
        title = str(getattr(page, "title", "") or "").strip() or "Untitled"
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
