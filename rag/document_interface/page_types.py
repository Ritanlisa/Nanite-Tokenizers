from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from .page_assets import ImageAsset, PageAssets, _normalize_image_asset


class PageType:
    COVER = "cover"
    CATALOGUE = "catalogue"
    INTRODUCTION = "introduction"
    CONTENT = "content"
    APPENDIX = "appendix"
    CHAPTER = "chapter"



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

    @staticmethod
    def _serialize_image_items(items: Sequence[Any]) -> List[Any]:
        return [item.to_payload() if isinstance(item, ImageAsset) else str(item or "") for item in list(items or [])]

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        try:
            text = str(value).strip()
            if not text:
                return None
            number = int(float(text))
        except Exception:
            return None
        if number <= 0:
            return None
        return number

    @classmethod
    def _extract_markdown_image_targets(cls, markdown_text: str) -> List[tuple[int, int]]:
        targets: List[tuple[int, int]] = []
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", str(markdown_text or "")):
            target = str(match.group(1) or "").strip()
            page_match = re.fullmatch(r"image://page-(\d+)/(\d+)", target, flags=re.IGNORECASE)
            if page_match is None:
                continue
            page_no = cls._coerce_positive_int(page_match.group(1))
            image_index = cls._coerce_positive_int(page_match.group(2))
            if page_no is None or image_index is None:
                continue
            targets.append((page_no, image_index))
        return targets

    def _payload_markdown_text(self) -> str:
        return str(self.markdown_text or "")

    def _payload_image_lookup(self) -> Dict[tuple[int, int], Any]:
        page_no = self._coerce_positive_int(self.metadata.get("page"))
        if page_no is None:
            page_no = self._coerce_positive_int(self.metadata.get("physical_page"))
        if page_no is None:
            page_no = self._coerce_positive_int(self.metadata.get("section_start_page"))
        if page_no is None:
            return {}

        page_image_indexes = [
            int(page_index)
            for page_index in [self._coerce_positive_int(item) for item in list(self.metadata.get("page_image_indexes") or [])]
            if page_index is not None
        ]

        lookup: Dict[tuple[int, int], Any] = {}
        for local_index, item in enumerate(list(self.assets.images or []), start=1):
            image_index = page_image_indexes[local_index - 1] if local_index - 1 < len(page_image_indexes) else local_index
            if image_index <= 0:
                continue
            lookup[(int(page_no), int(image_index))] = item
        return lookup

    def _payload_image_items(self, markdown_text: Optional[str] = None) -> List[Any]:
        text = self._payload_markdown_text() if markdown_text is None else str(markdown_text or "")
        targets = self._extract_markdown_image_targets(text)
        if not targets:
            return []
        lookup = self._payload_image_lookup()
        resolved: List[Any] = []
        for key in targets:
            item = lookup.get(key)
            if item is not None:
                resolved.append(item)
        return resolved

    def to_payload(self) -> Dict[str, Any]:
        payload_markdown = self._payload_markdown_text()
        return {
            "title": self.title,
            "category": self.category,
            "markdown_text": payload_markdown,
            "headers": list(self.assets.headers),
            "footers": list(self.assets.footers),
            "annotations": list(self.assets.annotations),
            "citations": list(self.assets.citations),
            "page_numbers": list(self.assets.page_numbers),
            "images": self._serialize_image_items(self._payload_image_items(payload_markdown)),
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
            if isinstance(item, Chapter):
                content = item.merged_markdown().strip()
            else:
                content = item.markdown_text.strip()
            if content:
                parts.append(content)
        if parts:
            return "\n\n".join(parts).strip()
        return self.markdown_text.strip()

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

    def _payload_markdown_text(self) -> str:
        return self.merged_markdown()

    def _payload_image_lookup(self) -> Dict[tuple[int, int], Any]:
        lookup: Dict[tuple[int, int], Any] = {}
        for mono_page in self.flatten_mono_pages():
            for key, value in mono_page._payload_image_lookup().items():
                if key not in lookup:
                    lookup[key] = value
        return lookup

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload["markdown_text"] = self.merged_markdown()
        payload["SubContent"] = [item.to_payload() for item in self.SubContent]
        return payload
