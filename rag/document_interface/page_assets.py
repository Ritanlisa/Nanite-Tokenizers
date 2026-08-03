from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


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
    ocr_text: str = ""

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
        ocr_text: str = "",
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
            ocr_text=str(ocr_text or "").strip(),
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
            ocr_text=str(payload.get("ocr_text") or "").strip(),
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
            "ocr_text": self.ocr_text,
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



_OCR_RESULT_CACHE: Dict[str, str] = {}



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
