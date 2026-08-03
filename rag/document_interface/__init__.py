"""Compatibility package for the former rag/document_interface.py.

Split into focused modules (page_assets / page_types / document);
all public and private symbols are re-exported here so existing
consumers keep working unchanged."""

from .page_assets import (
    ImageAsset,
    PageAssets,
    _OCR_RESULT_CACHE,
    _dedupe_image_values,
    _dedupe_text_values,
    _image_asset_key,
    _normalize_image_asset,
)
from .page_types import (
    Appendix,
    Catalogue,
    Chapter,
    Content,
    Cover,
    Introduction,
    MonoPage,
    Page,
    PageType,
    SemiPage,
)
from .document import RAG_DB_Document

__all__ = [
    "PageType", "ImageAsset", "PageAssets", "Page", "MonoPage", "Cover",
    "Catalogue", "Introduction", "Content", "SemiPage", "Appendix", "Chapter",
    "RAG_DB_Document", "_dedupe_text_values", "_dedupe_image_values",
    "_normalize_image_asset", "_image_asset_key", "_OCR_RESULT_CACHE",
]
