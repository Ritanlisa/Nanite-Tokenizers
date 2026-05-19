from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai import OpenAI as OpenAIClient

import config

logger = logging.getLogger(__name__)

_SUPPORTED_MIME: Dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

_MIN_IMAGE_BYTES = 512


def ocr_enabled() -> bool:
    return bool(config.settings.OCR_API_URL and config.settings.OCR_MODEL)


def _normalize_image_bytes(image_bytes: bytes) -> Tuple[bytes, str]:
    from PIL import Image, UnidentifiedImageError

    if len(image_bytes) < _MIN_IMAGE_BYTES:
        return b"", ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "").upper()
    except (UnidentifiedImageError, Exception):
        return b"", ""

    if fmt in _SUPPORTED_MIME:
        return image_bytes, _SUPPORTED_MIME[fmt]

    try:
        buf = io.BytesIO()
        if img.mode in ("RGBA", "LA", "PA"):
            img.save(buf, format="PNG")
        else:
            img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception:
        return b"", ""


def _extract_message_text(message_content) -> str:
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    if isinstance(message_content, str):
        return message_content
    return ""


def _ocr_image(image_bytes: bytes) -> str:
    if not ocr_enabled():
        return ""

    image_bytes, mime_type = _normalize_image_bytes(image_bytes)
    if not image_bytes or not mime_type:
        return ""

    client = OpenAIClient(
        api_key=config.settings.OCR_API_KEY,
        base_url=config.settings.OCR_API_URL,
        timeout=config.settings.OCR_TIMEOUT,
        max_retries=1,
    )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_b64}"
    try:
        ocr_model = config.settings.OCR_MODEL
        if not ocr_model:
            logger.error("OCR_MODEL is not set in config.settings.")
            return ""
        response = client.chat.completions.create(
            model=ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": config.settings.OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
    except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
        logger.warning("OCR request failed: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("OCR error (skipping image): %s", exc)
        return ""

    if not response.choices:
        return ""
    content = response.choices[0].message.content
    return _extract_message_text(content).strip()


def ocr_image_data_url(data_url: str) -> str:
    if not ocr_enabled():
        return ""
    if not data_url.startswith("data:image/"):
        logger.warning("Unsupported image data URL")
        return ""
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError:
        logger.warning("Malformed image data URL")
        return ""
    if "base64" not in header:
        logger.warning("Image data URL is not base64-encoded")
        return ""
    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:
        logger.warning("Failed to decode image data URL: %s", exc)
        return ""
    return _ocr_image(image_bytes)


class OCRPDFReader(BaseReader):
    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    def load_data(
        self, file: Path, extra_info: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        document = fitz.open(file)
        parts: List[str] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text("text").strip()
            has_images = bool(page.get_images(full=True))
            ocr_text = ""
            if has_images and ocr_enabled():
                pixmap = page.get_pixmap(dpi=self.dpi)
                ocr_text = _ocr_image(pixmap.tobytes("png"))

            combined = ""
            if page_text:
                combined = page_text
            if ocr_text:
                combined = f"{combined}\n{ocr_text}" if combined else ocr_text

            if combined:
                parts.append(f"Page {page_index + 1}\n{combined}")

        if not parts:
            return []

        metadata = {"file_name": str(file)}
        if extra_info:
            metadata.update(extra_info)

        text = "\n\n".join(parts)
        return [Document(text=text, metadata=metadata)]
