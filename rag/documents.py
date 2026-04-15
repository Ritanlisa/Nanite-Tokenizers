from __future__ import annotations

import os
import re
import hashlib
import logging
import shutil
import subprocess
import site
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from llama_index.core import Document

from rag.document_doc import DocRAGDocument
from rag.document_docx import DocxRAGDocument
from rag.document_interface import ImageAsset, RAG_DB_Document
from rag.document_pdf import PDFRAGDocument
from rag.document_spreadsheet import SpreadsheetRAGDocument
from rag.document_text import TextRAGDocument
from rag.pdf_reader import get_pdf_reader

logger = logging.getLogger(__name__)
_UNO_BRIDGE_READY = False
_NATIVE_PAGE_COUNT_ERROR_PREFIX = "NATIVE_PAGE_COUNT_ERROR:"


def _enable_python3_uno_bridge() -> bool:
    global _UNO_BRIDGE_READY
    if _UNO_BRIDGE_READY:
        return True

    try:
        import uno  # type: ignore # noqa: F401

        _UNO_BRIDGE_READY = True
        return True
    except Exception:
        pass

    bridge_paths = [
        "/usr/lib/python3/dist-packages",
        "/usr/lib/libreoffice/program",
    ]
    existing_paths = [path for path in bridge_paths if os.path.isdir(path)]
    if not existing_paths:
        return False

    in_venv = bool(getattr(sys, "prefix", "") != getattr(sys, "base_prefix", ""))
    if in_venv:
        pth_written = False
        for sp in list(site.getsitepackages() or []):
            if not os.path.isdir(sp):
                continue
            pth_path = os.path.join(sp, "uno_bridge.pth")
            try:
                with open(pth_path, "w", encoding="utf-8") as fout:
                    fout.write("\n".join(existing_paths) + "\n")
                pth_written = True
            except Exception as exc:
                logger.debug("Failed to write uno bridge pth %s: %s", pth_path, exc)
        if not pth_written:
            logger.debug("No writable site-packages for uno bridge in current venv")

    for path in existing_paths:
        if path not in sys.path:
            sys.path.append(path)

    try:
        import uno  # type: ignore # noqa: F401

        _UNO_BRIDGE_READY = True
        return True
    except Exception as exc:
        logger.debug("Failed to import uno after bridge setup: %s", exc)
        return False


def _probe_libreoffice_physical_page_count(file_path: str) -> int:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return 0
    if not _enable_python3_uno_bridge():
        logger.debug("python3-uno bridge is unavailable for %s", file_path)
        return 0

    script = """
import os
import sys
import time
import random
import subprocess
import tempfile

path = sys.argv[1]
port = str(random.randint(20020, 22999))
with tempfile.TemporaryDirectory(prefix="lo_uno_profile_") as profile_dir:
    profile_url = "file://" + os.path.abspath(profile_dir).replace("\\\\", "/")
    cmd = [
        "soffice",
        "--headless",
        "--invisible",
        "--norestore",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        "--nologo",
        f"-env:UserInstallation={profile_url}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import uno
        from com.sun.star.beans import PropertyValue

        local = uno.getComponentContext()
        smgr = local.ServiceManager
        resolver = smgr.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)

        ctx = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.2)

        if ctx is None:
            print(0)
            raise SystemExit(0)

        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        props = (PropertyValue("Hidden", 0, True, 0), PropertyValue("ReadOnly", 0, True, 0))
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(path)), "_blank", 0, props)
        if doc is None:
            print(0)
            raise SystemExit(0)
        try:
            cursor = doc.getCurrentController().getViewCursor()
            if not cursor.jumpToPage(1):
                print(0)
                raise SystemExit(0)
            page_count = 0
            seen = set()
            while True:
                current = int(cursor.getPage() or 0)
                if current <= 0 or current in seen:
                    break
                seen.add(current)
                page_count += 1
                try:
                    moved = bool(cursor.jumpToNextPage())
                except Exception:
                    moved = False
                if not moved:
                    break
        except Exception:
            page_count = 0
        print(page_count if page_count > 0 else 0)
        try:
            doc.close(True)
        except Exception:
            pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
""".strip()

    try:
        proc = subprocess.run(
            [str(sys.executable), "-c", script, file_path],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        if proc.returncode != 0:
            logger.debug("LibreOffice page probe subprocess failed for %s: rc=%s stderr=%s", file_path, proc.returncode, str(proc.stderr or "").strip())
            return 0
        text = str(proc.stdout or "").strip()
        value = int(float(text)) if text else 0
        return value if value > 0 else 0
    except Exception as exc:
        logger.debug("Failed to probe LibreOffice page count for %s: %s", file_path, exc)
        return 0


def _require_libreoffice_physical_page_count(file_path: str, *, ext: str, fallback_counts: Optional[Sequence[int]] = None) -> int:
    value = int(_probe_libreoffice_physical_page_count(file_path) or 0)
    if value > 0:
        return value
    for fallback in list(fallback_counts or []):
        count = int(fallback or 0)
        if count > 0:
            return count
    raise RuntimeError(
        f"{_NATIVE_PAGE_COUNT_ERROR_PREFIX} "
        f"无法获取 {ext} 文档的 LibreOffice 物理页数。"
        "请确认系统已安装 python3-uno 且 soffice 可用；"
        "当前已禁用该场景下的自动回退，以避免错误分类。"
    )


def _extract_office_text_by_uno_pages(file_path: str) -> tuple[str, bool]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return "", False
    if not _enable_python3_uno_bridge():
        return "", False

    script = """
import os
import sys
import time
import random
import subprocess
import tempfile

path = sys.argv[1]
port = str(random.randint(23000, 25999))
with tempfile.TemporaryDirectory(prefix="lo_uno_pages_") as profile_dir:
    profile_url = "file://" + os.path.abspath(profile_dir).replace("\\\\", "/")
    cmd = [
        "soffice",
        "--headless",
        "--invisible",
        "--norestore",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        "--nologo",
        f"-env:UserInstallation={profile_url}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import uno
        from com.sun.star.beans import PropertyValue

        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)

        ctx = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.2)

        if ctx is None:
            print("")
            raise SystemExit(0)

        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        props = (PropertyValue("Hidden", 0, True, 0), PropertyValue("ReadOnly", 0, True, 0))
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(path)), "_blank", 0, props)
        if doc is None:
            print("")
            raise SystemExit(0)
        try:
            vc = doc.getCurrentController().getViewCursor()
            page_texts = []
            if not vc.jumpToPage(1):
                print("")
                raise SystemExit(0)
            seen = set()
            while True:
                current_page = int(vc.getPage() or 0)
                if current_page <= 0 or current_page in seen:
                    break
                seen.add(current_page)
                text = ""
                moved = False
                try:
                    vc.jumpToStartOfPage()
                    start = vc.getStart()
                    moved = bool(vc.jumpToNextPage())
                    if moved:
                        vc.jumpToStartOfPage()
                        end = vc.getStart()
                    else:
                        end = doc.Text.getEnd()
                    cursor = doc.Text.createTextCursorByRange(start)
                    cursor.gotoRange(end, True)
                    text = str(cursor.getString() or "")
                except Exception:
                    text = ""
                page_texts.append(text.strip())
                if not moved:
                    break
            print("\\n\\f\\n".join(page_texts).strip())
        finally:
            try:
                doc.close(True)
            except Exception:
                pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
""".strip()

    try:
        proc = subprocess.run(
            [str(sys.executable), "-c", script, file_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            logger.debug(
                "UNO page text extraction failed for %s: rc=%s stderr=%s",
                file_path,
                proc.returncode,
                str(proc.stderr or "").strip(),
            )
            return "", False
        text = str(proc.stdout or "").strip()
        if not text:
            return "", False
        return text, ("\f" in text)
    except Exception as exc:
        logger.debug("UNO page text extraction exception for %s: %s", file_path, exc)
        return "", False


def stable_doc_id(doc: Document) -> str:
    import hashlib

    source = (doc.metadata or {}).get("file_name") or (doc.text or "")[:200]
    return hashlib.md5(str(source).encode()).hexdigest()


def _split_native_pages(text: str) -> List[str]:
    raw_pages = [part.strip() for part in str(text or "").split("\f")]
    if raw_pages:
        return raw_pages
    stripped = str(text or "").strip()
    return [stripped] if stripped else []


def _normalize_image_value(value: Any) -> Optional[Any]:
    if isinstance(value, ImageAsset):
        return value
    asset = ImageAsset.from_payload(value)
    if asset is not None:
        return asset
    text = str(value or "").strip()
    return text or None


def _image_value_key(value: Any) -> str:
    normalized = _normalize_image_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, ImageAsset):
        return f"image:{normalized.asset_id}"
    return f"text:{normalized}"


def _dedupe_text_items(values: Sequence[Any]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in list(values or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _dedupe_image_items(values: Sequence[Any]) -> List[Any]:
    seen: Set[str] = set()
    ordered: List[Any] = []
    for item in list(values or []):
        normalized = _normalize_image_value(item)
        if normalized is None:
            continue
        key = _image_value_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _merge_page_layouts(
    primary_layouts: Sequence[Dict[str, Any]],
    secondary_layouts: Sequence[Dict[str, Any]],
    *,
    total_pages: int = 0,
) -> List[Dict[str, Any]]:
    page_total = max(
        int(total_pages or 0),
        len(list(primary_layouts or [])),
        len(list(secondary_layouts or [])),
        0,
    )
    merged: List[Dict[str, Any]] = []
    for index in range(page_total):
        primary = dict(primary_layouts[index]) if index < len(primary_layouts) and isinstance(primary_layouts[index], dict) else {}
        secondary = dict(secondary_layouts[index]) if index < len(secondary_layouts) and isinstance(secondary_layouts[index], dict) else {}
        page_no = int(
            RAG_DB_Document.coerce_page_number(secondary.get("page"))
            or RAG_DB_Document.coerce_page_number(primary.get("page"))
            or (index + 1)
        )
        merged.append(
            {
                "page": page_no,
                "headers": _dedupe_text_items(list(secondary.get("headers") or []) + list(primary.get("headers") or [])),
                "footers": _dedupe_text_items(list(secondary.get("footers") or []) + list(primary.get("footers") or [])),
                "page_number": str(secondary.get("page_number") or primary.get("page_number") or "").strip(),
                "images": _dedupe_image_items(list(primary.get("images") or []) + list(secondary.get("images") or [])),
            }
        )
    return merged


def _catalog_item_key(value: Any) -> tuple[str, int]:
    if not isinstance(value, dict):
        return ("", 0)
    title = re.sub(r"\s+", "", str(value.get("title") or "").strip().lower())
    level = int(RAG_DB_Document.coerce_page_number(value.get("level")) or 0)
    return (title, level)


def _normalize_catalog_items(values: Sequence[Any]) -> List[Dict[str, Any]]:
    ordered: Dict[tuple[str, int], Dict[str, Any]] = {}
    for order, item in enumerate(list(values or [])):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        page = int(RAG_DB_Document.coerce_page_number(item.get("page")) or 0)
        level = max(1, int(RAG_DB_Document.coerce_page_number(item.get("level")) or 1))
        if not title or page <= 0:
            continue
        row = {
            "title": title[:160],
            "page": page,
            "level": level,
            "order": int(item.get("order", order) or order),
        }
        key = _catalog_item_key(row)
        if not key[0]:
            continue
        old = ordered.get(key)
        if old is None or int(row["page"]) > int(old.get("page") or 0):
            ordered[key] = row
    return sorted(
        ordered.values(),
        key=lambda row: (
            int(row.get("page") or 0),
            int(row.get("level") or 1),
            int(row.get("order") or 0),
            str(row.get("title") or ""),
        ),
    )


def _prefer_richer_catalog(primary: Sequence[Any], secondary: Sequence[Any]) -> List[Dict[str, Any]]:
    primary_items = _normalize_catalog_items(primary)
    secondary_items = _normalize_catalog_items(secondary)
    if not primary_items:
        return secondary_items
    if not secondary_items:
        return primary_items

    primary_keys = {_catalog_item_key(item) for item in primary_items}
    secondary_keys = {_catalog_item_key(item) for item in secondary_items}
    secondary_only = secondary_keys - primary_keys
    if len(secondary_items) > len(primary_items) or len(secondary_only) >= max(3, len(primary_items) // 5):
        return secondary_items
    return primary_items


def _append_missing_tail_pages(primary_pages: Sequence[str], secondary_pages: Sequence[str]) -> List[str]:
    base = [str(item or "").strip() for item in list(primary_pages or [])]
    reference = [str(item or "").strip() for item in list(secondary_pages or [])]
    if not base:
        return reference
    if len(reference) <= len(base):
        return base
    tail = reference[len(base):]
    if not any(str(item or "").strip() for item in tail):
        return base
    return base + tail


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


def _extract_docx_markdown_with_mammoth(file_path: str) -> str:
    try:
        import mammoth  # type: ignore

        with open(file_path, "rb") as handle:
            result = mammoth.convert_to_markdown(handle)
        markdown = str(getattr(result, "value", "") or "").strip()
        return markdown
    except Exception:
        return ""


def _convert_office_to_docx(file_path: str) -> str:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return ""
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            file_path,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            docx_path = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.docx")
            if not os.path.isfile(docx_path):
                candidates = [
                    os.path.join(tmp_dir, name)
                    for name in os.listdir(tmp_dir)
                    if name.lower().endswith(".docx")
                ]
                if not candidates:
                    return ""
                docx_path = sorted(candidates)[0]
            markdown = _extract_docx_markdown_with_mammoth(docx_path)
            return markdown
        except Exception as exc:
            logger.warning("libreoffice doc->docx conversion failed for %s: %s", file_path, exc)
            return ""


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


def _extract_office_text_by_pdf_pages(
    file_path: str,
    *,
    include_images: bool = True,
) -> tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return "", 0, [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            office,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            tmp_dir,
            file_path,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_pdf = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.pdf")
            pdf_path = expected_pdf
            if not os.path.isfile(pdf_path):
                candidates = [
                    os.path.join(tmp_dir, name)
                    for name in os.listdir(tmp_dir)
                    if name.lower().endswith(".pdf")
                ]
                if not candidates:
                    return "", 0, [], []
                pdf_path = sorted(candidates)[0]

            try:
                import fitz  # type: ignore

                page_texts: List[str] = []
                page_layouts: List[Dict[str, Any]] = []
                native_catalog: List[Dict[str, Any]] = []

                def _clean_text(value: str) -> str:
                    return re.sub(r"\s+", " ", str(value or "")).strip()

                def _looks_like_page_number(value: str) -> bool:
                    text = _clean_text(value)
                    if not text:
                        return False
                    return bool(
                        re.fullmatch(
                            r"(?:第\s*\d{1,4}\s*页(?:\s*/\s*共\s*\d{1,4}\s*页)?|\d{1,4}|\d{1,4}\s*/\s*\d{1,4}|[IVXLCDM]{1,8})",
                            text,
                            flags=re.IGNORECASE,
                        )
                    )

                with fitz.open(pdf_path) as pdf_doc:
                    try:
                        toc_rows = list(pdf_doc.get_toc(simple=True) or [])
                    except Exception:
                        toc_rows = []
                    for row in toc_rows:
                        if not isinstance(row, (list, tuple)) or len(row) < 3:
                            continue
                        try:
                            level = int(row[0])
                            title = str(row[1] or "").strip()
                            page_no = int(row[2])
                        except Exception:
                            continue
                        if not title or page_no <= 0:
                            continue
                        native_catalog.append({"title": title[:160], "page": page_no, "level": max(1, min(level, 6))})

                    for page_idx in range(int(pdf_doc.page_count or 0)):
                        page = pdf_doc.load_page(page_idx)
                        page_dict_raw = page.get_text("dict")
                        page_dict = page_dict_raw if isinstance(page_dict_raw, dict) else {}
                        page_height = float(getattr(page.rect, "height", 0.0) or 0.0)
                        header_cut = page_height * 0.12 if page_height > 0 else 0.0
                        footer_cut = page_height * 0.88 if page_height > 0 else 0.0

                        headers: List[str] = []
                        footers: List[str] = []
                        body_lines: List[str] = []
                        images: List[Any] = []
                        page_number = ""

                        if include_images:
                            seen_xrefs: Set[int] = set()
                            for image_meta in page.get_images(full=True):
                                if not image_meta:
                                    continue
                                xref = int(image_meta[0] or 0)
                                if xref <= 0 or xref in seen_xrefs:
                                    continue
                                seen_xrefs.add(xref)
                                try:
                                    image_info = pdf_doc.extract_image(xref)
                                except Exception:
                                    image_info = None
                                if not isinstance(image_info, dict):
                                    continue
                                image_bytes = image_info.get("image")
                                if not image_bytes:
                                    continue
                                ext = str(image_info.get("ext") or "png").strip().lower() or "png"
                                filename = f"page-{page_idx + 1:04d}-xref-{xref}.{ext}"
                                images.append(
                                    ImageAsset.from_bytes(
                                        data=bytes(image_bytes),
                                        filename=filename,
                                        media_type=str(image_info.get("smask") or "") and f"image/{ext}" or f"image/{ext}",
                                        width=int(image_info.get("width") or 0),
                                        height=int(image_info.get("height") or 0),
                                        source=f"{os.path.basename(pdf_path)}#page={page_idx + 1};xref={xref}",
                                        page=page_idx + 1,
                                    )
                                )

                        for block in list(page_dict.get("blocks") or []):
                            block_type = int(block.get("type", 0) or 0)
                            bbox = block.get("bbox") or [0, 0, 0, 0]
                            y0 = float(bbox[1] if len(bbox) > 1 else 0.0)
                            y1 = float(bbox[3] if len(bbox) > 3 else 0.0)

                            if block_type == 1:
                                continue

                            if block_type != 0:
                                continue

                            spans: List[str] = []
                            for line in list(block.get("lines") or []):
                                line_parts: List[str] = []
                                for span in list(line.get("spans") or []):
                                    text = _clean_text(span.get("text") or "")
                                    if text:
                                        line_parts.append(text)
                                if line_parts:
                                    spans.append(" ".join(line_parts))
                            block_text = _clean_text("\n".join(spans))
                            if not block_text:
                                continue

                            if y1 <= header_cut:
                                headers.append(block_text)
                                continue
                            if y0 >= footer_cut:
                                footers.append(block_text)
                                if (not page_number) and _looks_like_page_number(block_text):
                                    page_number = block_text
                                continue

                            body_lines.append(block_text)

                        body_text = "\n".join(line for line in body_lines if line).strip()
                        page_texts.append(body_text)
                        page_layouts.append(
                            {
                                "page": page_idx + 1,
                                "headers": headers,
                                "footers": footers,
                                "page_number": page_number,
                                "images": _dedupe_image_items(images),
                            }
                        )
                if page_texts:
                    return "\n\f\n".join(page_texts).strip(), len(page_texts), page_layouts, native_catalog
            except Exception:
                pass

            try:
                import pymupdf4llm  # type: ignore
            except Exception:
                return "", 0, [], []

            chunks = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                extract_words=True,
                show_progress=False,
            )
            if not isinstance(chunks, list):
                return "", 0, [], []
            page_texts = []
            page_layouts: List[Dict[str, Any]] = []
            for item in chunks:
                if isinstance(item, dict):
                    page_texts.append(str(item.get("text") or ""))
                else:
                    page_texts.append(str(item or ""))
                page_layouts.append({"page": len(page_layouts) + 1, "headers": [], "footers": [], "page_number": "", "images": []})
            if not page_texts:
                return "", 0, [], []
            return "\n\f\n".join(page_texts).strip(), len(page_texts), page_layouts, []
        except Exception as exc:
            logger.warning("libreoffice pdf-page extraction failed for %s: %s", file_path, exc)
            return "", 0, [], []


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


def _docx_paragraph_has_rendered_page_break(paragraph: ET.Element, ns: Dict[str, str]) -> bool:
    for br in paragraph.findall(".//w:br", ns):
        br_type = str(br.attrib.get(f"{{{ns['w']}}}type") or "").strip().lower()
        if br_type == "page":
            return True
    if paragraph.find(".//w:lastRenderedPageBreak", ns) is not None:
        return True
    return False


def _docx_paragraph_has_section_page_break(paragraph: ET.Element, ns: Dict[str, str]) -> bool:
    sect = paragraph.find("w:pPr/w:sectPr", ns)
    if sect is None:
        return False
    type_node = sect.find("w:type", ns)
    if type_node is None:
        return False
    val = str(type_node.attrib.get(f"{{{ns['w']}}}val") or "").strip().lower()
    return val in {"nextpage", "oddpage", "evenpage"}


def _docx_paragraph_page_break_count(paragraph: ET.Element, ns: Dict[str, str]) -> int:
    count = 0
    for br in paragraph.findall(".//w:br", ns):
        br_type = str(br.attrib.get(f"{{{ns['w']}}}type") or "").strip().lower()
        if br_type == "page":
            count += 1
    count += len(paragraph.findall(".//w:lastRenderedPageBreak", ns))
    if _docx_paragraph_has_section_page_break(paragraph, ns):
        count += 1
    return count


def _docx_text_heading_level(text: str) -> Optional[int]:
    stripped = str(text or "").strip()
    if not stripped or len(stripped) > 96:
        return None
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇]", stripped):
        return 1
    if re.match(r"^附录[一二三四五六七八九十百千万A-Za-z0-9]+", stripped):
        return 1
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", stripped)
    if not match:
        return None
    return max(1, min(match.group(1).count(".") + 1, 6))


def _docx_asset_output_dir(file_path: str, namespace: str) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    digest = hashlib.md5(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    out_dir = os.path.join(repo_root, "tmp", "doc_tree_assets", digest, namespace)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _extract_docx_relationship_targets(archive: zipfile.ZipFile) -> Dict[str, str]:
    try:
        rels_xml = archive.read("word/_rels/document.xml.rels")
    except Exception:
        return {}
    ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    try:
        root = ET.fromstring(rels_xml)
    except Exception:
        return {}
    rel_targets: Dict[str, str] = {}
    for rel in root.findall("rel:Relationship", ns):
        rel_id = str(rel.attrib.get("Id") or "").strip()
        target = str(rel.attrib.get("Target") or "").strip()
        if rel_id and target:
            rel_targets[rel_id] = target
    return rel_targets


def _copy_docx_media_asset(
    archive: zipfile.ZipFile,
    target: str,
    output_dir: str,
    cache: Dict[str, ImageAsset],
) -> Optional[ImageAsset]:
    normalized = os.path.normpath(os.path.join("word", str(target or "").lstrip("/"))).replace("\\", "/")
    if normalized in cache:
        return cache[normalized]
    if not normalized.startswith("word/"):
        return None
    try:
        data = archive.read(normalized)
    except Exception:
        return None
    base_name = os.path.basename(normalized)
    if not base_name:
        return None
    asset = ImageAsset.from_bytes(
        data=data,
        filename=base_name,
        source=normalized,
    )
    cache[normalized] = asset
    return asset


def _extract_docx_images_from_element(
    element: ET.Element,
    archive: zipfile.ZipFile,
    rel_targets: Dict[str, str],
    output_dir: str,
    cache: Dict[str, ImageAsset],
    ns: Dict[str, str],
) -> List[ImageAsset]:
    rel_ids: List[str] = []
    for blip in element.findall(".//a:blip", ns):
        embed = str(blip.attrib.get(f"{{{ns['r']}}}embed") or "").strip()
        if embed:
            rel_ids.append(embed)
    for image_data in element.findall(".//v:imagedata", ns):
        rel_id = str(image_data.attrib.get(f"{{{ns['r']}}}id") or "").strip()
        if rel_id:
            rel_ids.append(rel_id)

    image_paths: List[ImageAsset] = []
    seen: Set[str] = set()
    for rel_id in rel_ids:
        target = str(rel_targets.get(rel_id) or "").strip()
        if not target:
            continue
        image_path = _copy_docx_media_asset(archive, target, output_dir, cache)
        if image_path is None:
            continue
        key = image_path.asset_id
        if key in seen:
            continue
        seen.add(key)
        image_paths.append(image_path)
    return image_paths


def _extract_docx_run_sizes(paragraph: ET.Element, ns: Dict[str, str]) -> List[int]:
    run_sizes: List[int] = []
    for run in paragraph.findall(".//w:r", ns):
        sz_node = run.find("w:rPr/w:sz", ns)
        if sz_node is None:
            continue
        raw = str(sz_node.attrib.get(f"{{{ns['w']}}}val") or "").strip()
        if raw.isdigit():
            run_sizes.append(int(raw))
    return run_sizes


def _docx_table_to_markdown(table: ET.Element, ns: Dict[str, str]) -> str:
    rows: List[List[str]] = []
    for tr in table.findall("w:tr", ns):
        row: List[str] = []
        for tc in tr.findall("w:tc", ns):
            parts: List[str] = []
            for paragraph in tc.findall("w:p", ns):
                text = _extract_docx_text_from_paragraph(paragraph, ns)
                if text:
                    parts.append(text)
            row.append(" ".join(parts).strip())
        if any(cell for cell in row):
            rows.append(row)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    out = ["| " + " | ".join(cell.replace("|", "\\|") for cell in normalized_rows[0]) + " |"]
    out.append("| " + " | ".join(["---"] * width) + " |")
    for row in normalized_rows[1:]:
        out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(out)


def _extract_docx_structured_payload(file_path: str) -> Dict[str, Any]:
    native_catalog: List[Dict[str, Any]] = []
    style_catalog: List[Dict[str, Any]] = []
    font_catalog: List[Dict[str, Any]] = []

    def _normalize_title(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    def _roman_to_int(token: str) -> int:
        mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        value = 0
        prev = 0
        for ch in reversed(str(token or "").upper()):
            cur = mapping.get(ch, 0)
            if cur < prev:
                value -= cur
            else:
                value += cur
                prev = cur
        return value

    def _parse_catalog_tail_page(text: str) -> tuple[str, Optional[int]]:
        raw = str(text or "").strip()
        if not raw:
            return "", None
        match = re.match(r"^(.{1,200}?)(?:\t+|[·•.]{2,}|\s{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*$", raw, flags=re.IGNORECASE)
        if not match:
            match = re.match(r"^(.{1,200}?)\s+(\d{1,4}|[IVXLCDM]{1,8})\s*$", raw, flags=re.IGNORECASE)
        if not match:
            return raw[:160], None
        title = str(match.group(1) or "").strip()[:160]
        page_token = str(match.group(2) or "").strip()
        if re.fullmatch(r"\d{1,4}", page_token):
            return title, int(page_token)
        if re.fullmatch(r"[IVXLCDM]{1,8}", page_token, flags=re.IGNORECASE):
            roman_page = _roman_to_int(page_token)
            return title, roman_page if roman_page > 0 else None
        return title, None

    paragraph_rows: List[Dict[str, Any]] = []
    pages: List[List[str]] = [[]]
    page_assets: List[Dict[str, Any]] = [{"images": []}]
    structured_sections: List[Dict[str, Any]] = []

    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            document_xml = archive.read("word/document.xml")
            styles_xml: Optional[bytes]
            try:
                styles_xml = archive.read("word/styles.xml")
            except Exception:
                styles_xml = None
            rel_targets = _extract_docx_relationship_targets(archive)
            asset_output_dir = _docx_asset_output_dir(file_path, "office")
            asset_cache: Dict[str, ImageAsset] = {}

            ns = {
                "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
                "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "v": "urn:schemas-microsoft-com:vml",
            }
        style_name_by_id: Dict[str, str] = {}
        if styles_xml:
            styles_root = ET.fromstring(styles_xml)
            for style in styles_root.findall(".//w:style", ns):
                style_id = str(style.attrib.get(f"{{{ns['w']}}}styleId") or "").strip()
                if not style_id:
                    continue
                name_node = style.find("w:name", ns)
                if name_node is not None:
                    style_name = str(name_node.attrib.get(f"{{{ns['w']}}}val") or "").strip()
                    if style_name:
                        style_name_by_id[style_id] = style_name

        with zipfile.ZipFile(file_path, "r") as archive:
            doc_root = ET.fromstring(document_xml)
            body = doc_root.find("w:body", ns)

            section_stack: List[Dict[str, Any]] = []

            def _unique_paths(values: List[str]) -> List[str]:
                unique: List[str] = []
                seen_paths: Set[str] = set()
                for item in values:
                    text = str(item or "").strip()
                    if not text or text in seen_paths:
                        continue
                    seen_paths.add(text)
                    unique.append(text)
                return unique

            def _start_section(title: str, level: int, page: int) -> None:
                cleaned_title = str(title or "").strip()[:160]
                if not cleaned_title:
                    return
                while section_stack and int(section_stack[-1].get("level") or 1) >= int(level):
                    section_stack.pop()
                path_parts = [str(item.get("title") or "").strip() for item in section_stack if str(item.get("title") or "").strip()]
                path_parts.append(cleaned_title)
                section = {
                    "title": cleaned_title,
                    "level": max(1, min(int(level or 1), 6)),
                    "page": int(page or 1),
                    "path": " > ".join(path_parts),
                    "blocks": [],
                    "images": [],
                    "order": len(structured_sections),
                }
                structured_sections.append(section)
                section_stack.append(section)

            def _append_block(text: str) -> None:
                block = str(text or "").strip()
                if not block or not section_stack:
                    return
                section_stack[-1]["blocks"].append(block)

            def _append_images(images: List[Any]) -> None:
                if not images:
                    return
                current_images = page_assets[-1].setdefault("images", [])
                current_images.extend(images)
                if section_stack:
                    section_stack[-1]["images"].extend(images)

            page_idx = 1
            for child in list(body or []):
                tag = child.tag.rsplit("}", 1)[-1]
                child_images = _extract_docx_images_from_element(
                    child,
                    archive,
                    rel_targets,
                    asset_output_dir,
                    asset_cache,
                    ns,
                )
                _append_images(child_images)

                if tag == "p":
                    paragraph = child
                    text = _extract_docx_text_from_paragraph(paragraph, ns)
                    if text:
                        pages[-1].append(text)

                    p_style = paragraph.find("w:pPr/w:pStyle", ns)
                    style_id = ""
                    if p_style is not None:
                        style_id = str(p_style.attrib.get(f"{{{ns['w']}}}val") or "").strip()
                    style_name = style_name_by_id.get(style_id, "")
                    run_sizes = _extract_docx_run_sizes(paragraph, ns)
                    max_sz = max(run_sizes) if run_sizes else 0

                    if text:
                        paragraph_rows.append(
                            {
                                "text": text,
                                "style_id": style_id,
                                "style_name": style_name,
                                "max_sz": max_sz,
                                "page": page_idx,
                            }
                        )

                    lower_style = f"{style_id} {style_name}".lower()
                    heading_level = _docx_heading_level_from_style(style_id, style_name)
                    if heading_level is None:
                        heading_level = _docx_text_heading_level(text)
                    title_candidate, catalog_page = _parse_catalog_tail_page(text)
                    is_catalogue_paragraph = bool(catalog_page is not None and title_candidate)
                    if text and heading_level is not None and "toc" not in lower_style and "目录" not in lower_style and not is_catalogue_paragraph:
                        _start_section(text, heading_level, page_idx)
                    elif text:
                        _append_block(text)

                    page_break_count = _docx_paragraph_page_break_count(paragraph, ns)
                    for _ in range(page_break_count):
                        page_idx += 1
                        pages.append([])
                        page_assets.append({"images": []})
                    continue

                if tag != "tbl":
                    continue

                table_markdown = _docx_table_to_markdown(child, ns)
                if table_markdown:
                    pages[-1].append(table_markdown)
                    _append_block(table_markdown)

                table_break_count = 0
                for paragraph in child.findall(".//w:p", ns):
                    table_break_count += _docx_paragraph_page_break_count(paragraph, ns)
                for _ in range(table_break_count):
                    page_idx += 1
                    pages.append([])
                    page_assets.append({"images": []})

        while pages and not pages[-1]:
            pages.pop()
            if page_assets:
                page_assets.pop()
        if not pages:
            pages = [[]]
        if not page_assets:
            page_assets = [{"images": []}]

        for row in paragraph_rows:
            style_id = str(row.get("style_id") or "")
            if not style_id:
                continue
            text = str(row.get("text") or "")
            style_name = style_name_by_id.get(style_id, "")
            lower_style = f"{style_id} {style_name}".lower()

            if "toc" in lower_style or "目录" in lower_style:
                title, page = _parse_catalog_tail_page(text)
                if title and page and page > 0:
                    native_catalog.append(
                        {
                            "title": title,
                            "page": page,
                            "level": _docx_heading_level_from_style(style_id, style_name) or 1,
                        }
                    )

        page_by_title: Dict[str, int] = {}
        for row in native_catalog:
            key = _normalize_title(str(row.get("title") or ""))
            page = int(row.get("page") or 0)
            if not key or page <= 0:
                continue
            old = page_by_title.get(key)
            if old is None or page < old:
                page_by_title[key] = page

        for row in paragraph_rows:
            style_id = str(row.get("style_id") or "")
            if not style_id:
                continue
            text = str(row.get("text") or "")
            style_name = str(row.get("style_name") or "")
            lower_style = f"{style_id} {style_name}".lower()
            if "toc" in lower_style or "目录" in lower_style:
                continue

            normalized = _normalize_title(text)
            mapped_page = page_by_title.get(normalized) or int(row.get("page") or 1)
            heading_level = _docx_heading_level_from_style(style_id, style_name)
            if heading_level is not None:
                style_catalog.append(
                    {
                        "title": text[:160],
                        "page": mapped_page,
                        "level": heading_level,
                    }
                )

            max_sz = int(row.get("max_sz") or 0)
            if max_sz >= 28 and len(text.strip()) <= 120:
                inferred_level = 1 if max_sz >= 40 else (2 if max_sz >= 32 else 3)
                font_catalog.append(
                    {
                        "title": text[:160],
                        "page": mapped_page,
                        "level": inferred_level,
                    }
                )

    except Exception as exc:
        logger.debug("Failed to extract DOCX structured payload for %s: %s", file_path, exc)

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

    page_texts = ["\n".join(lines).strip() for lines in pages]
    text_with_breaks = "\n\f\n".join(page_texts).strip()
    has_breaks = len(page_texts) > 1
    catalogs = {
        "native_catalog": _dedupe(native_catalog),
        "style_catalog": _dedupe(style_catalog),
        "font_catalog": _dedupe(font_catalog),
    }
    structured_section_payload: List[Dict[str, Any]] = []
    for section in structured_sections:
        title = str(section.get("title") or "").strip()
        if not title:
            continue
        blocks = [str(item or "").strip() for item in list(section.get("blocks") or []) if str(item or "").strip()]
        images: List[Any] = []
        seen_images: Set[str] = set()
        for item in list(section.get("images") or []):
            normalized = _normalize_image_value(item)
            if normalized is None:
                continue
            key = _image_value_key(normalized)
            if not key or key in seen_images:
                continue
            seen_images.add(key)
            images.append(normalized)
        if not blocks and not images:
            continue
        structured_section_payload.append(
            {
                "title": title,
                "level": int(section.get("level") or 1),
                "page": int(section.get("page") or 1),
                "path": str(section.get("path") or title),
                "order": int(section.get("order") or 0),
                "blocks": blocks,
                "images": images,
            }
        )

    normalized_page_assets: List[Dict[str, Any]] = []
    for item in page_assets:
        normalized_page_assets.append(
            {
                "images": _dedupe_image_items(list(item.get("images") or [])),
            }
        )

    return {
        "text_with_breaks": text_with_breaks,
        "has_breaks": has_breaks,
        "catalogs": catalogs,
        "page_markdowns": page_texts,
        "page_assets": normalized_page_assets,
        "structured_sections": structured_section_payload,
    }


def _extract_docx_structure_from_xml(file_path: str) -> Tuple[str, bool, Dict[str, List[Dict[str, Any]]]]:
    payload = _extract_docx_structured_payload(file_path)
    return (
        str(payload.get("text_with_breaks") or ""),
        bool(payload.get("has_breaks")),
        dict(payload.get("catalogs") or {}),
    )


def _extract_docx_catalog_metadata(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    _, _, catalogs = _extract_docx_structure_from_xml(file_path)
    return catalogs


def _extract_doc_catalog_metadata(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return {"native_catalog": [], "style_catalog": [], "font_catalog": []}

    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            file_path,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_docx = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.docx")
            docx_path = expected_docx
            if not os.path.isfile(docx_path):
                candidates = [
                    os.path.join(tmp_dir, name)
                    for name in os.listdir(tmp_dir)
                    if name.lower().endswith(".docx")
                ]
                if not candidates:
                    return {"native_catalog": [], "style_catalog": [], "font_catalog": []}
                docx_path = sorted(candidates)[0]
            return _extract_docx_catalog_metadata(docx_path)
        except Exception:
            return {"native_catalog": [], "style_catalog": [], "font_catalog": []}


def _extract_doc_text_via_converted_docx(file_path: str) -> Tuple[str, bool, Dict[str, List[Dict[str, Any]]]]:
    payload = _extract_doc_structured_payload_via_converted_docx(file_path)
    return (
        str(payload.get("text_with_breaks") or ""),
        bool(payload.get("has_breaks")),
        dict(payload.get("catalogs") or {}),
    )


def _extract_doc_structured_payload_via_converted_docx(file_path: str) -> Dict[str, Any]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        return {
            "text_with_breaks": "",
            "has_breaks": False,
            "catalogs": {"native_catalog": [], "style_catalog": [], "font_catalog": []},
            "page_assets": [],
            "structured_sections": [],
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            file_path,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_docx = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.docx")
            docx_path = expected_docx
            if not os.path.isfile(docx_path):
                candidates = [
                    os.path.join(tmp_dir, name)
                    for name in os.listdir(tmp_dir)
                    if name.lower().endswith(".docx")
                ]
                if not candidates:
                    return {
                        "text_with_breaks": "",
                        "has_breaks": False,
                        "catalogs": {"native_catalog": [], "style_catalog": [], "font_catalog": []},
                        "page_assets": [],
                        "structured_sections": [],
                    }
                docx_path = sorted(candidates)[0]

            return _extract_docx_structured_payload(docx_path)
        except Exception:
            return {
                "text_with_breaks": "",
                "has_breaks": False,
                "catalogs": {"native_catalog": [], "style_catalog": [], "font_catalog": []},
                "page_assets": [],
                "structured_sections": [],
            }


def load_single_file_document(file_path: str, supported_extensions: Set[str]) -> Optional[Document]:
    from llama_index.readers.file import DocxReader, PandasExcelReader
    from llama_index.core import SimpleDirectoryReader

    file_path = os.path.abspath(file_path)
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
            doc_payload = _extract_doc_structured_payload_via_converted_docx(file_path)
            xml_text = str(doc_payload.get("text_with_breaks") or "")
            xml_has_breaks = bool(doc_payload.get("has_breaks"))
            doc_catalogs = dict(doc_payload.get("catalogs") or {})
            doc_structured_sections = list(doc_payload.get("structured_sections") or [])
            doc_page_assets = list(doc_payload.get("page_assets") or [])
            uno_text, uno_has_breaks = _extract_office_text_by_uno_pages(file_path)
            office_text, has_native_breaks = _extract_office_text_with_page_breaks(file_path)
            pdf_text, pdf_page_count, pdf_page_layouts, pdf_native_catalog = _extract_office_text_by_pdf_pages(
                file_path,
                include_images=False,
            )
            uno_pages = _split_native_pages(uno_text) if uno_has_breaks and uno_text else []
            pdf_reference_pages = _split_native_pages(pdf_text)
            uno_pdf_gap = max(0, len(pdf_reference_pages) - len(uno_pages))
            use_pdf_tail_fallback = bool(
                uno_pages
                and pdf_reference_pages
                and uno_pdf_gap >= max(8, max(1, len(uno_pages) // 5))
            )
            effective_uno_pages = _append_missing_tail_pages(uno_pages, pdf_reference_pages) if use_pdf_tail_fallback else list(uno_pages)
            effective_uno_text = "\n\f\n".join(effective_uno_pages).strip() if effective_uno_pages else uno_text
            native_page_count = _require_libreoffice_physical_page_count(
                file_path,
                ext=".doc",
                fallback_counts=[
                    len(effective_uno_pages),
                    len(xml_text.split("\f")) if xml_has_breaks and xml_text else 0,
                    len(office_text.split("\f")) if has_native_breaks and office_text else 0,
                    int(pdf_page_count or 0),
                ],
            )
            effective_native_page_count = max(int(native_page_count or 0), len(effective_uno_pages), int(pdf_page_count or 0))
            doc_native_catalog = _prefer_richer_catalog(doc_catalogs.get("native_catalog") or [], pdf_native_catalog)
            doc_page_layout = _merge_page_layouts(
                doc_page_assets,
                pdf_page_layouts,
                total_pages=int(effective_native_page_count or pdf_page_count or 0),
            )
            doc_common_metadata = {
                "file_name": file_path,
                "source_extension": ".doc",
                "native_page_count": int(effective_native_page_count or 0),
                "native_catalog": doc_native_catalog,
                "style_catalog": doc_catalogs.get("style_catalog") or [],
                "font_catalog": doc_catalogs.get("font_catalog") or [],
                "structured_sections": doc_structured_sections,
                "structured_page_assets": doc_page_assets,
                "page_layout": doc_page_layout,
                "pdf_reference_catalog": list(pdf_native_catalog or []),
                "pdf_reference_pages": pdf_reference_pages,
                "txt_reference_pages": _split_native_pages(office_text),
            }
            if uno_text and uno_has_breaks:
                return Document(
                    text=effective_uno_text,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice-uno-pages+pdf-tail-fallback" if use_pdf_tail_fallback else "libreoffice-uno-pages",
                        "native_pagination": True,
                    },
                    doc_id=file_path,
                )
            if xml_text and xml_has_breaks:
                return Document(
                    text=xml_text,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice-docx-xml",
                        "native_pagination": True,
                    },
                    doc_id=file_path,
                )

            if office_text:
                return Document(
                    text=office_text,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice-txt",
                        "native_pagination": bool(has_native_breaks),
                    },
                    doc_id=file_path,
                )

            if xml_text:
                return Document(
                    text=xml_text,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice-docx-xml",
                        "native_pagination": bool(xml_has_breaks),
                    },
                    doc_id=file_path,
                )

            markdown = _convert_office_to_docx(file_path)
            if markdown:
                return Document(
                    text=markdown,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice+mammoth-markdown",
                        "text_format": "markdown",
                        "native_pagination": False,
                    },
                    doc_id=file_path,
                )

            text = _extract_legacy_doc_text(file_path)
            if not text:
                logger.warning("Skipping .doc file with unsupported parser: %s", file_path)
                return None
            return Document(
                text=text,
                metadata={
                    **doc_common_metadata,
                    "native_pagination": ("\f" in text),
                },
                doc_id=file_path,
            )

        if ext == ".docx":
            docx_payload = _extract_docx_structured_payload(file_path)
            xml_text = str(docx_payload.get("text_with_breaks") or "")
            xml_has_breaks = bool(docx_payload.get("has_breaks"))
            docx_catalogs = dict(docx_payload.get("catalogs") or {})
            docx_structured_sections = list(docx_payload.get("structured_sections") or [])
            docx_page_assets = list(docx_payload.get("page_assets") or [])
            uno_text, uno_has_breaks = _extract_office_text_by_uno_pages(file_path)
            office_text, has_native_breaks = _extract_office_text_with_page_breaks(file_path)
            native_page_count = _require_libreoffice_physical_page_count(
                file_path,
                ext=".docx",
                fallback_counts=[
                    len(uno_text.split("\f")) if uno_has_breaks and uno_text else 0,
                    len(xml_text.split("\f")) if xml_has_breaks and xml_text else 0,
                    len(office_text.split("\f")) if has_native_breaks and office_text else 0,
                ],
            )
            docx_common_metadata = {
                "file_name": file_path,
                "source_extension": ".docx",
                "native_page_count": int(native_page_count or 0),
                "native_catalog": docx_catalogs.get("native_catalog") or [],
                "style_catalog": docx_catalogs.get("style_catalog") or [],
                "font_catalog": docx_catalogs.get("font_catalog") or [],
                "structured_sections": docx_structured_sections,
                "structured_page_assets": docx_page_assets,
                "page_layout": _merge_page_layouts(docx_page_assets, [], total_pages=int(native_page_count or 0)),
                "txt_reference_pages": _split_native_pages(office_text),
            }
            if uno_text and uno_has_breaks:
                return Document(
                    text=uno_text,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": True,
                        "docx_parser": "libreoffice-uno-pages",
                    },
                    doc_id=file_path,
                )
            if xml_text and xml_has_breaks:
                return Document(
                    text=xml_text,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": True,
                        "docx_parser": "docx-xml",
                    },
                    doc_id=file_path,
                )

            if office_text:
                return Document(
                    text=office_text,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": bool(has_native_breaks),
                        "docx_parser": "libreoffice-txt",
                    },
                    doc_id=file_path,
                )

            if xml_text:
                return Document(
                    text=xml_text,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": bool(xml_has_breaks),
                        "docx_parser": "docx-xml",
                    },
                    doc_id=file_path,
                )

            markdown = _extract_docx_markdown_with_mammoth(file_path)
            if markdown:
                return Document(
                    text=markdown,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": has_native_breaks,
                        "docx_parser": "mammoth-markdown",
                        "text_format": "markdown",
                    },
                    doc_id=file_path,
                )

            if office_text:
                return Document(
                    text=office_text,
                    metadata={
                        **docx_common_metadata,
                        "native_pagination": has_native_breaks,
                        "docx_parser": "libreoffice-txt",
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
        metadata["file_name"] = file_path
        metadata.setdefault("source_extension", ext)
        if ext == ".docx":
            docx_payload = _extract_docx_structured_payload(file_path)
            docx_catalogs = dict(docx_payload.get("catalogs") or {})
            metadata.setdefault("native_catalog", docx_catalogs.get("native_catalog") or [])
            metadata.setdefault("style_catalog", docx_catalogs.get("style_catalog") or [])
            metadata.setdefault("font_catalog", docx_catalogs.get("font_catalog") or [])
            metadata.setdefault("structured_sections", list(docx_payload.get("structured_sections") or []))
            metadata.setdefault("structured_page_assets", list(docx_payload.get("page_assets") or []))
            metadata.setdefault("page_layout", list(docx_payload.get("page_assets") or []))
        return Document(
            text=first.text,
            metadata=metadata,
            doc_id=first.doc_id or file_path,
        )
    except Exception as exc:
        if _NATIVE_PAGE_COUNT_ERROR_PREFIX in str(exc):
            raise
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
        chunk_docs = list(getattr(rag_doc, "chunk_documents", []) or [])
        tree_markdown = ""
        try:
            tree_markdown = str(rag_doc.export_markdown_from_tree() or "")
        except Exception:
            tree_markdown = ""
        if not chunk_docs and not tree_markdown.strip():
            continue
        rag_docs.append(rag_doc)
    return rag_docs


def chunk_documents_from_rag_documents(rag_docs: Sequence[RAG_DB_Document]) -> List[Document]:
    docs: List[Document] = []
    for rag_doc in rag_docs:
        docs.extend(getattr(rag_doc, "chunk_documents", []) or [])
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

    if ext == ".pdf":
        return PDFRAGDocument(source_document, stable_doc_id)
    if ext == ".doc":
        return DocRAGDocument(source_document, stable_doc_id)
    if ext == ".docx":
        return DocxRAGDocument(source_document, stable_doc_id)
    if ext in {".xlsx", ".xls"}:
        return SpreadsheetRAGDocument(source_document, stable_doc_id)
    return TextRAGDocument(source_document, stable_doc_id)
