from __future__ import annotations

import os
import re
import logging
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Set

from llama_index.core import Document

from rag.document_doc import DocRAGDocument
from rag.document_docx import DocxRAGDocument
from rag.document_interface import RAG_DB_Document
from rag.document_pdf import PDFRAGDocument
from rag.document_spreadsheet import SpreadsheetRAGDocument
from rag.document_text import TextRAGDocument
from rag.pdf_reader import get_pdf_reader

logger = logging.getLogger(__name__)


def stable_doc_id(doc: Document) -> str:
    import hashlib

    source = (doc.metadata or {}).get("file_name") or (doc.text or "")[:200]
    return hashlib.md5(str(source).encode()).hexdigest()


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


def _extract_office_text_by_pdf_pages(file_path: str) -> tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                        images: List[str] = []
                        page_number = ""

                        for block in list(page_dict.get("blocks") or []):
                            block_type = int(block.get("type", 0) or 0)
                            bbox = block.get("bbox") or [0, 0, 0, 0]
                            y0 = float(bbox[1] if len(bbox) > 1 else 0.0)
                            y1 = float(bbox[3] if len(bbox) > 3 else 0.0)

                            if block_type == 1:
                                width = int(block.get("width") or 0)
                                height = int(block.get("height") or 0)
                                images.append(f"image[{len(images) + 1}] {width}x{height} @ ({int(y0)}-{int(y1)})")
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
                                "images": images,
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


def _extract_docx_catalog_metadata(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    native_catalog: List[Dict[str, Any]] = []
    style_catalog: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            document_xml = archive.read("word/document.xml")
            styles_xml: Optional[bytes] = None
            try:
                styles_xml = archive.read("word/styles.xml")
            except Exception:
                styles_xml = None

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        style_name_by_id: Dict[str, str] = {}
        if styles_xml:
            styles_root = ET.fromstring(styles_xml)
            for style in styles_root.findall('.//w:style', ns):
                style_id = str(style.attrib.get(f"{{{ns['w']}}}styleId") or "").strip()
                if not style_id:
                    continue
                name_node = style.find('w:name', ns)
                if name_node is not None:
                    style_name = str(name_node.attrib.get(f"{{{ns['w']}}}val") or "").strip()
                    if style_name:
                        style_name_by_id[style_id] = style_name

        doc_root = ET.fromstring(document_xml)
        for paragraph in doc_root.findall('.//w:body/w:p', ns):
            text = _extract_docx_text_from_paragraph(paragraph, ns)
            if not text:
                continue
            p_style = paragraph.find('w:pPr/w:pStyle', ns)
            style_id = ""
            if p_style is not None:
                style_id = str(p_style.attrib.get(f"{{{ns['w']}}}val") or "").strip()
            if not style_id:
                continue
            style_name = style_name_by_id.get(style_id, "")
            lower_style = f"{style_id} {style_name}".lower()

            toc_match = re.match(r"^(.{1,160}?)(?:\t+|[·•.\s]{2,})(\d{1,4})\s*$", text)
            if ("toc" in lower_style or "目录" in lower_style) and toc_match:
                page = int(toc_match.group(2))
                if page > 0:
                    native_catalog.append(
                        {
                            "title": toc_match.group(1).strip()[:160],
                            "page": page,
                            "level": _docx_heading_level_from_style(style_id, style_name) or 1,
                        }
                    )
                continue

            heading_level = _docx_heading_level_from_style(style_id, style_name)
            if heading_level is not None:
                style_catalog.append(
                    {
                        "title": text[:160],
                        "page": 1,
                        "level": heading_level,
                    }
                )
    except Exception as exc:
        logger.debug("Failed to extract DOCX structured catalog for %s: %s", file_path, exc)

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

    return {
        "native_catalog": _dedupe(native_catalog),
        "style_catalog": _dedupe(style_catalog),
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
            office_text, has_native_breaks = _extract_office_text_with_page_breaks(file_path)
            if office_text and has_native_breaks:
                return Document(
                    text=office_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".doc",
                        "doc_parser": "libreoffice-txt",
                        "native_pagination": True,
                    },
                    doc_id=file_path,
                )

            paged_text, page_count, page_layout, native_catalog = _extract_office_text_by_pdf_pages(file_path)
            if paged_text:
                return Document(
                    text=paged_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".doc",
                        "doc_parser": "libreoffice+pdf-page-chunks",
                        "native_pagination": True,
                        "native_page_count": int(page_count),
                        "page_layout": page_layout,
                        "native_catalog": native_catalog,
                    },
                    doc_id=file_path,
                )

            markdown = _convert_office_to_docx(file_path)
            if markdown:
                return Document(
                    text=markdown,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".doc",
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
                    "file_name": file_path,
                    "source_extension": ".doc",
                    "native_pagination": ("\f" in text),
                },
                doc_id=file_path,
            )

        if ext == ".docx":
            office_text, has_native_breaks = _extract_office_text_with_page_breaks(file_path)
            docx_catalogs = _extract_docx_catalog_metadata(file_path)
            if office_text and has_native_breaks:
                return Document(
                    text=office_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".docx",
                        "native_pagination": True,
                        "docx_parser": "libreoffice-txt",
                        "native_catalog": docx_catalogs.get("native_catalog") or [],
                        "style_catalog": docx_catalogs.get("style_catalog") or [],
                    },
                    doc_id=file_path,
                )

            paged_text, page_count, page_layout, native_catalog = _extract_office_text_by_pdf_pages(file_path)
            if paged_text:
                return Document(
                    text=paged_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".docx",
                        "native_pagination": True,
                        "native_page_count": int(page_count),
                        "docx_parser": "libreoffice+pdf-page-chunks",
                        "native_catalog": (docx_catalogs.get("native_catalog") or []) + list(native_catalog or []),
                        "style_catalog": docx_catalogs.get("style_catalog") or [],
                        "page_layout": page_layout,
                    },
                    doc_id=file_path,
                )

            markdown = _extract_docx_markdown_with_mammoth(file_path)
            if markdown:
                return Document(
                    text=markdown,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".docx",
                        "native_pagination": has_native_breaks,
                        "docx_parser": "mammoth-markdown",
                        "text_format": "markdown",
                        "native_catalog": docx_catalogs.get("native_catalog") or [],
                        "style_catalog": docx_catalogs.get("style_catalog") or [],
                    },
                    doc_id=file_path,
                )

            if office_text:
                return Document(
                    text=office_text,
                    metadata={
                        "file_name": file_path,
                        "source_extension": ".docx",
                        "native_pagination": has_native_breaks,
                        "docx_parser": "libreoffice-txt",
                        "native_catalog": docx_catalogs.get("native_catalog") or [],
                        "style_catalog": docx_catalogs.get("style_catalog") or [],
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
            docx_catalogs = _extract_docx_catalog_metadata(file_path)
            metadata.setdefault("native_catalog", docx_catalogs.get("native_catalog") or [])
            metadata.setdefault("style_catalog", docx_catalogs.get("style_catalog") or [])
        return Document(
            text=first.text,
            metadata=metadata,
            doc_id=first.doc_id or file_path,
        )
    except Exception as exc:
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
