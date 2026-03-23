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
        cleaned_text = str(getattr(rag_doc, "cleaned_text", "") or "")
        if not chunk_docs and not cleaned_text:
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
