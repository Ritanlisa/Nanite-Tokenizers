from __future__ import annotations

import os
import re
import glob
import hashlib
import logging
import shutil
import subprocess
import site
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
BuildProgressCallback = Callable[[str, Dict[str, Any]], None]


def _emit_build_progress(
    progress_callback: Optional[BuildProgressCallback],
    stage: str,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(str(stage), {str(key): value for key, value in payload.items()})
    except Exception as exc:
        logger.debug("Ignoring build progress callback failure (%s): %s", stage, exc)


def _expand_existing_paths(
    candidates: Sequence[str],
    *,
    want_dir: bool = False,
    want_file: bool = False,
    executable_only: bool = False,
) -> List[str]:
    resolved: List[str] = []
    seen: Set[str] = set()
    for raw in list(candidates or []):
        value = str(raw or "").strip()
        if not value:
            continue
        expanded = os.path.expanduser(value)
        matches = glob.glob(expanded) if any(char in expanded for char in "*?[") else [expanded]
        for match in matches:
            normalized = os.path.realpath(os.path.abspath(match))
            if normalized in seen:
                continue
            if want_dir and not os.path.isdir(normalized):
                continue
            if want_file and not os.path.isfile(normalized):
                continue
            if executable_only and not os.access(normalized, os.X_OK):
                continue
            seen.add(normalized)
            resolved.append(normalized)
    return resolved


def _libreoffice_program_dir(office: str) -> str:
    value = str(office or "").strip()
    if not value:
        return ""
    return os.path.dirname(os.path.realpath(os.path.abspath(value)))


def _libreoffice_writer_component_paths(office: str) -> List[str]:
    program_dir = _libreoffice_program_dir(office)
    if not program_dir or not os.path.isdir(program_dir):
        return []

    candidates: List[str] = []
    seen: Set[str] = set()
    for pattern in ("libsw*.so", "libsw*.dylib", "sw*.dll"):
        for path in sorted(glob.glob(os.path.join(program_dir, pattern))):
            normalized = os.path.realpath(os.path.abspath(path))
            name = os.path.basename(normalized).lower()
            if "writerperfect" in name:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
    return candidates


def _libreoffice_has_writer_support(office: str) -> bool:
    return bool(_libreoffice_writer_component_paths(office))


@lru_cache(maxsize=128)
def _probe_file_description(file_path: str) -> str:
    file_cmd = shutil.which("file")
    if not file_cmd:
        return ""
    try:
        proc = subprocess.run(
            [file_cmd, "-b", os.path.abspath(file_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or proc.stderr or "").strip()


def _probe_file_creator_application(file_path: str) -> str:
    description = _probe_file_description(file_path)
    match = re.search(r"Name of Creating Application:\s*([^,]+)", description, flags=re.IGNORECASE)
    if match is None:
        return ""
    return str(match.group(1) or "").strip()


def _probe_file_reported_page_count(file_path: str) -> int:
    description = _probe_file_description(file_path)
    match = re.search(r"Number of Pages:\s*(\d+)", description, flags=re.IGNORECASE)
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _build_office_document_failure_message(file_path: str, *, ext: str) -> str:
    office = _get_libreoffice_executable()
    creator_app = _probe_file_creator_application(file_path)
    reported_pages = _probe_file_reported_page_count(file_path)

    if ext == ".doc" and creator_app and "visio" in creator_app.lower():
        return (
            f"检测到该 {ext} 文件由 {creator_app} 创建，不是 Word 文档；"
            "当前项目仅支持 Word .doc/.docx，不支持将 Visio 复合文档按 Word 文档解析。"
        )

    if not office:
        return (
            f"无法获取 {ext} 文档的 LibreOffice 物理页数。"
            "请确认系统已安装 python3-uno 且 LibreOffice 可用；"
            "如为手动安装，可通过 LIBREOFFICE_PATH/SOFFICE_PATH 与 "
            "LIBREOFFICE_PROGRAM_PATH/UNO_PATH 指定路径；"
            "当前已禁用该场景下的自动回退，以避免错误分类。"
        )

    if not _libreoffice_has_writer_support(office):
        message = (
            f"已定位到 LibreOffice 可执行文件 {office}，但其安装缺少 Writer 核心组件，"
            "当前无法加载 Word 文档。"
            "请安装完整的 libreoffice-writer，或将 LIBREOFFICE_PATH 指向包含 Writer 组件的 soffice。"
        )
        if creator_app:
            message += f" 文件元数据显示创建程序为 {creator_app}。"
        if reported_pages > 0:
            message += f" `file` 元数据显示页数约为 {reported_pages}。"
        return message

    if creator_app:
        message = f"已定位到 LibreOffice，但仍无法加载该 {ext} 文档；文件元数据显示创建程序为 {creator_app}。"
        if reported_pages > 0:
            message += f" `file` 元数据显示页数约为 {reported_pages}。"
        message += " 这通常表示当前 LibreOffice 缺少相应导入过滤器，或该文件本身并非 Writer 可打开的文档。"
        return message

    return (
        f"无法获取 {ext} 文档的 LibreOffice 物理页数。"
        "请确认系统已安装 python3-uno 且 LibreOffice 可用；"
        "如为手动安装，可通过 LIBREOFFICE_PATH/SOFFICE_PATH 与 "
        "LIBREOFFICE_PROGRAM_PATH/UNO_PATH 指定路径；"
        "当前已禁用该场景下的自动回退，以避免错误分类。"
    )


@lru_cache(maxsize=1)
def _resolve_libreoffice_runtime() -> Tuple[str, Tuple[str, ...]]:
    env_executable_candidates: List[str] = []
    auto_executable_candidates: List[str] = []
    program_hints: List[str] = []

    for env_key in ("LIBREOFFICE_PATH", "SOFFICE_PATH"):
        value = str(os.environ.get(env_key) or "").strip()
        if value:
            env_executable_candidates.append(value)

    for env_key in ("LIBREOFFICE_PROGRAM_PATH", "UNO_PATH"):
        value = str(os.environ.get(env_key) or "").strip()
        if not value:
            continue
        program_hints.append(value)
        env_executable_candidates.extend(
            [
                os.path.join(value, "soffice"),
                os.path.join(value, "libreoffice"),
            ]
        )

    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            auto_executable_candidates.append(found)

    auto_executable_candidates.extend(
        [
            "/usr/bin/soffice",
            "/usr/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/usr/local/bin/libreoffice",
            "/snap/bin/libreoffice",
            "/snap/bin/soffice",
            "/opt/libreoffice*/program/soffice",
            "/opt/libreoffice*/program/libreoffice",
            "/usr/local/lib/libreoffice*/program/soffice",
            "/usr/local/lib/libreoffice*/program/libreoffice",
        ]
    )

    env_office_candidates = _expand_existing_paths(env_executable_candidates, want_file=True, executable_only=True)
    if env_office_candidates:
        office = env_office_candidates[0]
    else:
        auto_office_candidates = _expand_existing_paths(auto_executable_candidates, want_file=True, executable_only=True)
        office = next((path for path in auto_office_candidates if _libreoffice_has_writer_support(path)), "")
        if not office:
            office = auto_office_candidates[0] if auto_office_candidates else ""

    bridge_candidates: List[str] = [
        *program_hints,
        "/usr/lib/python3/dist-packages",
        "/usr/lib/libreoffice/program",
        "/usr/local/lib/libreoffice/program",
        "/opt/libreoffice/program",
        "/snap/libreoffice/current/usr/lib/libreoffice/program",
        "/var/lib/snapd/snap/libreoffice/current/usr/lib/libreoffice/program",
        "/opt/libreoffice*/program",
        "/usr/local/lib/libreoffice*/program",
    ]
    if office:
        executable_dir = os.path.dirname(office)
        bridge_candidates.extend(
            [
                executable_dir,
                os.path.join(os.path.dirname(executable_dir), "program"),
            ]
        )

    bridge_paths = tuple(_expand_existing_paths(bridge_candidates, want_dir=True))
    return office, bridge_paths


def _get_libreoffice_executable() -> str:
    office, _ = _resolve_libreoffice_runtime()
    return office


def _build_uno_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    office, bridge_paths = _resolve_libreoffice_runtime()

    merged_pythonpath: List[str] = []
    seen: Set[str] = set()
    for path in list(bridge_paths) + [part for part in str(env.get("PYTHONPATH") or "").split(os.pathsep) if part.strip()]:
        value = str(path or "").strip()
        if not value:
            continue
        normalized = os.path.realpath(os.path.abspath(os.path.expanduser(value))) if os.path.exists(value) else value
        if normalized in seen:
            continue
        seen.add(normalized)
        merged_pythonpath.append(normalized)
    if merged_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(merged_pythonpath)

    if office:
        office_dir = os.path.dirname(office)
        merged_path: List[str] = []
        seen_path: Set[str] = set()
        for path in [office_dir] + [part for part in str(env.get("PATH") or "").split(os.pathsep) if part.strip()]:
            value = str(path or "").strip()
            if not value:
                continue
            normalized = os.path.realpath(os.path.abspath(os.path.expanduser(value))) if os.path.exists(value) else value
            if normalized in seen_path:
                continue
            seen_path.add(normalized)
            merged_path.append(normalized)
        if merged_path:
            env["PATH"] = os.pathsep.join(merged_path)

    program_dir = ""
    if office:
        office_dir = os.path.dirname(office)
        if os.path.basename(office_dir) == "program" and office_dir in bridge_paths:
            program_dir = office_dir
    if not program_dir:
        program_dir = next((path for path in bridge_paths if os.path.basename(path) == "program"), "")
    if program_dir:
        env.setdefault("UNO_PATH", program_dir)

    return env


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

    _, bridge_paths = _resolve_libreoffice_runtime()
    existing_paths = list(bridge_paths)
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
    office = _get_libreoffice_executable()
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

office = sys.argv[1]
path = sys.argv[2]
port = str(random.randint(20020, 22999))
with tempfile.TemporaryDirectory(prefix="lo_uno_profile_") as profile_dir:
    profile_url = "file://" + os.path.abspath(profile_dir).replace("\\\\", "/")
    cmd = [
        office,
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
            page_count = 0
            try:
                cursor.jumpToLastPage()
                page_count = int(cursor.getPage() or 0)
            except Exception:
                page_count = 0
            if page_count <= 0 and cursor.jumpToPage(1):
                seen = set()
                while True:
                    current = int(cursor.getPage() or 0)
                    if current <= 0 or current in seen:
                        break
                    seen.add(current)
                    page_count = max(page_count, current)
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
        for attempt in range(3):
            proc = subprocess.run(
                [str(sys.executable), "-c", script, office, file_path],
                capture_output=True,
                text=True,
                timeout=40,
                check=False,
                env=_build_uno_subprocess_env(),
            )
            if proc.returncode != 0:
                logger.debug(
                    "LibreOffice page probe subprocess failed for %s on attempt %s: rc=%s stderr=%s",
                    file_path,
                    attempt + 1,
                    proc.returncode,
                    str(proc.stderr or "").strip(),
                )
                continue
            text = str(proc.stdout or "").strip()
            value = int(float(text)) if text else 0
            if value > 0:
                return value
        return 0
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
        f"{_build_office_document_failure_message(file_path, ext=ext)}"
    )


def _extract_office_text_by_uno_pages(file_path: str) -> tuple[str, bool]:
    office = _get_libreoffice_executable()
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

office = sys.argv[1]
path = sys.argv[2]
port = str(random.randint(23000, 25999))
with tempfile.TemporaryDirectory(prefix="lo_uno_pages_") as profile_dir:
    profile_url = "file://" + os.path.abspath(profile_dir).replace("\\\\", "/")
    cmd = [
        office,
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

            def _goto_page(target):
                try:
                    vc.jumpToPage(int(target))
                except Exception:
                    pass
                try:
                    current = int(vc.getPage() or 0)
                except Exception:
                    current = 0
                correction = 0
                while current > 0 and current < int(target) and correction < 3:
                    try:
                        moved = bool(vc.jumpToNextPage())
                    except Exception:
                        moved = False
                    if not moved:
                        break
                    try:
                        next_current = int(vc.getPage() or 0)
                    except Exception:
                        next_current = current
                    if next_current <= current:
                        break
                    current = next_current
                    correction += 1
                return current

            try:
                vc.jumpToLastPage()
                total_pages = int(vc.getPage() or 0)
            except Exception:
                total_pages = 0
            if total_pages <= 0:
                print("")
                raise SystemExit(0)

            page_texts = []
            target_page = 1
            while target_page <= total_pages:
                current_page = _goto_page(target_page)
                if current_page <= 0:
                    page_texts.extend([""] * (total_pages - len(page_texts)))
                    break
                if current_page > target_page:
                    page_texts.extend([""] * (current_page - target_page))
                    target_page = current_page
                if current_page < target_page:
                    page_texts.append("")
                    target_page += 1
                    continue

                text = ""
                boundary_page = current_page
                try:
                    vc.jumpToStartOfPage()
                    start = vc.getStart()
                except Exception:
                    start = None

                next_target = current_page + 1
                while next_target <= total_pages and boundary_page <= current_page:
                    boundary_page = _goto_page(next_target)
                    if boundary_page <= current_page:
                        next_target += 1

                try:
                    if start is not None:
                        if boundary_page > current_page:
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
                if boundary_page > current_page + 1:
                    page_texts.extend([""] * (boundary_page - current_page - 1))
                target_page = boundary_page if boundary_page > current_page else current_page + 1

            if len(page_texts) < total_pages:
                page_texts.extend([""] * (total_pages - len(page_texts)))
            print("\\n\\f\\n".join(page_texts[:total_pages]).strip())
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
        for attempt in range(3):
            proc = subprocess.run(
                [str(sys.executable), "-c", script, office, file_path],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=_build_uno_subprocess_env(),
            )
            if proc.returncode != 0:
                logger.debug(
                    "UNO page text extraction failed for %s on attempt %s: rc=%s stderr=%s",
                    file_path,
                    attempt + 1,
                    proc.returncode,
                    str(proc.stderr or "").strip(),
                )
                continue
            text = str(proc.stdout or "").strip()
            if not text:
                continue
            return text, ("\f" in text)
        return "", False
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
                "citations": _dedupe_text_items(list(secondary.get("citations") or []) + list(primary.get("citations") or [])),
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


def _page_alignment_signature(text: str) -> str:
    parts: List[str] = []
    for raw in str(text or "").splitlines():
        normalized = RAG_DB_Document._normalized_heading_text(str(raw or "").strip())
        if not normalized:
            continue
        parts.append(normalized[:80])
        if len(parts) >= 2:
            break
    if parts:
        return "|".join(parts)
    return ""


def _page_alignment_signature_matches(source_sig: str, target_sig: str) -> bool:
    left = str(source_sig or "").strip()
    right = str(target_sig or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True

    left_head = left.split("|", 1)[0]
    right_head = right.split("|", 1)[0]
    if left_head and right_head and left_head == right_head:
        return True
    if left_head and (left_head == right or left_head == right_head):
        return True
    if right_head and (right_head == left or right_head == left_head):
        return True
    if len(left_head) >= 6 and len(right_head) >= 6:
        if left_head == right_head or left_head in right or right_head in left:
            return True
    if len(left) >= 12 and len(right) >= 12 and (left in right or right in left):
        return True
    return False


def _trim_spurious_leading_blank_office_page(
    reference_text: str,
    target_page_texts: Sequence[str],
) -> tuple[List[str], int]:
    reference_pages = _split_native_pages(reference_text)
    target_pages = [str(item or "") for item in list(target_page_texts or [])]
    removed = 0
    while len(reference_pages) >= 2 and len(target_pages) >= 3:
        if str(target_pages[1] or "").strip():
            break

        reference_first_sig = _page_alignment_signature(reference_pages[0])
        reference_second_sig = _page_alignment_signature(reference_pages[1])
        target_first_sig = _page_alignment_signature(target_pages[0])
        target_third_sig = _page_alignment_signature(target_pages[2])
        if not reference_second_sig or not target_third_sig:
            break
        if reference_first_sig and target_first_sig and not _page_alignment_signature_matches(reference_first_sig, target_first_sig):
            break
        if not _page_alignment_signature_matches(reference_second_sig, target_third_sig):
            break

        target_pages = [target_pages[0]] + target_pages[2:]
        removed += 1
    return target_pages, removed


def _trim_trailing_blank_office_pages(target_page_texts: Sequence[str]) -> tuple[List[str], int]:
    target_pages = [str(item or "") for item in list(target_page_texts or [])]
    removed = 0
    while target_pages and not str(target_pages[-1] or "").strip():
        target_pages.pop()
        removed += 1
    return target_pages, removed


def _normalize_office_uno_pages(
    reference_text: str,
    target_page_texts: Sequence[str],
    *,
    raw_native_page_count: int = 0,
) -> tuple[List[str], int, int]:
    normalized_pages, removed_leading_blanks = _trim_spurious_leading_blank_office_page(
        reference_text,
        target_page_texts,
    )
    normalized_pages, _ = _trim_trailing_blank_office_pages(normalized_pages)
    effective_native_page_count = int(raw_native_page_count or 0)
    if (
        removed_leading_blanks > 0
        and effective_native_page_count > len(normalized_pages)
        and effective_native_page_count >= len(normalized_pages) + removed_leading_blanks
    ):
        effective_native_page_count -= removed_leading_blanks
    if effective_native_page_count <= 0:
        effective_native_page_count = len(normalized_pages)
    return normalized_pages, effective_native_page_count, removed_leading_blanks


def _materialize_effective_office_pages(
    reference_text: str,
    primary_pages: Sequence[str],
    secondary_pages: Sequence[str],
) -> List[str]:
    normalized_pages, _, _ = _normalize_office_uno_pages(
        reference_text,
        primary_pages,
        raw_native_page_count=0,
    )
    materialized_pages = _append_missing_tail_pages(normalized_pages, secondary_pages)
    materialized_pages, _ = _trim_trailing_blank_office_pages(materialized_pages)
    return [str(item or "") for item in materialized_pages]


def _map_source_pages_to_target_pages(source_pages: Sequence[str], target_pages: Sequence[str]) -> Dict[int, int]:
    source_signatures = [_page_alignment_signature(text) for text in list(source_pages or [])]
    target_signatures = [_page_alignment_signature(text) for text in list(target_pages or [])]
    if not source_signatures or not target_signatures:
        return {}

    mapping: Dict[int, int] = {}
    src_idx = 0
    tgt_idx = 0
    matched = 0
    while src_idx < len(source_signatures) and tgt_idx < len(target_signatures):
        source_sig = source_signatures[src_idx]
        target_sig = target_signatures[tgt_idx]

        if not target_sig:
            tgt_idx += 1
            continue
        if not source_sig:
            mapping[src_idx + 1] = min(len(target_signatures), tgt_idx + 1)
            src_idx += 1
            continue
        if _page_alignment_signature_matches(source_sig, target_sig):
            mapping[src_idx + 1] = tgt_idx + 1
            matched += 1
            src_idx += 1
            tgt_idx += 1
            continue

        next_target_sig = target_signatures[tgt_idx + 1] if tgt_idx + 1 < len(target_signatures) else ""
        if next_target_sig and _page_alignment_signature_matches(source_sig, next_target_sig):
            tgt_idx += 1
            continue

        next_source_sig = source_signatures[src_idx + 1] if src_idx + 1 < len(source_signatures) else ""
        if next_source_sig and _page_alignment_signature_matches(next_source_sig, target_sig):
            src_idx += 1
            continue

        mapping[src_idx + 1] = min(len(target_signatures), tgt_idx + 1)
        src_idx += 1
        tgt_idx += 1

    if matched <= 0 or not mapping:
        return {}

    ordered = sorted(mapping.items())
    last_source, last_target = ordered[-1]
    offset = int(last_target) - int(last_source)
    for source_page in range(1, len(source_signatures) + 1):
        mapping.setdefault(source_page, max(1, min(len(target_signatures), source_page + offset)))
    return mapping


def _project_page_number_by_mapping(
    source_page: int,
    page_mapping: Dict[int, int],
    *,
    target_page_count: int,
) -> int:
    if source_page <= 0 or target_page_count <= 0:
        return 0
    mapped = int(page_mapping.get(int(source_page)) or 0)
    if mapped > 0:
        return max(1, min(int(target_page_count), mapped))
    if not page_mapping:
        return max(1, min(int(target_page_count), int(source_page)))

    ordered = sorted((int(src), int(dst)) for src, dst in page_mapping.items() if int(src) > 0 and int(dst) > 0)
    if not ordered:
        return max(1, min(int(target_page_count), int(source_page)))
    if source_page <= ordered[0][0]:
        offset = ordered[0][1] - ordered[0][0]
        return max(1, min(int(target_page_count), int(source_page) + int(offset)))
    if source_page >= ordered[-1][0]:
        offset = ordered[-1][1] - ordered[-1][0]
        return max(1, min(int(target_page_count), int(source_page) + int(offset)))

    for index in range(len(ordered) - 1):
        left_source, left_target = ordered[index]
        right_source, right_target = ordered[index + 1]
        if left_source <= source_page <= right_source:
            if right_source <= left_source:
                return max(1, min(int(target_page_count), int(left_target)))
            ratio = float(int(source_page) - int(left_source)) / float(int(right_source) - int(left_source))
            mapped = int(round(float(left_target) + ratio * float(int(right_target) - int(left_target))))
            return max(1, min(int(target_page_count), mapped))
    return max(1, min(int(target_page_count), int(source_page)))


def _align_office_structured_payload_with_uno_pages(
    *,
    source_text: str,
    target_page_texts: Sequence[str],
    source_page_assets: Sequence[Dict[str, Any]],
    source_sections: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    source_pages = _split_native_pages(source_text)
    target_pages = [str(item or "") for item in list(target_page_texts or [])]
    target_page_count = len(target_pages)
    if not source_pages or not target_pages:
        return {
            "page_assets": [dict(item) for item in list(source_page_assets or []) if isinstance(item, dict)],
            "structured_sections": [dict(item) for item in list(source_sections or []) if isinstance(item, dict)],
        }
    if len(target_pages) <= len(source_pages):
        return {
            "page_assets": [dict(item) for item in list(source_page_assets or []) if isinstance(item, dict)],
            "structured_sections": [dict(item) for item in list(source_sections or []) if isinstance(item, dict)],
        }
    if not any(not str(text or "").strip() for text in target_pages):
        return {
            "page_assets": [dict(item) for item in list(source_page_assets or []) if isinstance(item, dict)],
            "structured_sections": [dict(item) for item in list(source_sections or []) if isinstance(item, dict)],
        }

    page_mapping = _map_source_pages_to_target_pages(source_pages, target_pages)
    if not page_mapping or not any(int(target) != int(source) for source, target in page_mapping.items()):
        return {
            "page_assets": [dict(item) for item in list(source_page_assets or []) if isinstance(item, dict)],
            "structured_sections": [dict(item) for item in list(source_sections or []) if isinstance(item, dict)],
        }

    projected_assets: List[Dict[str, Any]] = [
        {"page": page_number, "headers": [], "footers": [], "citations": [], "page_number": "", "images": []}
        for page_number in range(1, target_page_count + 1)
    ]
    for index, item in enumerate(list(source_page_assets or []), start=1):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        source_page = int(RAG_DB_Document.coerce_page_number(row.get("page")) or index)
        target_page = _project_page_number_by_mapping(source_page, page_mapping, target_page_count=target_page_count)
        if target_page <= 0:
            continue
        target_row = projected_assets[target_page - 1]
        target_row["headers"] = _dedupe_text_items(list(target_row.get("headers") or []) + list(row.get("headers") or []))
        target_row["footers"] = _dedupe_text_items(list(target_row.get("footers") or []) + list(row.get("footers") or []))
        target_row["citations"] = _dedupe_text_items(list(target_row.get("citations") or []) + list(row.get("citations") or []))
        if not str(target_row.get("page_number") or "").strip():
            target_row["page_number"] = str(row.get("page_number") or "").strip()
        target_row["images"] = _dedupe_image_items(list(target_row.get("images") or []) + list(row.get("images") or []))

    projected_sections: List[Dict[str, Any]] = []
    for item in list(source_sections or []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        source_page = int(RAG_DB_Document.coerce_page_number(row.get("page")) or 0)
        if source_page > 0:
            row["page"] = _project_page_number_by_mapping(source_page, page_mapping, target_page_count=target_page_count)
        projected_sections.append(row)

    return {
        "page_assets": projected_assets,
        "structured_sections": projected_sections,
    }


def _stage_office_cli_source(file_path: str, work_dir: str) -> str:
    source_path = os.path.abspath(file_path)
    suffix = os.path.splitext(source_path)[1].lower() or ".bin"
    staged_path = os.path.join(work_dir, f"input{suffix}")
    try:
        shutil.copyfile(source_path, staged_path)
        return staged_path
    except Exception:
        return source_path


def _probe_effective_native_page_count(file_path: str) -> int:
    file_path = os.path.abspath(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    raw_count = int(_probe_libreoffice_physical_page_count(file_path) or 0)
    if ext not in {".doc", ".docx"}:
        return raw_count

    try:
        if ext == ".doc":
            payload = _extract_doc_structured_payload_via_converted_docx(file_path)
        else:
            payload = _extract_docx_structured_payload(file_path)
    except Exception:
        payload = {}

    reference_text = str(payload.get("text_with_breaks") or "")
    uno_text, uno_has_breaks = _extract_office_text_by_uno_pages(file_path)
    office_text, office_has_breaks = _extract_office_text_with_page_breaks(file_path)
    uno_pages = _split_native_pages(uno_text) if uno_has_breaks and uno_text else []
    fallback_pages = _split_native_pages(office_text) if office_has_breaks and office_text else []
    materialized_pages = _materialize_effective_office_pages(
        reference_text or office_text,
        uno_pages,
        fallback_pages,
    )
    if materialized_pages:
        return len(materialized_pages)
    return raw_count


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
    office = _get_libreoffice_executable()
    if not office:
        return ""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = _stage_office_cli_source(file_path, tmp_dir)
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            staged_source,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            docx_path = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(staged_source))[0]}.docx")
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
    office = _get_libreoffice_executable()
    if not office:
        return "", False
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = _stage_office_cli_source(file_path, tmp_dir)
        command = [
            office,
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            tmp_dir,
            staged_source,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=60, check=False)
            txt_name = f"{os.path.splitext(os.path.basename(staged_source))[0]}.txt"
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
    office = _get_libreoffice_executable()
    if not office:
        return "", 0, [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = _stage_office_cli_source(file_path, tmp_dir)
        command = [
            office,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            tmp_dir,
            staged_source,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_pdf = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(staged_source))[0]}.pdf")
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
    # TOC styles are catalogue rows, not real content headings.
    if "toc" in source or "目录" in source:
        return None

    # Chinese Word style hierarchy should be: 标题 > 标题1 > 标题2 > 标题3 ...
    if re.search(r"(?:^|\s)标题(?:$|\s)", source):
        return 1
    match_cn = re.search(r"标题\s*([1-9])", source)
    if match_cn:
        # Shift by +1 so 标题1 becomes level-2 under 标题.
        return max(1, min(int(match_cn.group(1)) + 1, 6))

    # Keep default English Heading semantics for compatibility.
    match_en = re.search(r"heading\s*([1-9])", source)
    if match_en:
        return max(1, min(int(match_en.group(1)), 6))
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


def _extract_docx_note_text_map(
    archive: zipfile.ZipFile,
    *,
    part_path: str,
    node_name: str,
    ns: Dict[str, str],
) -> Dict[str, str]:
    try:
        raw_xml = archive.read(part_path)
    except Exception:
        return {}

    try:
        root = ET.fromstring(raw_xml)
    except Exception:
        return {}

    note_map: Dict[str, str] = {}
    for node in root.findall(f"w:{node_name}", ns):
        note_id = str(node.attrib.get(f"{{{ns['w']}}}id") or "").strip()
        if not note_id:
            continue
        note_type = str(node.attrib.get(f"{{{ns['w']}}}type") or "").strip().lower()
        if note_type in {"separator", "continuationseparator", "continuationnotice"}:
            continue
        parts: List[str] = []
        for paragraph in node.findall(".//w:p", ns):
            text = _extract_docx_text_from_paragraph(paragraph, ns)
            if text:
                parts.append(text)
        merged = "\n".join(str(item or "").strip() for item in parts if str(item or "").strip()).strip()
        if merged:
            note_map[note_id] = merged[:1200]
    return note_map


def _extract_docx_paragraph_note_refs(paragraph: ET.Element, ns: Dict[str, str]) -> List[tuple[str, str]]:
    refs: List[tuple[str, str]] = []
    for ref in paragraph.findall(".//w:footnoteReference", ns):
        note_id = str(ref.attrib.get(f"{{{ns['w']}}}id") or "").strip()
        if note_id:
            refs.append(("footnote", note_id))
    for ref in paragraph.findall(".//w:endnoteReference", ns):
        note_id = str(ref.attrib.get(f"{{{ns['w']}}}id") or "").strip()
        if note_id:
            refs.append(("endnote", note_id))
    return refs


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

    def _is_toc_like_row(cells: List[str]) -> bool:
        if not cells:
            return False
        cleaned_cells = [str(cell or "").strip() for cell in cells if str(cell or "").strip()]
        if len(cleaned_cells) >= 2:
            tail = str(cleaned_cells[-1] or "").strip()
            if re.fullmatch(r"\d{1,4}|[IVXLCDM]{1,8}", tail, flags=re.IGNORECASE):
                return True
        merged = " ".join(str(cell or "").strip() for cell in cells if str(cell or "").strip()).strip()
        if not merged:
            return False
        compact = re.sub(r"\s+", "", merged).lower()
        if compact in {"目录", "目錄", "contents", "tableofcontents", "toc"}:
            return True
        return bool(
            re.match(
                r"^.{1,220}?(?:\t+|[·•.]{2,}|\s{2,})(\d{1,4}|[IVXLCDM]{1,8})\s*$",
                merged,
                flags=re.IGNORECASE,
            )
        )

    toc_like_count = sum(1 for row in rows if _is_toc_like_row(row))
    if toc_like_count >= max(1, len(rows) - 1):
        plain_rows: List[str] = []
        for row in rows:
            cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
            if not cells:
                continue
            if len(cells) >= 2:
                plain_rows.append(f"{cells[0]}\t{cells[-1]}")
            else:
                plain_rows.append(cells[0])
        return "\n".join(plain_rows).strip()

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
    page_assets: List[Dict[str, Any]] = [{"images": [], "citations": []}]
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
            footnote_text_map = _extract_docx_note_text_map(
                archive,
                part_path="word/footnotes.xml",
                node_name="footnote",
                ns=ns,
            )
            endnote_text_map = _extract_docx_note_text_map(
                archive,
                part_path="word/endnotes.xml",
                node_name="endnote",
                ns=ns,
            )
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

            def _append_citations(citations: List[str]) -> None:
                if not citations:
                    return
                current_citations = page_assets[-1].setdefault("citations", [])
                current_citations.extend(citations)

            page_idx = 1
            footnote_display_index_by_id: Dict[str, int] = {}
            endnote_display_index_by_id: Dict[str, int] = {}
            next_footnote_display_index = 1
            next_endnote_display_index = 1
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
                    note_refs = _extract_docx_paragraph_note_refs(paragraph, ns)
                    note_markers: List[str] = []
                    citation_rows: List[str] = []
                    seen_note_ids: Set[str] = set()
                    for note_kind, note_id in note_refs:
                        key = f"{note_kind}:{note_id}"
                        if key in seen_note_ids:
                            continue
                        seen_note_ids.add(key)
                        if note_kind == "footnote":
                            note_text = str(footnote_text_map.get(note_id) or "").strip()
                            display_index = footnote_display_index_by_id.get(note_id)
                            if display_index is None:
                                display_index = next_footnote_display_index
                                footnote_display_index_by_id[note_id] = int(display_index)
                                next_footnote_display_index += 1
                            marker = f"[^f{int(display_index)}]"
                            if note_text:
                                citation_rows.append(f"[^f{int(display_index)}]: {note_text}")
                            note_markers.append(marker)
                        else:
                            note_text = str(endnote_text_map.get(note_id) or "").strip()
                            display_index = endnote_display_index_by_id.get(note_id)
                            if display_index is None:
                                display_index = next_endnote_display_index
                                endnote_display_index_by_id[note_id] = int(display_index)
                                next_endnote_display_index += 1
                            marker = f"[^e{int(display_index)}]"
                            if note_text:
                                citation_rows.append(f"[^e{int(display_index)}]: {note_text}")
                            note_markers.append(marker)

                    if note_markers:
                        marker_suffix = " ".join(note_markers).strip()
                        if text:
                            text = f"{text} {marker_suffix}".strip()
                        else:
                            text = marker_suffix

                    if citation_rows:
                        _append_citations(citation_rows)

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
                        page_assets.append({"images": [], "citations": []})
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
                    page_assets.append({"images": [], "citations": []})

        while pages and not pages[-1]:
            pages.pop()
            if page_assets:
                page_assets.pop()
        if not pages:
            pages = [[]]
        if not page_assets:
            page_assets = [{"images": [], "citations": []}]

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
                "citations": _dedupe_text_items(list(item.get("citations") or [])),
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
    office = _get_libreoffice_executable()
    if not office:
        return {"native_catalog": [], "style_catalog": [], "font_catalog": []}

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = _stage_office_cli_source(file_path, tmp_dir)
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            staged_source,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_docx = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(staged_source))[0]}.docx")
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
    office = _get_libreoffice_executable()
    if not office:
        return {
            "text_with_breaks": "",
            "has_breaks": False,
            "catalogs": {"native_catalog": [], "style_catalog": [], "font_catalog": []},
            "page_assets": [],
            "structured_sections": [],
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = _stage_office_cli_source(file_path, tmp_dir)
        command = [
            office,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            staged_source,
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=120, check=False)
            expected_docx = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(staged_source))[0]}.docx")
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
            raw_uno_pages = _split_native_pages(uno_text) if uno_has_breaks and uno_text else []
            fallback_pages = _split_native_pages(office_text) if has_native_breaks and office_text else []
            reference_text = xml_text if xml_has_breaks and xml_text else office_text
            uno_pages = _materialize_effective_office_pages(reference_text, raw_uno_pages, fallback_pages)
            if uno_pages:
                uno_text = "\n\f\n".join(uno_pages).strip()
                uno_has_breaks = len(uno_pages) > 1
            aligned_doc_payload = _align_office_structured_payload_with_uno_pages(
                source_text=reference_text,
                target_page_texts=uno_pages,
                source_page_assets=doc_page_assets,
                source_sections=doc_structured_sections,
            )
            doc_page_assets = list(aligned_doc_payload.get("page_assets") or doc_page_assets)
            doc_structured_sections = list(aligned_doc_payload.get("structured_sections") or doc_structured_sections)
            raw_native_page_count = _require_libreoffice_physical_page_count(
                file_path,
                ext=".doc",
                fallback_counts=[
                    len(uno_pages),
                    len(xml_text.split("\f")) if xml_has_breaks and xml_text else 0,
                    len(office_text.split("\f")) if has_native_breaks and office_text else 0,
                ],
            )
            effective_native_page_count = len(uno_pages) if uno_pages else int(raw_native_page_count or 0)
            doc_page_layout = _merge_page_layouts(doc_page_assets, [], total_pages=int(effective_native_page_count or 0))
            doc_common_metadata = {
                "file_name": file_path,
                "source_extension": ".doc",
                "native_page_count": int(effective_native_page_count or 0),
                "native_catalog": doc_catalogs.get("native_catalog") or [],
                "style_catalog": doc_catalogs.get("style_catalog") or [],
                "font_catalog": doc_catalogs.get("font_catalog") or [],
                "structured_sections": doc_structured_sections,
                "structured_page_assets": doc_page_assets,
                "page_layout": doc_page_layout,
                "txt_reference_pages": _split_native_pages(office_text),
            }
            if uno_text and uno_has_breaks:
                return Document(
                    text=uno_text,
                    metadata={
                        **doc_common_metadata,
                        "doc_parser": "libreoffice-uno-pages",
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
            raw_uno_pages = _split_native_pages(uno_text) if uno_has_breaks and uno_text else []
            fallback_pages = _split_native_pages(office_text) if has_native_breaks and office_text else []
            reference_text = xml_text if xml_has_breaks and xml_text else office_text
            uno_pages = _materialize_effective_office_pages(reference_text, raw_uno_pages, fallback_pages)
            if uno_pages:
                uno_text = "\n\f\n".join(uno_pages).strip()
                uno_has_breaks = len(uno_pages) > 1
            aligned_docx_payload = _align_office_structured_payload_with_uno_pages(
                source_text=reference_text,
                target_page_texts=uno_pages,
                source_page_assets=docx_page_assets,
                source_sections=docx_structured_sections,
            )
            docx_page_assets = list(aligned_docx_payload.get("page_assets") or docx_page_assets)
            docx_structured_sections = list(aligned_docx_payload.get("structured_sections") or docx_structured_sections)
            raw_native_page_count = _require_libreoffice_physical_page_count(
                file_path,
                ext=".docx",
                fallback_counts=[
                    len(uno_pages),
                    len(xml_text.split("\f")) if xml_has_breaks and xml_text else 0,
                    len(office_text.split("\f")) if has_native_breaks and office_text else 0,
                ],
            )
            effective_native_page_count = len(uno_pages) if uno_pages else int(raw_native_page_count or 0)
            docx_common_metadata = {
                "file_name": file_path,
                "source_extension": ".docx",
                "native_page_count": int(effective_native_page_count or 0),
                "native_catalog": docx_catalogs.get("native_catalog") or [],
                "style_catalog": docx_catalogs.get("style_catalog") or [],
                "font_catalog": docx_catalogs.get("font_catalog") or [],
                "structured_sections": docx_structured_sections,
                "structured_page_assets": docx_page_assets,
                "page_layout": _merge_page_layouts(docx_page_assets, [], total_pages=int(effective_native_page_count or 0)),
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


def build_rag_db_documents(
    loaded_docs: Sequence[Document],
    *,
    progress_callback: Optional[BuildProgressCallback] = None,
) -> List[RAG_DB_Document]:
    rag_docs: List[RAG_DB_Document] = []
    total_docs = len(loaded_docs)
    for index, loaded_doc in enumerate(loaded_docs, start=1):
        source_name = str((loaded_doc.metadata or {}).get("file_name") or loaded_doc.doc_id or "").strip()
        display_name = os.path.basename(source_name) or source_name or f"document_{index}"
        _emit_build_progress(
            progress_callback,
            "build_doc_started",
            doc_name=display_name,
            index=index,
            total=total_docs,
        )
        try:
            doc_id = stable_doc_id(loaded_doc)
            rag_doc = create_rag_db_document(loaded_doc, stable_doc_id=doc_id).build()
        except Exception as exc:
            logger.warning("Failed to build RAG_DB_Document for %s: %s", loaded_doc.doc_id, exc)
            _emit_build_progress(
                progress_callback,
                "build_doc_failed",
                doc_name=display_name,
                index=index,
                total=total_docs,
                error=str(exc),
            )
            continue
        chunk_docs = list(getattr(rag_doc, "chunk_documents", []) or [])
        tree_markdown = ""
        try:
            tree_markdown = str(rag_doc.export_markdown_from_tree() or "")
        except Exception:
            tree_markdown = ""
        if not chunk_docs and not tree_markdown.strip():
            _emit_build_progress(
                progress_callback,
                "build_doc_skipped",
                doc_name=str(getattr(rag_doc, "doc_name", display_name) or display_name),
                index=index,
                total=total_docs,
                reason="empty_document",
            )
            continue
        rag_docs.append(rag_doc)
        _emit_build_progress(
            progress_callback,
            "build_doc_completed",
            doc_name=str(getattr(rag_doc, "doc_name", display_name) or display_name),
            index=index,
            total=total_docs,
            page_count=int(getattr(rag_doc, "page_count", 0) or 0),
            chunk_count=len(chunk_docs),
        )
    return rag_docs


def chunk_documents_from_rag_documents(rag_docs: Sequence[RAG_DB_Document]) -> List[Document]:
    docs: List[Document] = []
    for rag_doc in rag_docs:
        docs.extend(getattr(rag_doc, "chunk_documents", []) or [])
    return docs


def prepare_documents_for_indexing_from_loaded_docs(loaded_docs: Sequence[Document]) -> List[Document]:
    rag_docs = build_rag_db_documents(loaded_docs)
    return chunk_documents_from_rag_documents(rag_docs)


def load_rag_documents_from_paths(
    paths: Sequence[str],
    supported_extensions: Set[str],
    *,
    progress_callback: Optional[BuildProgressCallback] = None,
) -> List[RAG_DB_Document]:
    loaded_docs: List[Document] = []
    total_paths = len(paths)
    _emit_build_progress(progress_callback, "load_started", total=total_paths)
    for index, file_path in enumerate(paths, start=1):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_extensions:
            logger.warning("Skipping unsupported upload file %s", file_path)
            _emit_build_progress(
                progress_callback,
                "load_doc_skipped",
                doc_name=os.path.basename(file_path) or file_path,
                index=index,
                total=total_paths,
                reason="unsupported_extension",
            )
            continue
        loaded_doc = load_single_file_document(file_path, supported_extensions)
        if loaded_doc is None:
            _emit_build_progress(
                progress_callback,
                "load_doc_skipped",
                doc_name=os.path.basename(file_path) or file_path,
                index=index,
                total=total_paths,
                reason="load_failed",
            )
            continue
        loaded_docs.append(loaded_doc)
        _emit_build_progress(
            progress_callback,
            "load_doc_completed",
            doc_name=os.path.basename(file_path) or file_path,
            index=index,
            total=total_paths,
        )
    rag_docs = build_rag_db_documents(loaded_docs, progress_callback=progress_callback)
    _emit_build_progress(
        progress_callback,
        "load_completed",
        total=total_paths,
        loaded=len(loaded_docs),
        built=len(rag_docs),
    )
    return rag_docs


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
