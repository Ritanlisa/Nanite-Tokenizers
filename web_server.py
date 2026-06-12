import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional

import _patch_py314  # noqa: F401  (Python 3.14+ PEP 649 compatibility)
import yaml

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from capabilities import get_capabilities
from agent.agent import Agent
from agent.tools import tools as registered_tools
from main import health_check, setup_logging
from monitoring import start_metrics_server
from rag.ocr import ocr_enabled
from rag.engine import RAGEngine, SUPPORTED_RAG_EXTENSIONS
from tool_usage import get_tool_usage, reset_tool_usage
from mcp_client.client import get_mcp_client
from locale_context import normalize_language, set_current_language, reset_current_language
import tool_approval

logger = logging.getLogger(__name__)

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field("default", min_length=1)
    stream: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    images: Optional[list[str]] = None
    rag_db_name: Optional[str] = None
    rag_db_names: Optional[list[str]] = None
    force_agent: bool = False
    allowed_mcp_tools: Optional[list[str]] = None
    messages: Optional[list[dict]] = None
    conversation_path: Optional[list[str]] = None
    language: Optional[str] = None
    auto_approve: bool = False


class ResetRequest(BaseModel):
    session_id: str = Field("default", min_length=1)


class SettingsUpdateRequest(BaseModel):
    settings: dict[str, object]
    restart: bool = False


class RagBuildRequest(BaseModel):
    db_name: Optional[str] = None


class RagDbNameRequest(BaseModel):
    name: str = Field(..., min_length=1)


class RagDbRenameRequest(BaseModel):
    old_name: str = Field(..., min_length=1)
    new_name: str = Field(..., min_length=1)


class RagDbCloneRequest(BaseModel):
    source_name: str = Field(..., min_length=1)
    target_name: str = Field(..., min_length=1)


class RagSelectionRequest(BaseModel):
    db_names: list[str] = Field(default_factory=list)
    enable_rag: Optional[bool] = None


class RagRetrieveRequest(BaseModel):
    query: Optional[str] = None
    doc_name: Optional[str] = None
    section: Optional[str] = None
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    regex: Optional[str] = None
    chunk: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class DebugToolInvokeRequest(BaseModel):
    session_id: str = Field("default", min_length=1)
    tool_name: str = Field(..., min_length=1)
    tool_args: dict[str, Any] = Field(default_factory=dict)
    rag_db_name: Optional[str] = None
    rag_db_names: Optional[list[str]] = None


class ToolConfirmRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    call_id: str = Field(..., min_length=1)
    approved: bool = True


class StopGenerationRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


SESSION_STORE: dict[str, Agent] = {}
SESSION_PARAMS: dict[str, dict[str, Optional[object]]] = {}


def get_or_create_agent(session_id: str, model: Optional[str], temperature: Optional[float]) -> Agent:
    params = {"model": model, "temperature": temperature}
    if session_id not in SESSION_STORE or SESSION_PARAMS.get(session_id) != params:
        existing = SESSION_STORE.get(session_id)
        if existing is not None:
            existing.close()
        SESSION_STORE[session_id] = Agent(
            session_id=session_id,
            model=model,
            temperature=temperature,
        )
        SESSION_PARAMS[session_id] = params
    return SESSION_STORE[session_id]


def clear_session(session_id: str) -> None:
    agent = SESSION_STORE.pop(session_id, None)
    if agent is not None:
        agent.close()
    SESSION_PARAMS.pop(session_id, None)
    reset_tool_usage(session_id)


def update_settings(args: argparse.Namespace) -> None:
    config.settings = config.settings.update(
        DATA_DIR=args.data_dir,
        PERSIST_DIR=args.persist_dir,
        AGENT_VERBOSE=args.verbose or config.settings.AGENT_VERBOSE,
        LOG_LEVEL=(str(args.log_level).upper() if getattr(args, "log_level", None) else ("DEBUG" if args.verbose else config.settings.LOG_LEVEL)),
    )


def _extract_sections_from_rag_doc(rag_doc: Any) -> list:
    """从 RAG_DB_Document 中提取 SectionInfo 列表（跳过封面/目录，智能分段）"""
    from agent.kg_build_agent import SectionInfo
    from rag.document_interface import PageType

    sections = []
    try:
        mono_pages = rag_doc.get_mono_pages()
    except Exception:
        return sections

    # Build page → catalog info mapping
    page_catalog: dict = {}
    try:
        for cat_item in rag_doc.catalog_payload() or []:
            if not isinstance(cat_item, dict):
                continue
            start = int(cat_item.get("page") or 0)
            title = str(cat_item.get("title") or "").strip()
            parent = str(cat_item.get("parent_title") or "").strip()
            if start > 0 and title:
                page_catalog.setdefault(start, []).append({"title": title, "parent": parent})
    except Exception:
        pass

    # Collect all content text in order, tracking catalog transitions
    all_text_parts: list = []  # [(page_num, text, title, parent)]

    for page in mono_pages:
        text = (getattr(page, "markdown_text", "") or "").strip()
        if not text or len(text) < 50:
            continue
        cat = getattr(page, "category", "")
        if cat in (PageType.COVER, PageType.CATALOGUE):
            continue

        page_num = int(getattr(page, "page_number", 0) or 0)
        title = ""
        parent = ""

        # Check if catalog has an entry for this page
        for cat_page, cat_entries in sorted(page_catalog.items()):
            if cat_page >= page_num:
                candidates = cat_entries
                if candidates:
                    title = candidates[0]["title"]
                    parent = candidates[0].get("parent", "")
                break

        if not title:
            title = (getattr(page, "title", "") or "").strip()
        if not title:
            m = re.match(r'#+\s*(.+)', text)
            if m:
                title = m.group(1).strip()

        all_text_parts.append((page_num, text, title, parent))

    if not all_text_parts:
        return sections

    # Merge consecutive fragments on same page, then chunk long pages
    CHUNK_SIZE = 2000
    TOC_PATTERN = re.compile(r'^\s*(#+\s+.*|第[一二三四五六七八九十\d]+章\s|[\d\.]+\s+\w+)')

    current_page = all_text_parts[0][0]
    current_text = ""
    current_title = all_text_parts[0][2]
    current_parent = all_text_parts[0][3]

    for page_num, text, title, parent in all_text_parts:
        if page_num == current_page and len(current_text) + len(text) < CHUNK_SIZE:
            current_text += "\n\n" + text
        else:
            if current_text.strip():
                lines = current_text.strip().splitlines()
                toc_lines = sum(1 for l in lines if TOC_PATTERN.match(l))
                # P2 fix: stricter TOC detection — skip if >30% TOC or title is "目 录"
                is_toc_title = "目" in (current_title or "") and "录" in (current_title or "")
                if len(lines) > 0 and toc_lines / len(lines) < 0.3 and not is_toc_title:
                    sections.append(SectionInfo(
                        section_id=f"sec-{len(sections)}",
                        title=current_title or f"Section {len(sections)+1}",
                        text=current_text.strip()[:4000],
                        parent_title=current_parent,
                        page=current_page,
                    ))
            current_page = page_num
            current_text = text
            current_title = title
            current_parent = parent

    if current_text.strip():
        lines = current_text.strip().splitlines()
        toc_lines = sum(1 for l in lines if TOC_PATTERN.match(l))
        is_toc_title = "目" in (current_title or "") and "录" in (current_title or "")
        if len(lines) > 0 and toc_lines / len(lines) < 0.3 and not is_toc_title:
            sections.append(SectionInfo(
                section_id=f"sec-{len(sections)}",
                title=current_title or f"Section {len(sections)+1}",
                text=current_text.strip()[:4000],
                parent_title=current_parent,
                page=current_page,
            ))

    return sections


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger = logging.getLogger(__name__)
        init_task: Optional[asyncio.Task] = None
        try:
            client = get_mcp_client()
            init_task = asyncio.create_task(client.initialize())
            # 启动阶段只等待一个很短的时间，避免首次运行被 MCP 初始化拖住。
            # MCP 若较慢，会在后台继续初始化；首次实际使用时仍会确保 initialize 已完成。
            startup_wait_s = min(2.0, float(getattr(config.settings, "MCP_INIT_TIMEOUT", 60)))
            try:
                await asyncio.wait_for(init_task, timeout=startup_wait_s)
            except asyncio.TimeoutError:
                logger.info("MCP init still running in background (waited %.1fs)", startup_wait_s)
            if getattr(client, "_fallback_mode", False):
                logger.warning("MCP server unavailable or disabled; using direct HTTP fallback")
            elif getattr(client, "_initialized", False):
                logger.info("MCP server reachable")
        except Exception as exc:
            logger.warning("MCP init failed at app startup: %s", exc)
        yield
        if init_task is not None and not init_task.done():
            init_task.cancel()
        await get_mcp_client().close()

    app = FastAPI(title="Nanite Agent API", lifespan=lifespan)
    db_name_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
    allowed_upload_exts = set(SUPPORTED_RAG_EXTENSIONS)
    rag_db_context_lock = threading.RLock()
    rag_build_jobs_lock = threading.Lock()
    rag_build_jobs: dict[str, dict[str, Any]] = {}
    rag_active_build_jobs: dict[str, str] = {}
    terminal_job_statuses = {"completed", "failed"}

    def _read_settings_yaml() -> dict[str, object]:
        if not os.path.exists("settings.yaml"):
            return {}
        with open("settings.yaml", "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data

    def _ordered_settings(data: dict[str, object]) -> dict[str, object]:
        ordered: dict[str, object] = {}
        for key in config.Settings.model_fields:
            if key in data:
                ordered[key] = data[key]
        for key, value in data.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    def _write_settings_yaml(data: dict[str, object]) -> None:
        ordered = _ordered_settings(data)
        with open("settings.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(ordered, handle, sort_keys=False, allow_unicode=False)

    def _normalize_db_name(name: str) -> str:
        normalized = (name or "").strip()
        if not db_name_pattern.fullmatch(normalized):
            raise HTTPException(
                status_code=400,
                detail="Invalid db name: use letters, numbers, ., _, - (max 40)",
            )
        return normalized

    def _db_dir(name: str) -> str:
        return os.path.join(config.settings.PERSIST_DIR, name)

    def _db_docs_dir(name: str) -> str:
        return os.path.join(_db_dir(name), "docs")

    def _is_valid_db(name: str) -> bool:
        db_dir = _db_dir(name)
        docs_dir = _db_docs_dir(name)
        return os.path.isdir(db_dir) and os.path.isdir(docs_dir)

    def _normalize_tool_name_list(values: Optional[list[str]]) -> list[str]:
        if not values:
            return []
        deduped: list[str] = []
        for item in values:
            name = str(item or "").strip()
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    def _is_rag_tool_name(name: str) -> bool:
        return str(name or "").strip().startswith("rag_")

    def _list_selectable_mcp_tools() -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for tool in registered_tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name or name in seen or _is_rag_tool_name(name):
                continue
            seen.add(name)
            items.append(
                {
                    "name": name,
                    "description": str(getattr(tool, "description", "") or ""),
                }
            )
        items.sort(key=lambda item: item["name"])
        return items

    def _list_valid_dbs() -> list[str]:
        base_dir = config.settings.PERSIST_DIR
        os.makedirs(base_dir, exist_ok=True)
        try:
            entries = os.listdir(base_dir)
        except FileNotFoundError:
            return []
        dbs = [name for name in entries if _is_valid_db(name)]
        dbs.sort()
        return dbs

    def _list_db_docs(name: str) -> list[str]:
        docs_dir = _db_docs_dir(name)
        if not os.path.isdir(docs_dir):
            return []
        docs = [
            item
            for item in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, item))
        ]
        docs.sort()
        return docs

    def _count_docstore_chunks(db_dir: str) -> int:
        docstore_path = os.path.join(db_dir, "docstore.json")
        if not os.path.isfile(docstore_path):
            return 0
        try:
            with open(docstore_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return 0
            node_map = data.get("docstore/data")
            if isinstance(node_map, dict):
                return len(node_map)
            metadata_map = data.get("docstore/metadata")
            if isinstance(metadata_map, dict):
                return len(metadata_map)
        except Exception:
            return 0
        return 0

    def _count_chroma_chunks(db_dir: str) -> int:
        chroma_dir = os.path.join(db_dir, "chroma")
        if not os.path.isdir(chroma_dir):
            return 0
        try:
            import chromadb

            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_or_create_collection("llama_index")
            return int(collection.count())
        except Exception:
            return 0

    def _collect_db_stats(name: str) -> dict[str, object]:
        db_dir = _db_dir(name)
        docs = _list_db_docs(name)
        chunk_count = _count_docstore_chunks(db_dir)
        if chunk_count <= 0:
            chunk_count = _count_chroma_chunks(db_dir)

        total_size = 0
        latest_mtime = 0.0
        file_count = 0
        for root, _, files in os.walk(db_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    stat = os.stat(file_path)
                except OSError:
                    continue
                total_size += stat.st_size
                latest_mtime = max(latest_mtime, stat.st_mtime)
                file_count += 1

        has_index = any(
            os.path.exists(os.path.join(db_dir, candidate))
            for candidate in (
                "docstore.json",
                "index_store.json",
                "faiss.index",
                "chroma",
            )
        )

        return {
            "database": name,
            "documents_count": len(docs),
            "chunks_count": chunk_count,
            "file_count": file_count,
            "size_bytes": total_size,
            "has_index": has_index,
            "updated_at": latest_mtime or None,
        }

    def _clear_db_index_artifacts(name: str) -> None:
        db_dir = _db_dir(name)
        if not os.path.isdir(db_dir):
            return
        for item in os.listdir(db_dir):
            if item == "docs":
                continue
            target = os.path.join(db_dir, item)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.remove(target)
                except FileNotFoundError:
                    pass

    def _persist_rag_selection(primary: Optional[str], selected: list[str]) -> None:
        valid_selected = [name for name in selected if _is_valid_db(name)]
        if primary and primary not in valid_selected:
            primary = valid_selected[0] if valid_selected else None
        config.settings = config.settings.update(RAG_DB_NAME=primary, RAG_DB_NAMES=valid_selected)
        settings_data = _read_settings_yaml()
        settings_data["RAG_DB_NAME"] = primary
        settings_data["RAG_DB_NAMES"] = valid_selected
        _write_settings_yaml(settings_data)

    @contextmanager
    def _scoped_rag_db(name: str):
        normalized = _normalize_db_name(name)
        if not _is_valid_db(normalized):
            raise HTTPException(status_code=404, detail="Database not found")

        rag_db_context_lock.acquire()
        original_db_name = config.settings.RAG_DB_NAME
        original_db_names = list(config.settings.RAG_DB_NAMES)
        try:
            config.settings = config.settings.update(
                RAG_DB_NAME=normalized,
                RAG_DB_NAMES=[normalized],
            )
            yield normalized
        finally:
            config.settings = config.settings.update(
                RAG_DB_NAME=original_db_name,
                RAG_DB_NAMES=original_db_names,
            )
            rag_db_context_lock.release()

    def _clone_job_payload(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job.get("job_id") or ""),
            "database": str(job.get("database") or ""),
            "mode": str(job.get("mode") or ""),
            "status": str(job.get("status") or "queued"),
            "phase": str(job.get("phase") or "queued"),
            "progress": max(0, min(100, int(job.get("progress") or 0))),
            "files": int(job.get("files") or 0),
            "added": int(job.get("added") or 0),
            "error": str(job.get("error") or "") or None,
            "documents": [
                {
                    "name": str(item.get("name") or ""),
                    "progress": max(0, min(100, int(item.get("progress") or 0))),
                    "status": str(item.get("status") or "queued"),
                    "page_count": int(item.get("page_count") or 0),
                    "chunk_count": int(item.get("chunk_count") or 0),
                    "error": str(item.get("error") or "") or None,
                }
                for item in list(job.get("documents") or [])
                if isinstance(item, dict)
            ],
        }

    def _snapshot_rag_build_job(job_id: str) -> Optional[dict[str, Any]]:
        with rag_build_jobs_lock:
            job = rag_build_jobs.get(job_id)
            if not isinstance(job, dict):
                return None
            return _clone_job_payload(job)

    def _resolve_job_document(job: dict[str, Any], doc_name: str) -> Optional[dict[str, Any]]:
        normalized = os.path.basename(str(doc_name or "")).strip() or str(doc_name or "").strip()
        if not normalized:
            return None
        for item in list(job.get("documents") or []):
            if not isinstance(item, dict):
                continue
            candidate = os.path.basename(str(item.get("name") or "")).strip() or str(item.get("name") or "").strip()
            if candidate == normalized:
                return item
        return None

    def _recompute_job_progress(job: dict[str, Any]) -> None:
        progresses = [
            max(0, min(100, int(item.get("progress") or 0)))
            for item in list(job.get("documents") or [])
            if isinstance(item, dict)
        ]
        if progresses:
            job["progress"] = max(0, min(100, int(round(sum(progresses) / len(progresses)))))

    def _register_rag_build_job(db_name: str, mode: str, doc_names: list[str]) -> dict[str, Any]:
        with rag_build_jobs_lock:
            active_job_id = rag_active_build_jobs.get(db_name)
            if active_job_id:
                active_job = rag_build_jobs.get(active_job_id)
                if isinstance(active_job, dict) and str(active_job.get("status") or "") not in terminal_job_statuses:
                    raise HTTPException(status_code=409, detail="A build job is already running for this database")
                rag_active_build_jobs.pop(db_name, None)

            job_id = uuid.uuid4().hex
            documents = [
                {
                    "name": os.path.basename(str(name or "")).strip() or str(name or "").strip(),
                    "progress": 0,
                    "status": "queued",
                    "page_count": 0,
                    "chunk_count": 0,
                    "error": None,
                }
                for name in doc_names
                if str(name or "").strip()
            ]
            job = {
                "job_id": job_id,
                "database": db_name,
                "mode": mode,
                "status": "queued",
                "phase": "queued",
                "progress": 0,
                "files": len(documents),
                "added": 0,
                "error": None,
                "documents": documents,
            }
            rag_build_jobs[job_id] = job
            rag_active_build_jobs[db_name] = job_id
            return _clone_job_payload(job)

    def _update_rag_build_job(job_id: str, stage: str, payload: dict[str, Any]) -> None:
        with rag_build_jobs_lock:
            job = rag_build_jobs.get(job_id)
            if not isinstance(job, dict):
                return
            if str(job.get("status") or "") in terminal_job_statuses:
                return

            job["status"] = "running"
            documents = [item for item in list(job.get("documents") or []) if isinstance(item, dict)]
            doc_name = str(payload.get("doc_name") or "").strip()
            target_doc = _resolve_job_document(job, doc_name)

            if stage == "engine_started":
                job["phase"] = "preparing"
                job["progress"] = max(int(job.get("progress") or 0), 1)
            elif stage == "load_started":
                job["phase"] = "loading"
                job["progress"] = max(int(job.get("progress") or 0), 5)
            elif stage == "load_doc_completed":
                job["phase"] = "loading"
                if target_doc is not None:
                    target_doc["status"] = "loaded"
                    target_doc["progress"] = max(int(target_doc.get("progress") or 0), 20)
            elif stage in {"load_doc_skipped", "build_doc_failed", "build_doc_skipped"}:
                job["phase"] = "building"
                if target_doc is not None:
                    target_doc["status"] = "failed" if stage == "build_doc_failed" else "skipped"
                    target_doc["progress"] = 100
                    target_doc["error"] = str(payload.get("error") or payload.get("reason") or "") or None
            elif stage == "build_doc_started":
                job["phase"] = "building"
                if target_doc is not None:
                    target_doc["status"] = "building"
                    target_doc["progress"] = max(int(target_doc.get("progress") or 0), 35)
            elif stage == "build_doc_completed":
                job["phase"] = "building"
                if target_doc is not None:
                    target_doc["status"] = "built"
                    target_doc["progress"] = max(int(target_doc.get("progress") or 0), 70)
                    target_doc["page_count"] = int(payload.get("page_count") or 0)
                    target_doc["chunk_count"] = int(payload.get("chunk_count") or 0)
            elif stage == "index_started":
                job["phase"] = "indexing"
                for item in documents:
                    if str(item.get("status") or "") not in {"failed", "skipped"}:
                        item["status"] = "indexing"
                    item["progress"] = max(int(item.get("progress") or 0), 85)
            elif stage == "index_completed":
                job["phase"] = "indexing"
                for item in documents:
                    if str(item.get("status") or "") not in {"failed", "skipped"}:
                        item["status"] = "indexed"
                    item["progress"] = max(int(item.get("progress") or 0), 95)
            elif stage == "persist_started":
                job["phase"] = "persisting"
                for item in documents:
                    if str(item.get("status") or "") not in {"failed", "skipped"}:
                        item["status"] = "persisting"
                    item["progress"] = max(int(item.get("progress") or 0), 97)
            elif stage == "persist_completed":
                job["phase"] = "persisting"
                job["added"] = int(payload.get("added") or job.get("added") or 0)
                for item in documents:
                    if str(item.get("status") or "") not in {"failed", "skipped"}:
                        item["status"] = "persisted"
                    item["progress"] = max(int(item.get("progress") or 0), 99)

            elif stage == "kg_build_started":
                job["phase"] = "kg_building"
                job["progress"] = max(int(job.get("progress") or 0), 72)
                if target_doc is not None:
                    target_doc["status"] = "kg_extracting"
                    target_doc["progress"] = 73
            elif stage == "kg_build_completed":
                job["phase"] = "kg_building"
                job["progress"] = max(int(job.get("progress") or 0), 80)
                if target_doc is not None:
                    target_doc["status"] = "kg_done"
                    target_doc["progress"] = 80
            elif stage == "kg_build_error":
                job["phase"] = "kg_building"
                if target_doc is not None:
                    target_doc["status"] = "kg_error"
                    target_doc["progress"] = 80
                    target_doc["error"] = str(payload.get("error") or "")[:500] or None

            _recompute_job_progress(job)

    def _complete_rag_build_job(job_id: str, *, db_name: str, added: int) -> None:
        with rag_build_jobs_lock:
            job = rag_build_jobs.get(job_id)
            if not isinstance(job, dict):
                return
            job["status"] = "completed"
            job["phase"] = "completed"
            job["added"] = int(added)
            job["error"] = None
            for item in list(job.get("documents") or []):
                if not isinstance(item, dict):
                    continue
                item["progress"] = 100
                if str(item.get("status") or "") not in {"failed", "skipped"}:
                    item["status"] = "completed"
            job["progress"] = 100
            if rag_active_build_jobs.get(db_name) == job_id:
                rag_active_build_jobs.pop(db_name, None)

    def _fail_rag_build_job(job_id: str, *, db_name: str, error_text: str) -> None:
        with rag_build_jobs_lock:
            job = rag_build_jobs.get(job_id)
            if not isinstance(job, dict):
                return
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(error_text or "")[:500]
            for item in list(job.get("documents") or []):
                if not isinstance(item, dict):
                    continue
                if str(item.get("status") or "") in {"queued", "loaded", "building", "indexed", "persisting"}:
                    item["status"] = "failed"
                    item["error"] = job["error"]
            if rag_active_build_jobs.get(db_name) == job_id:
                rag_active_build_jobs.pop(db_name, None)

    def _execute_rag_build_job(job_id: str, db_name: str, mode: str, paths: list[str]) -> int:
        with _scoped_rag_db(db_name):
            engine = RAGEngine()
            if mode == "rebuild":
                _clear_db_index_artifacts(db_name)
                added = engine.rebuild_index_from_paths(
                    paths,
                    progress_callback=lambda stage, payload: _update_rag_build_job(job_id, stage, payload),
                )
            else:
                added = engine.add_documents_from_paths(
                    paths,
                    progress_callback=lambda stage, payload: _update_rag_build_job(job_id, stage, payload),
                )
            engine.clear_query_cache()
            return int(added)

    async def _run_rag_build_job(job_id: str, db_name: str, mode: str, paths: list[str]) -> None:
        try:
            added = await asyncio.to_thread(_execute_rag_build_job, job_id, db_name, mode, paths)
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG build job failed")
            _fail_rag_build_job(job_id, db_name=db_name, error_text=str(exc))
            return

        if config.settings.KG_EXTRACTION_ENABLED:
            try:
                await _run_kg_build_for_job(job_id, db_name)
            except Exception as exc:
                logging.getLogger(__name__).warning("KG build failed, continuing: %s", exc)
                _update_rag_build_job(job_id, "kg_build_error", {"error": str(exc)})

        _complete_rag_build_job(job_id, db_name=db_name, added=added)

    async def _run_kg_build_for_job(job_id: str, db_name: str) -> None:
        """异步运行知识图谱构建（使用新 4 阶段管线 + 断点续跑）"""
        from rag.documents import load_rag_documents_from_persist_dir
        from agent.kg_build_agent import KGBuildAgent

        persist_dir = os.path.join(config.settings.PERSIST_DIR, db_name)
        if not os.path.isdir(persist_dir):
            return

        callback = lambda stage, payload: _update_rag_build_job(job_id, stage, payload)
        callback("kg_build_started", {"doc_name": db_name})

        rag_docs = load_rag_documents_from_persist_dir(persist_dir, SUPPORTED_RAG_EXTENSIONS)
        if not rag_docs:
            callback("kg_build_completed", {"doc_name": db_name})
            return

        agent = KGBuildAgent(db_name=db_name)
        try:
            for rag_doc in rag_docs:
                try:
                    await agent.build_kg_from_document(rag_doc)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "KG build failed for doc '%s': %s",
                        getattr(rag_doc, "doc_name", db_name), exc,
                    )
        finally:
            await agent.close()
        callback("kg_build_completed", {"doc_name": db_name})



    def _compact_tree_document_summary(item: Any) -> dict[str, object]:
        payload = dict(item or {})
        keywords = [str(keyword).strip() for keyword in list(payload.get("keywords") or []) if str(keyword).strip()]
        return {
            "doc_name": str(payload.get("doc_name") or "").strip(),
            "title": str(payload.get("title") or "").strip(),
            "page_count": int(payload.get("page_count") or 0),
            "chunk_count": int(payload.get("chunk_count") or 0),
            "pagination_mode": str(payload.get("pagination_mode") or "").strip(),
            "catalog_count": len(list(payload.get("catalog") or [])),
            "keywords": keywords[:12],
        }

    def _compact_catalog_rows(rows: Any) -> list[dict[str, object]]:
        compact: list[dict[str, object]] = []
        for item in list(rows or []):
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "page": int(item.get("page") or 0),
                    "end_page": int(item.get("end_page") or item.get("page") or 0),
                    "level": int(item.get("level") or 1),
                    "category": str(item.get("category") or "").strip(),
                    "parent_title": str(item.get("parent_title") or "").strip() or None,
                }
            )
        return compact

    def _compact_retrieve_payload(payload: Any) -> dict[str, object]:
        raw = dict(payload or {})
        compact_results: list[dict[str, object]] = []
        for item in list(raw.get("results") or []):
            row = dict(item or {})
            compact_results.append(
                {
                    "score": float(row.get("score") or 0.0),
                    "text": str(row.get("text") or ""),
                    "doc_name": str(row.get("doc_name") or "").strip(),
                    "section_path": str(row.get("section_path") or "").strip() or None,
                    "page": row.get("page"),
                    "page_start": row.get("page_start"),
                    "page_end": row.get("page_end"),
                }
            )
        return {
            "query": str(raw.get("query") or ""),
            "filters": dict(raw.get("filters") or {}),
            "count": int(raw.get("count") or len(compact_results)),
            "results": compact_results,
            "timestamp": raw.get("timestamp"),
        }

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    def _resolve_request_language(explicit_language: Optional[str], accept_language: Optional[str]) -> str:
        if explicit_language:
            return normalize_language(explicit_language)
        return normalize_language(accept_language)

    @app.post("/api/chat")
    async def chat(request: ChatRequest, http_request: Request):
        language = _resolve_request_language(request.language, http_request.headers.get("accept-language"))
        logger = logging.getLogger(__name__)
        agent = get_or_create_agent(
            request.session_id,
            request.model,
            request.temperature,
        )
        rag_db_names: list[str] = []
        rag_candidates: list[str] = []
        if request.rag_db_name:
            rag_candidates.append(request.rag_db_name)
        rag_candidates.extend(request.rag_db_names or [])
        for item in rag_candidates:
            normalized = _normalize_db_name(item)
            if _is_valid_db(normalized) and normalized not in rag_db_names:
                rag_db_names.append(normalized)

        tool_approval.set_auto_approve(request.session_id, request.auto_approve)

        logger.debug(
            "[chat:req] session=%s stream=%s model=%s temp=%s force_agent=%s lang=%s rag=%s allowed_mcp=%s msg_len=%s images=%s messages=%s",
            request.session_id,
            request.stream,
            request.model,
            request.temperature,
            request.force_agent,
            language,
            rag_db_names,
            _normalize_tool_name_list(request.allowed_mcp_tools),
            len(request.message or ""),
            len(request.images or []),
            len(request.messages or []),
        )

        # 如果前端提供了 messages，则直接使用，忽略 OCR 等处理
        if request.stream:
            async def streamer():
                language_token = set_current_language(language)
                try:
                    async for chunk in agent.astream(
                        request.message,
                        image_urls=request.images or [],
                        rag_db_names=rag_db_names,
                        force_agent=request.force_agent,
                        allowed_mcp_tools=_normalize_tool_name_list(request.allowed_mcp_tools),
                        messages=request.messages,
                        conversation_path=request.conversation_path,
                    ):
                        yield chunk
                        if await http_request.is_disconnected():
                            logger.debug("[chat:stream-disconnected] session=%s", request.session_id)
                            await tool_approval.request_stop_generation(request.session_id)
                            break
                except asyncio.CancelledError:
                    logger.debug("[chat:stream-cancelled] session=%s", request.session_id)
                    await tool_approval.request_stop_generation(request.session_id)
                except Exception as exc:
                    logging.getLogger(__name__).exception("Streaming chat failed")
                    yield f"Error: {exc}"
                finally:
                    tool_approval.unregister_running_task(request.session_id)
                    logger.debug("[chat:stream-done] session=%s rag=%s", request.session_id, rag_db_names)
                    reset_current_language(language_token)

            return StreamingResponse(streamer(), media_type="text/plain")

        language_token = set_current_language(language)
        try:
            answer = await agent.achat(
                request.message,
                image_urls=request.images or [],
                rag_db_names=rag_db_names,
                force_agent=request.force_agent,
                allowed_mcp_tools=_normalize_tool_name_list(request.allowed_mcp_tools),
                messages=request.messages,  # 新增参数
                conversation_path=request.conversation_path,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Chat failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            reset_current_language(language_token)

        logger.debug(
            "[chat:resp] session=%s answer_len=%s rag=%s",
            request.session_id,
            len(answer or ""),
            rag_db_names,
        )

        return {"answer": answer}

    @app.post("/api/reset")
    async def reset(request: ResetRequest):
        clear_session(request.session_id)
        return {"status": "ok"}

    @app.post("/api/confirm-tool")
    async def confirm_tool(request: ToolConfirmRequest):
        found = tool_approval.resolve_tool_approval(request.call_id, request.approved)
        if not found:
            raise HTTPException(status_code=404, detail="Tool approval request not found or already expired")
        logger.info(
            "Tool confirmation: session=%s call_id=%s approved=%s",
            request.session_id,
            request.call_id,
            request.approved,
        )
        return {"status": "ok", "approved": request.approved}

    @app.get("/api/pending-tool-approval")
    async def get_pending_tool_approval(session_id: str):
        info = tool_approval.get_pending_approval_for_session(session_id)
        if info is None:
            return {"pending": False}
        return {
            "pending": True,
            "call_id": info["call_id"],
            "tool_name": info["tool_name"],
            "tool_args": info["tool_args"],
            "created_at": info["created_at"],
        }

    @app.post("/api/stop-generation")
    async def stop_generation(request: StopGenerationRequest):
        stopped = await tool_approval.request_stop_generation(request.session_id)
        if stopped:
            logger.info("Generation stopped: session=%s", request.session_id)
        return {"status": "ok", "stopped": stopped}

    @app.get("/api/tool-usage")
    async def tool_usage(session_id: str):
        return get_tool_usage(session_id)

    @app.post("/api/tool-usage/reset")
    async def tool_usage_reset(request: ResetRequest):
        reset_tool_usage(request.session_id)
        return {"status": "ok"}

    @app.get("/api/capabilities")
    async def capabilities():
        caps = get_capabilities()
        return {
            "tool_calling_supported": caps.tool_calling_supported,
            "tool_calling_error": caps.tool_calling_error,
            "multimodal_supported": caps.multimodal_supported,
            "ocr_available": bool(ocr_enabled()),
            "last_checked": caps.last_checked,
        }

    @app.get("/api/debug/tools")
    async def debug_tools():
        if (config.settings.ENV or "").strip().lower() == "prod":
            raise HTTPException(status_code=403, detail="Debug tool API disabled in production")
        items: list[dict[str, Any]] = []
        for tool in registered_tools:
            schema: dict[str, Any] | None = None
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                schema_fn = getattr(args_schema, "model_json_schema", None)
                if callable(schema_fn):
                    try:
                        result = schema_fn()
                        if isinstance(result, dict):
                            schema = result
                        else:
                            schema = None
                    except Exception:
                        schema = None
            items.append(
                {
                    "name": getattr(tool, "name", "tool"),
                    "description": getattr(tool, "description", ""),
                    "schema": schema,
                }
            )
        return {"tools": items}

    @app.get("/api/mcp/tools")
    async def mcp_tools():
        return {"tools": _list_selectable_mcp_tools()}

    @app.post("/api/debug/tool-invoke")
    async def debug_tool_invoke(request: DebugToolInvokeRequest):
        if (config.settings.ENV or "").strip().lower() == "prod":
            raise HTTPException(status_code=403, detail="Debug tool API disabled in production")
        logger = logging.getLogger(__name__)
        tool_name = (request.tool_name or "").strip()
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")

        selected_tool = None
        for tool in registered_tools:
            if getattr(tool, "name", None) == tool_name:
                selected_tool = tool
                break
        if selected_tool is None:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

        rag_db_names: list[str] = []
        rag_candidates: list[str] = []
        if request.rag_db_name:
            rag_candidates.append(request.rag_db_name)
        rag_candidates.extend(request.rag_db_names or [])
        for item in rag_candidates:
            normalized = _normalize_db_name(item)
            if _is_valid_db(normalized) and normalized not in rag_db_names:
                rag_db_names.append(normalized)

        original_db_name = config.settings.RAG_DB_NAME
        original_db_names = list(config.settings.RAG_DB_NAMES)
        config.settings = config.settings.update(
            RAG_DB_NAME=rag_db_names[0] if rag_db_names else None,
            RAG_DB_NAMES=rag_db_names,
        )

        logger.debug(
            "[debug:tool:req] session=%s tool=%s rag=%s args=%s",
            request.session_id,
            tool_name,
            rag_db_names,
            request.tool_args,
        )
        try:
            output = await selected_tool.ainvoke(request.tool_args or {})
            logger.debug(
                "[debug:tool:resp] session=%s tool=%s output=%s",
                request.session_id,
                tool_name,
                str(output)[:500],
            )
            return {
                "tool_name": tool_name,
                "tool_args": request.tool_args or {},
                "rag_db_names": rag_db_names,
                "output": output,
            }
        except Exception as exc:
            logger.exception("Debug tool invoke failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            config.settings = config.settings.update(
                RAG_DB_NAME=original_db_name,
                RAG_DB_NAMES=original_db_names,
            )

    @app.get("/api/settings")
    async def get_settings():
        data = _read_settings_yaml()
        return {"settings": _ordered_settings(data)}

    @app.post("/api/rag/upload")
    async def rag_upload(
        request: Request,
        files: Optional[list[UploadFile]] = File(None),
        file: Optional[UploadFile] = File(None),
        db_name: Optional[str] = Form(None),
    ):
        upload_files: list[UploadFile] = list(files or [])
        if file is not None:
            upload_files.append(file)

        if not upload_files:
            form = await request.form()
            for _, value in form.multi_items():
                if isinstance(value, UploadFile):
                    upload_files.append(value)

        if not upload_files:
            raise HTTPException(status_code=400, detail="No files provided")

        selected_db = _normalize_db_name(db_name) if db_name else None
        if selected_db:
            config.settings = config.settings.update(RAG_DB_NAME=selected_db)
            upload_dir = _db_docs_dir(selected_db)
        else:
            upload_dir = os.path.join(config.settings.DATA_DIR, "uploads")

        os.makedirs(upload_dir, exist_ok=True)

        saved_paths = []
        existing_names = set(os.listdir(upload_dir))
        current_batch_names: set[str] = set()
        for upload in upload_files:
            filename = os.path.basename(upload.filename or "")
            if not filename:
                continue
            stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in allowed_upload_exts:
                await upload.close()
                continue
            candidate = filename
            suffix = 1
            while candidate in existing_names or candidate in current_batch_names:
                candidate = f"{stem}_{suffix}{ext}"
                suffix += 1
            target_path = os.path.join(upload_dir, candidate)
            content = await upload.read()
            with open(target_path, "wb") as handle:
                handle.write(content)
            saved_paths.append(target_path)
            current_batch_names.add(candidate)
            existing_names.add(candidate)
            await upload.close()

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid files saved")

        try:
            added = await asyncio.to_thread(RAGEngine().add_documents_from_paths, saved_paths)
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG upload failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {"status": "ok", "files": len(saved_paths), "added": added}

    @app.post("/api/rag/retrieve")
    async def rag_retrieve(request: RagRetrieveRequest):
        try:
            engine = RAGEngine()
            if (request.query or "").strip():
                result = await asyncio.to_thread(
                    engine.vector_retrieve,
                    request.query or "",
                    request.section,
                    request.page_start,
                    request.page_end,
                    request.regex,
                    request.chunk,
                    request.doc_name,
                    request.limit,
                )
            else:
                result = await asyncio.to_thread(
                    engine.regex_retrieve,
                    request.regex,
                    request.section,
                    request.page_start,
                    request.page_end,
                    request.chunk,
                    request.doc_name,
                    request.limit,
                )
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG retrieval failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result

    @app.post("/api/rag/build")
    async def rag_build(request: RagBuildRequest):
        requested_name = (request.db_name or "").strip()
        selected_db = _normalize_db_name(requested_name) if requested_name else None
        if not selected_db:
            current_name = (config.settings.RAG_DB_NAME or "").strip()
            if current_name:
                try:
                    normalized_current = _normalize_db_name(current_name)
                    if _is_valid_db(normalized_current):
                        selected_db = normalized_current
                except HTTPException:
                    selected_db = None
        if selected_db:
            if not _is_valid_db(selected_db):
                raise HTTPException(status_code=404, detail="Database not found")
            docs_dir = _db_docs_dir(selected_db)
            os.makedirs(docs_dir, exist_ok=True)
            paths = [
                os.path.join(docs_dir, name)
                for name in os.listdir(docs_dir)
                if os.path.isfile(os.path.join(docs_dir, name))
            ]
            paths.sort()
            if not paths:
                return {"status": "ok", "documents": 0}
            job = _register_rag_build_job(selected_db, "rebuild", [os.path.basename(path) for path in paths])
            asyncio.create_task(_run_rag_build_job(job["job_id"], selected_db, "rebuild", paths))
            return {**job, "status": "accepted"}
        try:
            count = await asyncio.to_thread(RAGEngine().rebuild_index)
            RAGEngine().clear_query_cache()
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG rebuild failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "documents": count}

    @app.get("/api/rag/dbs/{db_name}/build-jobs/{job_id}")
    async def rag_db_build_job_status(db_name: str, job_id: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        payload = _snapshot_rag_build_job(job_id)
        if payload is None or payload.get("database") != name:
            raise HTTPException(status_code=404, detail="Build job not found")
        return payload

    @app.get("/api/rag/dbs")
    async def rag_dbs():
        return {"databases": _list_valid_dbs()}

    @app.post("/api/rag/selection")
    async def rag_selection(request: RagSelectionRequest):
        selected: list[str] = []
        for item in request.db_names:
            normalized = _normalize_db_name(item)
            if not _is_valid_db(normalized):
                raise HTTPException(status_code=404, detail=f"Database not found: {normalized}")
            if normalized not in selected:
                selected.append(normalized)

        primary = selected[0] if selected else None
        updated_kwargs: dict[str, object] = {
            "RAG_DB_NAMES": selected,
            "RAG_DB_NAME": primary,
        }
        if request.enable_rag is not None:
            updated_kwargs["ENABLE_RAG"] = request.enable_rag

        config.settings = config.settings.update(**updated_kwargs)
        RAGEngine().clear_query_cache()
        settings_data = _read_settings_yaml()
        settings_data["RAG_DB_NAMES"] = selected
        settings_data["RAG_DB_NAME"] = primary
        if request.enable_rag is not None:
            settings_data["ENABLE_RAG"] = request.enable_rag
        _write_settings_yaml(settings_data)
        return {
            "status": "ok",
            "selected": selected,
            "primary": primary,
            "enable_rag": config.settings.ENABLE_RAG,
        }

    @app.post("/api/rag/dbs/create")
    async def rag_db_create(request: RagDbNameRequest):
        name = _normalize_db_name(request.name)
        db_dir = _db_dir(name)
        docs_dir = _db_docs_dir(name)
        if os.path.exists(db_dir):
            if not os.path.isdir(db_dir):
                raise HTTPException(status_code=400, detail="Path exists but is not a directory")
            if _is_valid_db(name):
                raise HTTPException(status_code=400, detail="Database already exists")
            os.makedirs(docs_dir, exist_ok=True)
            RAGEngine().invalidate_runtime_state(clear_chroma_cache=True)
            RAGEngine().clear_query_cache()
            return {"status": "ok", "database": name, "repaired": True}
        os.makedirs(docs_dir, exist_ok=True)
        RAGEngine().invalidate_runtime_state(clear_chroma_cache=True)
        RAGEngine().clear_query_cache()
        return {"status": "ok", "database": name}

    @app.post("/api/rag/dbs/rename")
    async def rag_db_rename(request: RagDbRenameRequest):
        old_name = _normalize_db_name(request.old_name)
        new_name = _normalize_db_name(request.new_name)
        old_dir = _db_dir(old_name)
        new_dir = _db_dir(new_name)
        if not _is_valid_db(old_name):
            raise HTTPException(status_code=404, detail="Source database not found")
        if os.path.exists(new_dir):
            raise HTTPException(status_code=400, detail="Target database already exists")
        os.rename(old_dir, new_dir)
        selected_names = list(config.settings.RAG_DB_NAMES)
        selected_names = [new_name if item == old_name else item for item in selected_names]
        deduped_names: list[str] = []
        for item in selected_names:
            if item not in deduped_names:
                deduped_names.append(item)
        primary = config.settings.RAG_DB_NAME
        if (primary or "") == old_name:
            primary = new_name
        _persist_rag_selection(primary, deduped_names)
        RAGEngine().invalidate_runtime_state(clear_chroma_cache=True)
        RAGEngine().clear_query_cache()
        return {"status": "ok", "database": new_name}

    @app.post("/api/rag/dbs/clone")
    async def rag_db_clone(request: RagDbCloneRequest):
        source_name = _normalize_db_name(request.source_name)
        target_name = _normalize_db_name(request.target_name)
        source_dir = _db_dir(source_name)
        target_dir = _db_dir(target_name)
        if not _is_valid_db(source_name):
            raise HTTPException(status_code=404, detail="Source database not found")
        if os.path.exists(target_dir):
            raise HTTPException(status_code=400, detail="Target database already exists")
        shutil.copytree(source_dir, target_dir)
        RAGEngine().invalidate_runtime_state(clear_chroma_cache=True)
        RAGEngine().clear_query_cache()
        return {"status": "ok", "database": target_name}

    @app.delete("/api/rag/dbs/{db_name}")
    async def rag_db_delete(db_name: str):
        name = _normalize_db_name(db_name)
        target_dir = _db_dir(name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        shutil.rmtree(target_dir, ignore_errors=True)
        selected_names = [item for item in config.settings.RAG_DB_NAMES if item != name]
        primary = config.settings.RAG_DB_NAME
        if (primary or "") == name:
            primary = selected_names[0] if selected_names else None
        _persist_rag_selection(primary, selected_names)
        RAGEngine().invalidate_runtime_state(clear_chroma_cache=True)
        RAGEngine().clear_query_cache()
        return {"status": "ok", "database": name}

    @app.get("/api/rag/dbs/{db_name}/docs")
    async def rag_db_docs(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        return {"database": name, "documents": _list_db_docs(name)}

    @app.get("/api/rag/dbs/{db_name}/stats")
    async def rag_db_stats(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        return _collect_db_stats(name)

    @app.post("/api/rag/dbs/{db_name}/build")
    async def rag_db_build(db_name: str):
        name = _normalize_db_name(db_name)
        return await rag_build(RagBuildRequest(db_name=name))

    @app.get("/api/rag/dbs/{db_name}/tree/documents")
    async def rag_db_tree_documents(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            with _scoped_rag_db(name):
                documents = await asyncio.to_thread(RAGEngine().list_documents)
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG tree document listing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "database": name,
            "documents": [_compact_tree_document_summary(item) for item in list(documents or [])],
        }

    @app.get("/api/rag/dbs/{db_name}/tree/catalog")
    async def rag_db_tree_catalog(db_name: str, doc_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        if not str(doc_name or "").strip():
            raise HTTPException(status_code=400, detail="doc_name is required")
        try:
            with _scoped_rag_db(name):
                catalog = await asyncio.to_thread(RAGEngine().get_document_catalog, doc_name)
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG tree catalog load failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "database": name,
            "doc_name": str(doc_name or "").strip(),
            "catalog": _compact_catalog_rows(catalog),
        }

    @app.post("/api/rag/dbs/{db_name}/tree/retrieve")
    async def rag_db_tree_retrieve(db_name: str, request: RagRetrieveRequest):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            with _scoped_rag_db(name):
                engine = RAGEngine()
                if (request.query or "").strip():
                    result = await asyncio.to_thread(
                        engine.vector_retrieve,
                        request.query or "",
                        request.section,
                        request.page_start,
                        request.page_end,
                        request.regex,
                        request.chunk,
                        request.doc_name,
                        request.limit,
                    )
                else:
                    result = await asyncio.to_thread(
                        engine.regex_retrieve,
                        request.regex,
                        request.section,
                        request.page_start,
                        request.page_end,
                        request.chunk,
                        request.doc_name,
                        request.limit,
                    )
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG tree retrieval failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _compact_retrieve_payload(result)

    @app.post("/api/rag/dbs/{db_name}/docs/upload")
    async def rag_db_docs_upload(
        db_name: str,
        request: Request,
        files: Optional[list[UploadFile]] = File(None),
        file: Optional[UploadFile] = File(None),
    ):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")

        upload_files: list[UploadFile] = list(files or [])
        if file is not None:
            upload_files.append(file)
        if not upload_files:
            form = await request.form()
            for _, value in form.multi_items():
                if isinstance(value, UploadFile):
                    upload_files.append(value)
        if not upload_files:
            raise HTTPException(status_code=400, detail="No files provided")

        docs_dir = _db_docs_dir(name)
        os.makedirs(docs_dir, exist_ok=True)
        saved_paths = []
        existing_names = set(os.listdir(docs_dir))
        current_batch_names: set[str] = set()
        for upload in upload_files:
            filename = os.path.basename(upload.filename or "")
            if not filename:
                continue
            stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in allowed_upload_exts:
                await upload.close()
                continue
            candidate = filename
            suffix = 1
            while candidate in existing_names or candidate in current_batch_names:
                candidate = f"{stem}_{suffix}{ext}"
                suffix += 1
            target_path = os.path.join(docs_dir, candidate)
            content = await upload.read()
            with open(target_path, "wb") as handle:
                handle.write(content)
            saved_paths.append(target_path)
            current_batch_names.add(candidate)
            existing_names.add(candidate)
            await upload.close()
        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid files saved")

        job = _register_rag_build_job(name, "add", [os.path.basename(path) for path in saved_paths])
        asyncio.create_task(_run_rag_build_job(job["job_id"], name, "add", saved_paths))

        return {
            **job,
            "status": "accepted",
        }

    @app.delete("/api/rag/dbs/{db_name}/docs/{doc_name}")
    async def rag_db_doc_delete(db_name: str, doc_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        safe_doc_name = os.path.basename(doc_name or "")
        if not safe_doc_name:
            raise HTTPException(status_code=400, detail="Invalid document name")

        target_doc = os.path.join(_db_docs_dir(name), safe_doc_name)
        if not os.path.isfile(target_doc):
            raise HTTPException(status_code=404, detail="Document not found")

        os.remove(target_doc)
        remaining_docs = _list_db_docs(name)
        paths = [os.path.join(_db_docs_dir(name), item) for item in remaining_docs]
        try:
            with _scoped_rag_db(name):
                _clear_db_index_artifacts(name)
                rebuilt = await asyncio.to_thread(RAGEngine().rebuild_index_from_paths, paths)
                RAGEngine().clear_query_cache()
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG db document delete failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "status": "ok",
            "database": name,
            "documents": remaining_docs,
            "rebuilt": rebuilt,
        }

    # ═══════════════════════════════════════════════════════════
    # KG (知识图谱) API 端点
    # ═══════════════════════════════════════════════════════════

    def _kg_file(db_name: str) -> str:
        return os.path.join(_db_dir(db_name), "knowledge_graph.sysml")

    def _kg_meta_file(db_name: str) -> str:
        return os.path.join(_db_dir(db_name), "knowledge_graph.meta.json")

    def _load_kg_manager(db_name: str) -> Any:
        """加载数据库的 KG 到 SysMLManager"""
        from sysml.sysml_manager import SysMLManager
        kg_file = _kg_file(db_name)
        mgr = SysMLManager()
        if os.path.isfile(kg_file):
            mgr.load_from_file(kg_file)
        return mgr

    def _kg_entity_detail(mgr: Any, entity: Any) -> dict:
        from scripts.sysml_rag_mcp_server import _entity_summary, _entity_type_name
        summary = _entity_summary(entity, include_body=True)

        # 获取关联关系
        relations = []
        all_rels = mgr.get_all_relations()
        entity_name = getattr(entity, "name", "")
        for rel in all_rels:
            if hasattr(rel, "ends") and rel.ends:
                for end in rel.ends:
                    if end.ref == entity_name or end.ref == entity.qualified_name.split("::")[-1]:
                        rel_info = _entity_summary(rel, include_body=False)
                        relations.append(rel_info)
                        break

        meta = mgr.get_entity_metadata(entity.qualified_name)
        return {
            "qualified_name": entity.qualified_name,
            "name": entity.name,
            "type": type(entity).__name__,
            "human_type": _entity_type_name(entity),
            "description": meta.get("description", ""),
            "aliases": mgr._alias_registry.get_aliases(entity.qualified_name),
            "source_sections": meta.get("source_sections", []),
            "source_text": meta.get("source_text", ""),
            "properties": meta.get("properties", {}),
            "related_relations": relations,
        }

    @app.get("/api/rag/dbs/{db_name}/kg/export")
    async def kg_export(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        kg_file = _kg_file(name)
        if not os.path.isfile(kg_file):
            raise HTTPException(status_code=404, detail="Knowledge graph not built yet")
        return FileResponse(kg_file, media_type="text/plain",
                           filename=f"{name}_knowledge_graph.sysml")

    @app.get("/api/rag/dbs/{db_name}/kg/summary")
    async def kg_summary(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        mgr = _load_kg_manager(name)
        entities = mgr.get_all_entities()
        relations = mgr.get_all_relations()

        type_dist: dict = {}
        for e in entities:
            t = type(e).__name__
            type_dist[t] = type_dist.get(t, 0) + 1

        return {
            "database": name,
            "kg_file": _kg_file(name),
            "has_kg": os.path.isfile(_kg_file(name)),
            "total_entities": len(entities),
            "total_relations": len(relations),
            "type_distribution": type_dist,
        }

    @app.get("/api/rag/dbs/{db_name}/kg/entities")
    async def kg_entities(db_name: str, type_filter: str = "", name_filter: str = ""):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        mgr = _load_kg_manager(name)
        entities = mgr.get_all_entities()

        result = []
        for e in entities:
            e_type = type(e).__name__
            e_name = getattr(e, "name", "")
            if type_filter and type_filter != e_type:
                continue
            if name_filter and name_filter.lower() not in e_name.lower():
                continue
            meta = mgr.get_entity_metadata(e.qualified_name)
            result.append({
                "qualified_name": e.qualified_name,
                "name": e_name,
                "type": e_type,
                "description": meta.get("description", "")[:200],
                "aliases": mgr._alias_registry.get_aliases(e.qualified_name),
                "source_sections": meta.get("source_sections", []),
            })

        return {
            "database": name,
            "total": len(result),
            "entities": result,
        }

    @app.get("/api/rag/dbs/{db_name}/kg/entity/{entity_name:path}")
    async def kg_entity(db_name: str, entity_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        mgr = _load_kg_manager(name)

        entity = mgr.find_definition(entity_name)
        if entity is None:
            found = mgr.find_element(qualified_name=entity_name)
            if found is not None:
                entity = found
        if entity is None:
            raise HTTPException(status_code=404, detail=f"Entity not found: {entity_name}")

        return _kg_entity_detail(mgr, entity)

    @app.post("/api/rag/dbs/{db_name}/kg/search")
    async def kg_search(db_name: str, request: dict):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        query = str(request.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        mgr = _load_kg_manager(name)
        try:
            threshold = float(request.get("threshold", 0.3))
        except (TypeError, ValueError):
            threshold = 0.3
        regex = request.get("regex_pattern")

        results = mgr.search_entities(query, threshold=threshold, regex_pattern=regex)
        return {
            "database": name,
            "query": query,
            "threshold": threshold,
            "total_matches": len(results),
            "matches": results,
        }

    @app.get("/api/rag/dbs/{db_name}/kg/graph")
    async def kg_graph(db_name: str):
        """返回完整知识图谱数据（节点+边），供可视化前端使用。"""
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")
        mgr = _load_kg_manager(name)
        from scripts.sysml_rag_mcp_server import _entity_type_name

        _TYPE_COLORS = {
            "PartDef": "#6bd1f5", "AttributeDef": "#a78bfa", "PortDef": "#f5c26b",
            "ItemDef": "#34d399", "ConnectionDef": "#f87171", "InterfaceDef": "#fb923c",
            "AllocationDef": "#818cf8", "CommandDef": "#f472b6", "RequirementDef": "#94a3b8",
            "PartUsage": "#6bd1f5", "PortUsage": "#f5c26b", "AttributeUsage": "#a78bfa",
            "ConnectionUsage": "#f87171", "InterfaceUsage": "#fb923c", "AllocationUsage": "#818cf8",
        }

        entities = mgr.get_all_entities()
        relations = mgr.get_all_relations()

        # Build entity node list
        entity_ids: set = set()
        nodes = []
        for e in entities:
            eid = getattr(e, "name", "")
            entity_ids.add(eid)
            t = type(e).__name__
            meta = mgr.get_entity_metadata(e.qualified_name)
            nodes.append({
                "id": eid,
                "qname": e.qualified_name,
                "type": t,
                "cntype": _entity_type_name(e),
                "color": _TYPE_COLORS.get(t, "#9aa3b2"),
                "aliases": mgr._alias_registry.get_aliases(e.qualified_name),
                "sections": meta.get("source_sections", []),
                "description": (meta.get("description", "") or "")[:200],
            })

        # ── Helper: resolve an entity by name or alias ──
        def _resolve_entity(name_str: str) -> Optional[str]:
            """Return canonical entity name if found via exact match or alias."""
            if name_str in entity_ids:
                return name_str
            qn = mgr._alias_registry.lookup(name_str)
            if qn:
                parts = qn.split("::")
                last = parts[-1] if parts else name_str
                if last in entity_ids:
                    return last
            return None

        # ── Helper: heuristic parsing of relation name → (source, target) ──
        _REL_TYPE_RE = re.compile(r'_(connection|allocation|interface)$', re.IGNORECASE)
        def _parse_ends_from_name(rel_name: str) -> Optional[tuple]:
            """If relation ends are empty, try to parse source/target from name."""
            m = _REL_TYPE_RE.search(rel_name)
            if not m:
                return None
            core = rel_name[:m.start()]
            if '_' not in core:
                return None
            parts = core.split('_')
            # Strategy 1: exact match both sides (entity name or alias)
            for split in range(1, len(parts)):
                src = '_'.join(parts[:split])
                tgt = '_'.join(parts[split:])
                if _resolve_entity(src) and _resolve_entity(tgt):
                    return (_resolve_entity(src), _resolve_entity(tgt))
            # Strategy 2: greedy longest source match
            for split in range(len(parts) - 1, 0, -1):
                src = '_'.join(parts[:split])
                if _resolve_entity(src):
                    tgt = '_'.join(parts[split:])
                    resolved_tgt = _resolve_entity(tgt)
                    if resolved_tgt:
                        return (_resolve_entity(src), resolved_tgt)
            # Strategy 3: greedy longest target match
            for split in range(1, len(parts)):
                tgt = '_'.join(parts[split:])
                if _resolve_entity(tgt):
                    src = '_'.join(parts[:split])
                    resolved_src = _resolve_entity(src)
                    if resolved_src:
                        return (resolved_src, _resolve_entity(tgt))
            return None

        # Build edge list from relations; only 1 directed edge per relation
        edges_seen: set = set()
        edges = []
        for rel in relations:
            ends = getattr(rel, "ends", None) or []
            rtype = type(rel).__name__
            rcntype = _entity_type_name(rel)
            rel_meta = mgr.get_entity_metadata(rel.qualified_name)
            rdesc = rel_meta.get("description", "") or ""

            if len(ends) >= 2:
                src = ends[0].ref
                tgt = ends[1].ref
                resolved_src = _resolve_entity(src) or src
                resolved_tgt = _resolve_entity(tgt) or tgt
                # Use description as primary label, fallback to name
                label = rdesc[:60] if rdesc else getattr(rel, "name", "")
                key = (resolved_src, resolved_tgt, label)
                if key in edges_seen:
                    continue
                edges_seen.add(key)
                edges.append({
                    "source": resolved_src, "target": resolved_tgt,
                    "label": label,
                    "type": rtype, "cntype": rcntype,
                    "sections": rel_meta.get("source_sections", []),
                    "description": rdesc,
                    "relation_name": getattr(rel, "name", ""),
                })
            else:
                # Heuristic fallback
                rlabel = getattr(rel, "name", "")
                parsed = _parse_ends_from_name(rlabel)
                if parsed:
                    src, tgt = parsed
                    label = rdesc[:60] if rdesc else rlabel
                    key = (src, tgt, label)
                    if key not in edges_seen:
                        edges_seen.add(key)
                        edges.append({
                            "source": src, "target": tgt,
                            "label": label,
                            "type": rtype, "cntype": rcntype,
                            "heuristic": True,
                            "sections": rel_meta.get("source_sections", []),
                            "description": rdesc,
                            "relation_name": rlabel,
                        })

        # Detect isolated nodes (no edge involvement)
        connected_ids: set = set()
        for e in edges:
            connected_ids.add(e["source"])
            connected_ids.add(e["target"])
        isolated = [n for n in nodes if n["id"] not in connected_ids]

        # Connected component analysis via BFS
        adj: dict[str, set] = {n["id"]: set() for n in nodes}
        for e in edges:
            s, t = e["source"], e["target"]
            if s in adj:
                adj[s].add(t)
            if t in adj:
                adj[t].add(s)
        visited: set = set()
        components = []
        for nid in adj:
            if nid not in visited:
                queue = [nid]
                visited.add(nid)
                comp: list = []
                while queue:
                    cur = queue.pop(0)
                    comp.append(cur)
                    for nb in adj.get(cur, set()):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                components.append(comp)

        # Collect all unique sections from nodes and edges
        all_sections: set = set()
        for n in nodes:
            for s in n.get("sections", []) or []:
                all_sections.add(s)
        for e in edges:
            for s in e.get("sections", []) or []:
                all_sections.add(s)

        return {
            "database": name,
            "total_entities": len(nodes),
            "total_relations": len(relations),
            "total_edges": len(edges),
            "isolated_count": len(isolated),
            "connected_components": len(components),
            "component_sizes": sorted([len(c) for c in components], reverse=True),
            "sections": sorted(all_sections),
            "nodes": nodes,
            "edges": edges,
        }

    @app.post("/api/rag/dbs/{db_name}/kg/re-extract")
    async def kg_re_extract(db_name: str):
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")

        docs_dir = _db_docs_dir(name)
        os.makedirs(docs_dir, exist_ok=True)
        paths = [
            os.path.join(docs_dir, item)
            for item in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, item))
        ]
        paths.sort()
        if not paths:
            return {"status": "ok", "message": "No documents to process"}

        # Remove old KG to force rebuild
        kg_file = _kg_file(name)
        if os.path.isfile(kg_file):
            os.remove(kg_file)
        meta_file = _kg_meta_file(name)
        if os.path.isfile(meta_file):
            os.remove(meta_file)

        job = _register_rag_build_job(name, "rebuild", [os.path.basename(p) for p in paths])
        asyncio.create_task(_run_rag_build_job(job["job_id"], name, "rebuild", paths))
        return {**job, "status": "accepted"}

    @app.post("/api/rag/dbs/{db_name}/kg/continue")
    async def kg_continue_build(db_name: str):
        """从断点恢复 KG 构建（不删除已有数据，不重跑 RAG 索引）"""
        name = _normalize_db_name(db_name)
        if not _is_valid_db(name):
            raise HTTPException(status_code=404, detail="Database not found")

        persist_dir = os.path.join(config.settings.PERSIST_DIR, name)
        build_state_file = os.path.join(persist_dir, "knowledge_graph.build.json")

        if not os.path.isfile(build_state_file):
            raise HTTPException(status_code=400,
                                 detail="No build state found; start a full build first")

        from rag.documents import load_rag_documents_from_persist_dir
        from agent.kg_build_agent import KGBuildAgent

        rag_docs = load_rag_documents_from_persist_dir(persist_dir,
                                                        SUPPORTED_RAG_EXTENSIONS)
        if not rag_docs:
            return {"status": "ok", "message": "No documents to process"}

        job_id = str(uuid.uuid4().hex)
        job_entry = {
            "job_id": job_id, "db_name": name, "status": "accepted",
            "phase": "kg_continuing", "progress": 0, "error": None,
            "per_doc": {},
        }
        with rag_build_jobs_lock:
            rag_build_jobs[job_id] = job_entry

        async def _continue_kg_build():
            try:
                agent = KGBuildAgent(db_name=name)
                try:
                    for rag_doc in rag_docs:
                        stats = await agent.build_kg_from_document(rag_doc)
                        job_entry["per_doc"][getattr(rag_doc, "doc_name", name)] = stats
                finally:
                    await agent.close()
                job_entry["status"] = "completed"
                job_entry["progress"] = 100
            except Exception as exc:
                logging.getLogger(__name__).error("KG continue failed: %s", exc)
                job_entry["status"] = "kg_error"
                job_entry["error"] = str(exc)[:500]

        asyncio.create_task(_continue_kg_build())
        return {**job_entry, "status": "accepted"}

    # ═══════════════════════════════════════════════════════════
    # SysML 知识图谱对话
    # ═══════════════════════════════════════════════════════════

    _SYSML_CHAT_SESSIONS: dict[str, Any] = {}

    async def _get_sysml_chat_agent(session_id: str):
        """Get or create a SysML chat agent for this session."""
        if session_id not in _SYSML_CHAT_SESSIONS:
            from importlib import util as _util
            _chat_file = os.path.join(os.path.dirname(__file__), "scripts", "sysml_chat_agent.py")
            _spec = _util.spec_from_file_location("sysml_chat_agent", _chat_file)
            _chat_mod = _util.module_from_spec(_spec)
            _spec.loader.exec_module(_chat_mod)
            agent = _chat_mod.SysMLChatAgent()
            await agent.initialize()
            _SYSML_CHAT_SESSIONS[session_id] = agent
        return _SYSML_CHAT_SESSIONS[session_id]

    @app.post("/api/kg/chat")
    async def kg_chat(request: dict):
        """Chat with the SysML Knowledge Graph agent.

        Body:
            message (str): User message
            session_id (str): Session ID (default: "sysml-default")
            history (list[dict], optional): Chat history
            stream (bool): Whether to stream response

        Returns:
            dict with "response" field
        """
        message = request.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        session_id = request.get("session_id", "sysml-default")
        history = request.get("history")
        stream = request.get("stream", False)

        agent = await _get_sysml_chat_agent(session_id)
        try:
            response = await agent.chat(message, history=history)
            return {"response": response}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/settings")
    async def update_settings_endpoint(request: SettingsUpdateRequest):
        try:
            validated = config.Settings(**request.settings).model_dump() # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _write_settings_yaml(validated)
        config.settings = config.Settings(**validated)

        if request.restart:
            async def delayed_exit():
                await asyncio.sleep(0.2)
                os._exit(0)

            asyncio.create_task(delayed_exit())
        return {"status": "ok"}

    @app.get("/kg/viz.html", response_class=HTMLResponse)
    async def kg_viz_page():
        viz_path = os.path.join(WEB_DIR, "kg_viz.html")
        if not os.path.isfile(viz_path):
            raise HTTPException(status_code=404, detail="kg_viz.html not found")
        with open(viz_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.get("/kg/viz/{db_name:path}", response_class=HTMLResponse)
    async def kg_viz(db_name: str = ""):
        viz_path = os.path.join(WEB_DIR, "kg_viz.html")
        if not os.path.isfile(viz_path):
            raise HTTPException(status_code=404, detail="kg_viz.html not found")
        with open(viz_path, encoding="utf-8") as f:
            html = f.read()
        if db_name:
            html = html.replace("AUTO_DB_NAME", db_name)
        return HTMLResponse(content=html)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nanite Agent Web API")
    parser.add_argument("--data-dir", default=config.settings.DATA_DIR)
    parser.add_argument("--persist-dir", default=config.settings.PERSIST_DIR)
    parser.add_argument("--log-level", default=None, help="Set root logger level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--metrics-port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_settings(args)
    setup_logging()
    start_metrics_server(args.metrics_port)

    try:
        asyncio.run(health_check(include_mcp=False))
    except SystemExit as exc:
        logging.getLogger(__name__).warning(
            "Health check failed; continuing startup: %s", exc
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Health check failed; continuing startup: %s", exc
        )

    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=str(config.settings.LOG_LEVEL).lower(),
        log_config=None,
    )


if __name__ == "__main__":
    main()
