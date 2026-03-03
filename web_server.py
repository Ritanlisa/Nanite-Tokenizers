import argparse
import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import asynccontextmanager
from typing import Any, Optional

import yaml

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from capabilities import get_capabilities
from agent.agent import SmartAgent
from agent.tools import tools as registered_tools
from main import health_check, setup_logging
from monitoring import start_metrics_server
from rag.ocr import ocr_enabled
from rag.engine import RAGEngine, SUPPORTED_RAG_EXTENSIONS
from tool_usage import get_tool_usage, reset_tool_usage
from mcp_client.client import get_mcp_client
from locale_context import normalize_language, set_current_language, reset_current_language

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
    messages: Optional[list[dict]] = None
    language: Optional[str] = None


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


SESSION_STORE: dict[str, SmartAgent] = {}
SESSION_PARAMS: dict[str, dict[str, Optional[object]]] = {}


def get_or_create_agent(session_id: str, model: Optional[str], temperature: Optional[float]) -> SmartAgent:
    params = {"model": model, "temperature": temperature}
    if session_id not in SESSION_STORE or SESSION_PARAMS.get(session_id) != params:
        SESSION_STORE[session_id] = SmartAgent(
            session_id=session_id,
            model=model,
            temperature=temperature,
        )
        SESSION_PARAMS[session_id] = params
    return SESSION_STORE[session_id]


def clear_session(session_id: str) -> None:
    SESSION_STORE.pop(session_id, None)
    SESSION_PARAMS.pop(session_id, None)
    reset_tool_usage(session_id)


def update_settings(args: argparse.Namespace) -> None:
    config.settings = config.settings.update(
        DATA_DIR=args.data_dir,
        PERSIST_DIR=args.persist_dir,
        AGENT_VERBOSE=args.verbose or config.settings.AGENT_VERBOSE,
        LOG_LEVEL=(str(args.log_level).upper() if getattr(args, "log_level", None) else ("DEBUG" if args.verbose else config.settings.LOG_LEVEL)),
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger = logging.getLogger(__name__)
        try:
            client = get_mcp_client()
            await client.initialize()
            if getattr(client, "_fallback_mode", False):
                logger.warning("MCP server unavailable or disabled; using direct HTTP fallback")
            else:
                logger.info("MCP server reachable")
        except Exception as exc:
            logger.warning("MCP init failed at app startup: %s", exc)
        yield
        await get_mcp_client().close()

    app = FastAPI(title="Nanite Agent API", lifespan=lifespan)
    db_name_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
    allowed_upload_exts = set(SUPPORTED_RAG_EXTENSIONS)

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

        logger.debug(
            "[chat:req] session=%s stream=%s model=%s temp=%s force_agent=%s lang=%s rag=%s msg_len=%s images=%s messages=%s",
            request.session_id,
            request.stream,
            request.model,
            request.temperature,
            request.force_agent,
            language,
            rag_db_names,
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
                        messages=request.messages,  # 新增参数
                    ):
                        yield chunk
                except Exception as exc:
                    logging.getLogger(__name__).exception("Streaming chat failed")
                    yield f"Error: {exc}"
                finally:
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
                messages=request.messages,  # 新增参数
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
                        schema = schema_fn()
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
                    None,
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
                    None,
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
            config.settings = config.settings.update(RAG_DB_NAME=selected_db)
            docs_dir = _db_docs_dir(selected_db)
            os.makedirs(docs_dir, exist_ok=True)
            paths = [
                os.path.join(docs_dir, name)
                for name in os.listdir(docs_dir)
                if os.path.isfile(os.path.join(docs_dir, name))
            ]
            paths.sort()
            _clear_db_index_artifacts(selected_db)
        try:
            if selected_db:
                count = await asyncio.to_thread(RAGEngine().rebuild_index_from_paths, paths) # type: ignore
            else:
                count = await asyncio.to_thread(RAGEngine().rebuild_index)
            RAGEngine().clear_query_cache()
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG rebuild failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "documents": count}

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

        config.settings = config.settings.update(RAG_DB_NAME=name)
        try:
            added = await asyncio.to_thread(RAGEngine().add_documents_from_paths, saved_paths)
            RAGEngine().clear_query_cache()
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG db document upload failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "status": "ok",
            "database": name,
            "files": len(saved_paths),
            "added": added,
            "documents": _list_db_docs(name),
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
        config.settings = config.settings.update(RAG_DB_NAME=name)
        _clear_db_index_artifacts(name)
        paths = [os.path.join(_db_docs_dir(name), item) for item in remaining_docs]
        try:
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
    )


if __name__ == "__main__":
    main()
