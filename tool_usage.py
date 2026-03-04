from __future__ import annotations

import json
import os
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config


@dataclass
class ToolCallRecord:
    call_id: str
    tool_name: str
    tool_input: str
    tool_output: str
    started_at: float
    ended_at: Optional[float]
    scope_key: str


class ToolUsageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, Dict[str, int]] = {}
        self._calls: Dict[str, List[ToolCallRecord]] = {}
        self._call_tool_map: Dict[str, Dict[str, str]] = {}
        self._loaded_sessions: set[str] = set()

    @staticmethod
    def _safe_session_name(session_id: str) -> str:
        normalized = (session_id or "default").strip() or "default"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in normalized)
        return safe[:120] or "default"

    def _store_dir(self) -> Path:
        base_dir = Path(config.settings.PERSIST_DIR).resolve()
        target = base_dir / "tool_usage"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _session_file(self, session_id: str) -> Path:
        return self._store_dir() / f"{self._safe_session_name(session_id)}.json"

    def _ensure_loaded(self, session_id: str) -> None:
        if session_id in self._loaded_sessions:
            return

        self._loaded_sessions.add(session_id)
        file_path = self._session_file(session_id)
        if not file_path.exists():
            return

        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return

        counts_raw = payload.get("counts") if isinstance(payload, dict) else {}
        calls_raw = payload.get("calls") if isinstance(payload, dict) else []
        counts: Dict[str, int] = {}
        if isinstance(counts_raw, dict):
            for key, value in counts_raw.items():
                if isinstance(key, str):
                    try:
                        counts[key] = int(value)
                    except Exception:
                        continue

        calls: List[ToolCallRecord] = []
        call_tool_map: Dict[str, str] = {}
        if isinstance(calls_raw, list):
            for item in calls_raw:
                if not isinstance(item, dict):
                    continue
                call_id = str(item.get("call_id") or "").strip()
                tool_name = str(item.get("tool_name") or "").strip()
                if not call_id or not tool_name:
                    continue
                try:
                    started_at = float(item.get("started_at") or 0.0)
                except Exception:
                    started_at = 0.0
                ended_value = item.get("ended_at")
                ended_at: Optional[float]
                if ended_value is None:
                    ended_at = None
                else:
                    try:
                        ended_at = float(ended_value)
                    except Exception:
                        ended_at = None
                record = ToolCallRecord(
                    call_id=call_id,
                    tool_name=tool_name,
                    tool_input=str(item.get("tool_input") or ""),
                    tool_output=str(item.get("tool_output") or ""),
                    started_at=started_at,
                    ended_at=ended_at,
                    scope_key=str(item.get("scope_key") or ""),
                )
                calls.append(record)
                call_tool_map[call_id] = tool_name

        self._counts[session_id] = counts
        self._calls[session_id] = calls
        self._call_tool_map[session_id] = call_tool_map

    def _save(self, session_id: str) -> None:
        file_path = self._session_file(session_id)
        payload = {
            "counts": self._counts.get(session_id, {}),
            "calls": [
                {
                    "call_id": record.call_id,
                    "tool_name": record.tool_name,
                    "tool_input": record.tool_input,
                    "tool_output": record.tool_output,
                    "started_at": record.started_at,
                    "ended_at": record.ended_at,
                    "scope_key": record.scope_key,
                }
                for record in self._calls.get(session_id, [])
            ],
        }

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, file_path)

    def record_start(self, session_id: str, call_id: str, tool_name: str, tool_input: str, scope_key: str) -> None:
        with self._lock:
            self._ensure_loaded(session_id)
            self._counts.setdefault(session_id, {})
            self._call_tool_map.setdefault(session_id, {})[call_id] = tool_name
            self._calls.setdefault(session_id, []).append(
                ToolCallRecord(
                    call_id=call_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output="",
                    started_at=time.time(),
                    ended_at=None,
                    scope_key=scope_key,
                )
            )
            self._save(session_id)

    def record_end(self, session_id: str, call_id: str, tool_output: str) -> None:
        with self._lock:
            self._ensure_loaded(session_id)
            self._counts.setdefault(session_id, {})
            tool_name = self._call_tool_map.get(session_id, {}).get(call_id, call_id)
            self._counts[session_id][tool_name] = self._counts[session_id].get(tool_name, 0) + 1
            calls = self._calls.get(session_id, [])
            for record in reversed(calls):
                if record.call_id == call_id and record.ended_at is None:
                    record.tool_output = tool_output
                    record.ended_at = time.time()
                    break
            self._save(session_id)

    def get_usage(self, session_id: str) -> Dict[str, object]:
        with self._lock:
            self._ensure_loaded(session_id)
            counts = self._counts.get(session_id, {}).copy()
            calls = list(self._calls.get(session_id, []))
        return {
            "counts": counts,
            "calls": [
                {
                    "call_id": record.call_id,
                    "tool_name": record.tool_name,
                    "tool_input": record.tool_input,
                    "tool_output": record.tool_output,
                    "started_at": record.started_at,
                    "ended_at": record.ended_at,
                    "scope_key": record.scope_key,
                }
                for record in calls
            ],
        }

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._ensure_loaded(session_id)
            self._counts.pop(session_id, None)
            self._calls.pop(session_id, None)
            self._call_tool_map.pop(session_id, None)
            self._loaded_sessions.add(session_id)
            file_path = self._session_file(session_id)
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass


_STORE = ToolUsageStore()
_CURRENT_SESSION_ID: ContextVar[str] = ContextVar("current_tool_usage_session", default="default")
_CURRENT_SCOPE_KEY: ContextVar[str] = ContextVar("current_tool_usage_scope", default="")


def _stringify(value: object, limit: int = 200000) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def record_tool_start(session_id: str, call_id: str, tool_name: str, tool_input: object) -> None:
    _STORE.record_start(session_id, call_id, tool_name, _stringify(tool_input), get_current_scope_key())


def record_tool_end(session_id: str, call_id: str, tool_output: object) -> None:
    _STORE.record_end(session_id, call_id, _stringify(tool_output))


def get_tool_usage(session_id: str) -> Dict[str, object]:
    return _STORE.get_usage(session_id)


def reset_tool_usage(session_id: str) -> None:
    _STORE.reset(session_id)


def set_current_session_id(session_id: str) -> Token:
    return _CURRENT_SESSION_ID.set(session_id or "default")


def reset_current_session_id(token: Token) -> None:
    _CURRENT_SESSION_ID.reset(token)


def get_current_session_id() -> str:
    return _CURRENT_SESSION_ID.get()


def set_current_scope_key(scope_key: str) -> Token:
    return _CURRENT_SCOPE_KEY.set((scope_key or "").strip())


def reset_current_scope_key(token: Token) -> None:
    _CURRENT_SCOPE_KEY.reset(token)


def get_current_scope_key() -> str:
    return _CURRENT_SCOPE_KEY.get()


def start_current_tool_call(tool_name: str, tool_input: object) -> str:
    call_id = f"manual-{tool_name}-{time.time_ns()}"
    record_tool_start(get_current_session_id(), call_id, tool_name, tool_input)
    return call_id


def end_current_tool_call(call_id: str, tool_output: object) -> None:
    record_tool_end(get_current_session_id(), call_id, tool_output)
