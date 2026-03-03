from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ToolCallRecord:
    call_id: str
    tool_name: str
    tool_input: str
    tool_output: str
    started_at: float
    ended_at: Optional[float]


class ToolUsageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, Dict[str, int]] = {}
        self._calls: Dict[str, List[ToolCallRecord]] = {}
        self._call_tool_map: Dict[str, Dict[str, str]] = {}

    def record_start(self, session_id: str, call_id: str, tool_name: str, tool_input: str) -> None:
        with self._lock:
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
                )
            )

    def record_end(self, session_id: str, call_id: str, tool_output: str) -> None:
        with self._lock:
            self._counts.setdefault(session_id, {})
            tool_name = self._call_tool_map.get(session_id, {}).get(call_id, call_id)
            self._counts[session_id][tool_name] = self._counts[session_id].get(tool_name, 0) + 1
            calls = self._calls.get(session_id, [])
            for record in reversed(calls):
                if record.call_id == call_id and record.ended_at is None:
                    record.tool_output = tool_output
                    record.ended_at = time.time()
                    break

    def get_usage(self, session_id: str) -> Dict[str, object]:
        with self._lock:
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
                }
                for record in calls
            ],
        }

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._counts.pop(session_id, None)
            self._calls.pop(session_id, None)
            self._call_tool_map.pop(session_id, None)


_STORE = ToolUsageStore()
_CURRENT_SESSION_ID: ContextVar[str] = ContextVar("current_tool_usage_session", default="default")


def _stringify(value: object, limit: int = 800) -> str:
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
    _STORE.record_start(session_id, call_id, tool_name, _stringify(tool_input))


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


def start_current_tool_call(tool_name: str, tool_input: object) -> str:
    call_id = f"manual-{tool_name}-{time.time_ns()}"
    record_tool_start(get_current_session_id(), call_id, tool_name, tool_input)
    return call_id


def end_current_tool_call(call_id: str, tool_output: object) -> None:
    record_tool_end(get_current_session_id(), call_id, tool_output)
