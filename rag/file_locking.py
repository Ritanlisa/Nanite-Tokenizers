from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any

def _lock_shared_file(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_SH)


def _lock_exclusive_file(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)


def _read_json_file_locked(path: str, default: Any) -> Any:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        _lock_shared_file(handle)
        handle.seek(0)
        try:
            raw = handle.read()
            if not str(raw).strip():
                return copy.deepcopy(default)
            return json.loads(raw)
        except json.JSONDecodeError:
            return copy.deepcopy(default)
        finally:
            _unlock_file(handle)


def _write_json_file_locked(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        _lock_exclusive_file(handle)
        try:
            json.dump(payload, handle, ensure_ascii=False)
        finally:
            _unlock_file(handle)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}

    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        try:
            return _json_safe_value(to_payload())
        except TypeError:
            pass
        except Exception:
            return str(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_value(item())
        except Exception:
            pass

    return str(value)
