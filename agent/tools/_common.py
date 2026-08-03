from __future__ import annotations

import json
import logging
import os
import random
import re
import shlex
import time
from html import unescape
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from locale_context import get_current_language
from tool_usage import (
    get_current_scope_key,
    get_current_session_id,
    get_tool_usage,
)

logger = logging.getLogger(__name__)

_EMPTY = object()

_TOOL_REF_PATTERN = re.compile(r"^tool\[(-?\d+)\]")
_ACCESSOR_PATTERN = re.compile(
    r"\[(?:(-?\d+)|\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)'|([A-Za-z_][A-Za-z0-9_]*))\]"
)

def _lang() -> str:
    language = (get_current_language() or "zh").strip().lower()
    if language.startswith("en"):
        return "en"
    return "zh"


def _t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def _bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


def _workspace_root() -> Path:
    return Path(os.getcwd()).resolve()


def _safe_path(path: str) -> Path:
    root = _workspace_root()
    target = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(_t("路径超出工作区根目录", "Path escapes workspace root"))
    return target


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_json_safe(item())
        except Exception:
            pass
    return str(value)


def _prune_empty_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _prune_empty_fields(item)
            if normalized is _EMPTY:
                continue
            cleaned[str(key)] = normalized
        return cleaned if cleaned else _EMPTY
    if isinstance(value, list):
        cleaned_list = [item for item in (_prune_empty_fields(item) for item in value) if item is not _EMPTY]
        return cleaned_list if cleaned_list else _EMPTY
    if isinstance(value, str):
        return _EMPTY if value == "" else value
    if value is None:
        return _EMPTY
    return value


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<.*?>", "", text or "")).strip()


def _sleep_with_jitter(base: float, jitter: float = 0.35) -> None:
    delay = max(0.0, base + random.uniform(0.0, max(0.0, jitter)))
    time.sleep(delay)


def _try_parse_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _select_tool_call_output(call_index: int) -> Any:
    usage = get_tool_usage(get_current_session_id())
    calls_obj = usage.get("calls") if isinstance(usage, dict) else []
    calls: list[Any] = calls_obj if isinstance(calls_obj, list) else []
    completed = _filter_completed_calls_by_scope(calls)
    if not completed:
        logger.debug(f"session id {get_current_session_id()} scope {get_current_scope_key()} calls: {calls}")
        raise ValueError(_t("当前没有可用的已完成工具调用。", "No completed tool calls available yet."))

    if call_index < 0:
        idx = len(completed) + call_index
    else:
        if call_index == 0:
            raise ValueError(_t("tool[0] 无效，正数索引从 1 开始。", "tool[0] is invalid; positive indices start at 1."))
        idx = call_index - 1

    if idx < 0 or idx >= len(completed):
        raise ValueError(
            _t(
                f"tool[{call_index}] 越界，当前已完成调用数为 {len(completed)}。",
                f"tool[{call_index}] is out of range; completed calls: {len(completed)}.",
            )
        )

    selected_call = completed[idx]
    output_text = str(selected_call.get("tool_output") or "")
    parsed = _try_parse_json(output_text)
    call_id = str(selected_call.get("call_id") or "")
    return _inject_hidden_links(parsed, call_id)


def _is_scope_ancestor(ancestor: str, descendant: str) -> bool:
    a = (ancestor or "").strip()
    d = (descendant or "").strip()
    if not a:
        return False
    return d == a or d.startswith(f"{a}>")


def _filter_completed_calls_by_scope(calls: list[Any]) -> list[dict[str, Any]]:
    completed = [item for item in calls if isinstance(item, dict) and item.get("ended_at") is not None]
    scope_key = (get_current_scope_key() or "").strip()
    if not scope_key:
        return completed

    scoped = [
        item
        for item in completed
        if _is_scope_ancestor(str(item.get("scope_key") or ""), scope_key)
    ]
    return scoped


def _inject_hidden_links(parsed_output: Any, call_id: str) -> Any:
    from .rag_tools import _HIDDEN_LINK_STORE
    links = _HIDDEN_LINK_STORE.get(call_id) if call_id else None
    if not links:
        return parsed_output

    def _attach(items: list[Any]) -> list[Any]:
        result: list[Any] = []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                row = dict(item)
                if index < len(links):
                    row["_url"] = links[index]
                result.append(row)
            else:
                result.append(item)
        return result

    if isinstance(parsed_output, list):
        return _attach(parsed_output)
    if isinstance(parsed_output, dict):
        results = parsed_output.get("results")
        if isinstance(results, list):
            cloned = dict(parsed_output)
            cloned["results"] = _attach(results)
            return cloned
    return parsed_output


def _apply_accessor_chain(base_value: Any, chain_text: str) -> Any:
    value = base_value
    cursor = 0
    for match in _ACCESSOR_PATTERN.finditer(chain_text or ""):
        if match.start() != cursor:
            raise ValueError(_t("无效的 tool 访问器语法。", "Invalid tool accessor syntax."))
        cursor = match.end()

        int_part = match.group(1)
        dq_key = match.group(2)
        sq_key = match.group(3)
        ident_key = match.group(4)

        if int_part is not None:
            key: Any = int(int_part)
        else:
            if ident_key is not None:
                key = ident_key
            else:
                key_text = dq_key if dq_key is not None else (sq_key or "")
                key = bytes(key_text, "utf-8").decode("unicode_escape")

        if isinstance(value, (list, tuple)):
            if not isinstance(key, int):
                raise ValueError(_t("列表索引必须是整数。", "List index must be an integer."))
            # 处理索引转换
            if key < 0:
                # 负数索引：转换为Python索引
                index = len(value) + key
            elif key > 0:
                # 正数索引：1-based → 0-based
                index = key - 1
            else:  # key == 0
                raise ValueError(_t("索引0无效，正数索引从1开始。", "Index 0 is invalid; positive indices start at 1."))
            if index < 0 or index >= len(value):
                raise ValueError(_t("列表索引越界。", "List index out of range."))
            value = value[index]
            continue

        if isinstance(value, dict):
            dict_key: Any = key
            if dict_key not in value and isinstance(dict_key, int):
                dict_key = str(dict_key)
            if dict_key == "link" and "_url" in value:
                value = value["_url"]
                continue
            if dict_key not in value:
                raise ValueError(_t("字典键不存在。", "Dictionary key does not exist."))
            value = value[dict_key]
            continue
        
        logger.error("Unsupported accessor on value: %s (type: %s)", value, type(value).__name__)
        raise ValueError(_t("当前值不支持继续索引。", "Current value is not indexable."))

    if cursor != len(chain_text or ""):
        raise ValueError(_t("无效的 tool 访问器语法。", "Invalid tool accessor syntax."))
    return value


def _render_regex_replacement(template: str, matched: re.Match[str]) -> str:
    result = template.replace("$0", matched.group(0))
    for idx, group in enumerate(matched.groups(), start=1):
        result = result.replace(f"${idx}", group or "")
    return result


def _apply_pipe_regex(value: Any, spec: str) -> Any:
    text = str(value)
    parts = shlex.split(spec)
    if not parts or parts[0].strip().lower() != "regex":
        raise ValueError(_t("仅支持 | regex ... 管道。", "Only | regex ... pipe is supported."))
    if len(parts) < 2:
        raise ValueError(_t("regex 管道缺少 pattern。", "regex pipe requires a pattern."))

    pattern_text = parts[1]
    repl = "$0" if len(parts) == 2 else " ".join(parts[2:])
    pattern = re.compile(pattern_text, re.S)
    matched = pattern.search(text)
    if not matched:
        return ""
    return _render_regex_replacement(repl, matched)


def _resolve_tool_reference_sugar(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_tool_reference_sugar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tool_reference_sugar(item) for item in value]
    if not isinstance(value, str):
        return value

    source = value.strip()
    if not source.startswith("tool["):
        return value

    segments = [segment.strip() for segment in source.split("|")]
    if not segments:
        return value

    expr = segments[0]
    match = _TOOL_REF_PATTERN.match(expr)
    if not match:
        return value

    call_index = int(match.group(1))
    accessor_text = expr[match.end():].strip()
    resolved: Any = _select_tool_call_output(call_index)

    if accessor_text:
        resolved = _apply_accessor_chain(resolved, accessor_text)

    for pipe_spec in segments[1:]:
        if not pipe_spec:
            continue
        resolved = _apply_pipe_regex(resolved, pipe_spec)

    _mark_sugar_url_candidates(resolved)
    return resolved

def _mark_sugar_url_candidates(value: Any) -> None:
    from .rag_tools import _SUGAR_URL_MARKS
    if isinstance(value, dict):
        for item in value.values():
            _mark_sugar_url_candidates(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _mark_sugar_url_candidates(item)
        return
    if not isinstance(value, str):
        return

    text = value.strip()
    if not text:
        return
    normalized = text if re.match(r"^https?://", text, re.I) else f"https://{text}"
    if re.match(r"^https?://", normalized, re.I):
        _SUGAR_URL_MARKS[normalized] = _SUGAR_URL_MARKS.get(normalized, 0) + 1


def _consume_sugar_url_mark(url: str) -> bool:
    from .rag_tools import _SUGAR_URL_MARKS
    count = int(_SUGAR_URL_MARKS.get(url) or 0)
    if count <= 0:
        return False
    if count == 1:
        _SUGAR_URL_MARKS.pop(url, None)
    else:
        _SUGAR_URL_MARKS[url] = count - 1
    return True

class InputSugarTool(BaseTool):
    @staticmethod
    def _normalize_tool_input(input_value: Any) -> Any:
        if isinstance(input_value, dict):
            return _resolve_tool_reference_sugar(input_value)
        return input_value

    async def ainvoke(self, input: Any = None, config: Any = None, **kwargs: Any) -> Any:
        normalized_input = self._normalize_tool_input(input)
        return await super().ainvoke(normalized_input, config=config, **kwargs)

    def invoke(self, input: Any = None, config: Any = None, **kwargs: Any) -> Any:
        normalized_input = self._normalize_tool_input(input)
        return super().invoke(normalized_input, config=config, **kwargs)
