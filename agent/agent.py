from __future__ import annotations

import asyncio
import ast
import html
import json
import logging
import re
import time
from typing import Any, Optional, cast
from uuid import UUID

from langchain.agents import create_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool
from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
from pydantic import SecretStr
try:
    from langgraph.errors import GraphRecursionError
except Exception:  # pragma: no cover - optional dependency
    GraphRecursionError = None

from agent.memory import get_chat_memory
import concurrent.futures
from agent.tools import tools
import config
from capabilities import get_capabilities
from exceptions import TokenLimitExceeded
from rag.engine import RAGEngine
from monitoring import agent_token_usage
from locale_context import get_current_language
from tool_usage import (
    record_tool_end,
    record_tool_start,
    set_current_session_id,
    reset_current_session_id,
    set_current_scope_key,
    reset_current_scope_key,
)
from contextvars import ContextVar

logger = logging.getLogger(__name__)


# Simple context counter to mark agent active state (used to avoid accidental recursion)
_agent_active_ctx: ContextVar[int] = ContextVar("agent_active", default=0)


def set_agent_active() -> int:
    prev = _agent_active_ctx.get()
    _agent_active_ctx.set(prev + 1)
    return prev


def reset_agent_active(prev: int) -> None:
    _agent_active_ctx.set(prev)


def _lang() -> str:
    language = (get_current_language() or "zh").strip().lower()
    if language.startswith("en"):
        return "en"
    return "zh"


def _t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def _bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


class TokenLimitCallback(BaseCallbackHandler):
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = int(max_tokens or 0)
        self._count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:  # pragma: no cover - simple guard
        # approximate tokens by counting new token callbacks
        self._count += 1
        if self.max_tokens and self._count > self.max_tokens:
            raise TokenLimitExceeded("token limit exceeded")

class SmartAgent:
    def __init__(
        self,
        session_id: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.session_id = session_id
        self.memory = get_chat_memory(session_id)
        resolved_api_key = api_key or config.settings.OPENAI_API_KEY
        self.llm = ChatOpenAIWithReasoning(
            model=model or config.settings.LLM_MODEL,
            temperature=(
                temperature if temperature is not None else config.settings.TEMPERATURE
            ),
            api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
            base_url=api_url or config.settings.OPENAI_API_URL,
            timeout=getattr(config.settings, "LLM_REQUEST_TIMEOUT", 120),
            streaming=True,
        )
        caps = get_capabilities()
        if caps.tool_calling_supported is False:
            self.tools: list[BaseTool] = []
        else:
            self.tools = list(tools)
        self.agent = self._create_agent()

    @staticmethod
    def _normalize_tool_names(values: Optional[list[str]]) -> list[str]:
        if not values:
            return []
        deduped: list[str] = []
        for item in values:
            name = str(item or "").strip()
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    @staticmethod
    def _is_rag_tool_name(name: str) -> bool:
        return str(name or "").strip().startswith("rag_")

    def _select_tools_for_request(self, allowed_mcp_tools: Optional[list[str]]) -> list[BaseTool]:
        if not self.tools:
            return []
        allowed_names = self._normalize_tool_names(allowed_mcp_tools)
        if not allowed_names:
            return list(self.tools)
        allowed_set = set(allowed_names)
        selected: list[BaseTool] = []
        for tool in self.tools:
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not tool_name:
                continue
            if self._is_rag_tool_name(tool_name) or tool_name in allowed_set:
                selected.append(tool)
        return selected

    @staticmethod
    def _tool_sugar_protocol_note() -> str:
        lines = [
            _bi(
                "MCP 工具输入协议扩展（tool 引用语法糖）：",
                "MCP Tool Input Protocol Extension (tool reference sugar):"
            ),
            _bi(
                "- 在任何工具输入字符串中，可以使用 `tool[index]` 引用历史工具输出。",
                "- In any tool input string, you may reference historical tool outputs with `tool[index]`."
            ),
            _bi(
                "  - 索引规则：正数从 1 开始，表示从最早的工具调用起算；负数从 -1 开始，表示从最近的工具调用倒数。`tool[0]` 无效。",
                "  - Index rules: positive indices start at 1 (earliest call), negative indices start at -1 (most recent call). `tool[0]` is invalid."
            ),
            _bi(
                "  - 示例：`tool[-1]` 最近一次工具输出，`tool[1]` 第一次工具输出。",
                "  - Example: `tool[-1]` refers to the most recent tool output, `tool[1]` to the first tool output."
            ),
            _bi(
                "- 支持访问器链，通过 `[key]` 或 `[index]` 逐级提取数据。",
                "- Accessor chains are supported: use `[key]` or `[index]` to drill down."
            ),
            _bi(
                "  - 对于列表：**索引与结果在返回列表中的显示序号一致**，第一个结果的索引为 1，第二个为 2，以此类推。",
                "  - For lists: **the index corresponds to the displayed order in the returned list**; the first result has index 1, the second index 2, etc."
            ),
            _bi(
                "  - 示例：若 `skill_search` 返回了 5 条结果，要引用第三条结果的链接，应使用 `tool[-1][3][link]`。",
                "  - Example: if `skill_search` returned 5 results, to reference the link of the third result, use `tool[-1][3][link]`."
            ),
            _bi(
                "- 支持简单的管道变换：`| regex <pattern> [replacement]`，用于从字符串中提取部分内容。",
                "- Simple pipe transform is available: `| regex <pattern> [replacement]`, used to extract part of a string."
            ),
            _bi(
                "  - `replacement` 默认为 `$0`（完整匹配），可使用 `$1`、`$2` 等引用捕获组。",
                "  - `replacement` defaults to `$0` (full match); `$1`, `$2`… can be used for captured groups."
            ),
            _bi(
                "  - 示例：`tool[-1][3][link] | regex https?://(.*?)/ $1` 可提取域名。",
                "  - Example: `tool[-1][3][link] | regex https?://(.*?)/ $1` extracts the domain."
            ),
            _bi(
                "⚠️ 重要提示：直接使用上述语法即可，所有索引已按规则自动转换，无需自行加/减 1。",
                "⚠️ Important: Use the syntax exactly as described; indices are automatically converted according to the rules, so you do not need to add/subtract 1 manually."
            ),
            _bi(
                "若引用无效或越界，工具调用将失败并返回明确错误信息。",
                "If the reference is invalid or out of bounds, the tool call will fail with a clear error message."
            ),
        ]
        return "\n\n" + "\n".join(lines)

    def _create_agent(self, selected_tools: Optional[list[BaseTool]] = None):
        tools_for_agent = list(selected_tools) if selected_tools is not None else list(self.tools)
        protocol_note = self._tool_sugar_protocol_note() if tools_for_agent else ""
        custom_prompt = (config.settings.SYSTEM_PROMPT or "").strip()
        if custom_prompt:
            system_prompt = custom_prompt + protocol_note
        elif tools_for_agent:
            system_prompt = (
                _bi(
                    "你是一个可使用检索与执行工具的智能助手。", 
                    "You are a helpful assistant with access to retrieval and execution tools."
                    ) + "\n"
                + _bi(
                    "策略：", 
                    "Strategy:"
                    ) + "\n"
                + _bi(
                    "1) 优先使用 rag_search 检索内部知识。事实证明， rag_doc_list -> rag_doc_catalog -> rag_regex_search 的执行流是十分高效且可靠的内部知识检索方案。", 
                    "1) Use rag_search first for internal knowledge. The rag_doc_list -> rag_doc_catalog -> rag_regex_search flow is an efficient and reliable internal retrieval solution."
                    ) + "\n"
                + _bi(
                    "   建议 rag_regex_search 中使用 可选捕获组()? + capture_group_weights 来按照你的需求排序搜索结果，每个捕获组对应一个权重。", 
                    "   It is recommended to use optional capture groups ()? + capture_group_weights in rag_regex_search to sort search results according to your needs, with each capture group corresponding to a weight."
                    ) + "\n"
                + _bi(
                    "2) 若 rag_search 无效，再使用 fetch_webpage、skill_web_visit 或 skill_search。",
                    "2) If rag_search is not useful, use fetch_webpage, skill_web_visit, or skill_search.",
                ) + "\n"
                + _bi(
                    "3) 需要工作区文件读写时使用 skill_file_io。",
                    "3) Use skill_file_io for workspace-local file read/write tasks.",
                ) + "\n"
                + _bi(
                    "4) 需要安全执行命令时使用 skill_shell。",
                    "4) Use skill_shell for safe command execution when needed.",
                ) + "\n"
                + _bi(
                    "5) 回答应基于检索内容与工具输出。",
                    "5) Ground answers on retrieved content and tool output.",
                ) + "\n"
                + _bi(
                    "6) 工具失败时应重试或说明限制。",
                    "6) If tools fail, retry or explain limitations.",
                ) + "\n"
                + _bi(
                    "⚠️**关键警告**⚠️",
                    "⚠️**Critical Warning**⚠️",
                ) + "\n"
                + _bi(
                    "⚠️ 你不应该在回答/思考中包含任何未经工具验证的内容，除非你明确说明这是基于模型推测的结论，并且可能不准确，否则你可能会造成难以挽回的损失。",
                    "⚠️ You should not include any content in your answers/thoughts that has not been verified by tools, unless you explicitly state that it is based on model inference and may be inaccurate. Failure to do so may lead to irreparable consequences."
                ) + "\n"
                + _bi(
                    "⚠️ 你不应该在回答/思考中直接提及 MCP 工具返回的无自然语言意义的长字符串（比如 URL 等），这可能会导致 LLM 对关键内容的错误复制或者引发 LLM 输出崩坏，你应该尽量使用 ```tool``` 语法糖代指这一类工具返回结果。",
                    "⚠️ You should avoid directly including long strings with no natural language meaning returned by MCP tools (like URLs) in your answers/thoughts, as this may lead to LLM mis-copying key content or output corruption. You should use the ```tool``` reference sugar to refer to such tool outputs whenever possible."
                )
            ) + protocol_note
        else:
            system_prompt = (
                _bi(
                    "你是一个乐于助人的助手。", 
                    "You are a helpful assistant."
                    )
                + "\n"
                + _bi(
                    "当前模型不可用工具，请直接回答。", 
                    "Tools are unavailable for this model; answer directly."
                )
            )
        return create_agent(
            model=self.llm,
            tools=tools_for_agent,
            system_prompt=system_prompt,
            debug=config.settings.AGENT_VERBOSE,
            name="smart_agent",
        )

    @staticmethod
    def _normalize_rag_db_names(rag_db_names: Optional[list[str]]) -> list[str]:
        if not rag_db_names:
            return []
        deduped: list[str] = []
        for item in rag_db_names:
            name = (item or "").strip()
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    def _apply_rag_scope(self, rag_db_names: Optional[list[str]]) -> tuple[Optional[str], list[str]]:
        original_db_name = config.settings.RAG_DB_NAME
        original_db_names = list(config.settings.RAG_DB_NAMES)
        selected = self._normalize_rag_db_names(rag_db_names)
        config.settings = config.settings.update(
            RAG_DB_NAME=selected[0] if selected else None,
            RAG_DB_NAMES=selected,
        )
        return original_db_name, original_db_names

    @staticmethod
    def _restore_rag_scope(original_db_name: Optional[str], original_db_names: list[str]) -> None:
        config.settings = config.settings.update(
            RAG_DB_NAME=original_db_name,
            RAG_DB_NAMES=original_db_names,
        )

    @staticmethod
    def _short_debug_text(value: Any, limit: int = 300) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...(truncated, len={len(text)})"

    @staticmethod
    def _build_tool_scope_key(conversation_path: Optional[list[str]]) -> str:
        if not conversation_path:
            return ""
        normalized: list[str] = []
        for item in conversation_path:
            text = str(item or "").strip()
            if not text:
                continue
            safe = re.sub(r"[^A-Za-z0-9_.:-]", "_", text)
            if safe:
                normalized.append(safe)
        return ">".join(normalized)

    @staticmethod
    def _is_retryable_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, concurrent.futures.TimeoutError)):
            return True
        type_name = type(exc).__name__.lower()
        if "timeout" in type_name:
            return True
        message = str(exc).lower()
        timeout_keys = (
            "readtimeout",
            "read timeout",
            "connecttimeout",
            "connect timeout",
            "timeout",
            "timed out",
        )
        return any(item in message for item in timeout_keys)

    async def _ainvoke_with_retry(
        self,
        input_messages: list[Any],
        callbacks: list[BaseCallbackHandler],
        agent_runner: Optional[Any] = None,
    ) -> dict:
        runner = agent_runner or self.agent
        retry_times = max(0, int(getattr(config.settings, "AGENT_LLM_RETRY_TIMES", 0)))
        base_delay = float(getattr(config.settings, "AGENT_LLM_RETRY_DELAY", 1.0))
        attempts = retry_times + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = await runner.ainvoke(
                    cast(Any, {"messages": input_messages}),
                    cast(
                        Any,
                        {
                            "callbacks": callbacks,
                            "recursion_limit": config.settings.RECURSION_LIMIT,
                        },
                    ),
                )
                return cast(dict, result)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_timeout_error(exc) or attempt >= attempts:
                    raise
                delay = max(0.0, base_delay * attempt)
                logger.warning(
                    "Agent ainvoke timeout, retrying (%s/%s) after %.2fs: %s",
                    attempt,
                    attempts,
                    delay,
                    self._short_debug_text(exc),
                )
                await asyncio.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("ainvoke retry loop exited unexpectedly")

    def _invoke_with_retry(
        self,
        input_messages: list[Any],
        callbacks: list[BaseCallbackHandler],
        agent_runner: Optional[Any] = None,
    ) -> dict:
        runner = agent_runner or self.agent
        retry_times = max(0, int(getattr(config.settings, "AGENT_LLM_RETRY_TIMES", 0)))
        base_delay = float(getattr(config.settings, "AGENT_LLM_RETRY_DELAY", 1.0))
        invoke_timeout = int(getattr(config.settings, "AGENT_INVOKE_TIMEOUT", 180))
        attempts = retry_times + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        runner.invoke,
                        cast(Any, {"messages": input_messages}),
                        cast(
                            Any,
                            {
                                "callbacks": callbacks,
                                "recursion_limit": config.settings.RECURSION_LIMIT,
                            },
                        ),
                    )
                    try:
                        return cast(dict, fut.result(timeout=invoke_timeout))
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        raise
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_timeout_error(exc) or attempt >= attempts:
                    raise
                delay = max(0.0, base_delay * attempt)
                logger.warning(
                    "Agent invoke timeout, retrying (%s/%s) after %.2fs: %s",
                    attempt,
                    attempts,
                    delay,
                    self._short_debug_text(exc),
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("invoke retry loop exited unexpectedly")

    def _tool_callback(self) -> BaseCallbackHandler:
        session_id = self.session_id

        class ToolUsageCallback(BaseCallbackHandler):
            def __init__(self) -> None:
                self._starts: dict[str, float] = {}

            def _start_if_needed(self, run_id: object, tool_name: object, tool_input: object) -> str:
                if run_id:
                    call_id = str(run_id)
                else:
                    import uuid
                    call_id = str(uuid.uuid4())  # 生成随机 UUID
                if call_id not in self._starts:
                    self._starts[call_id] = time.time()
                    record_tool_start(session_id, call_id, str(tool_name or "tool"), tool_input or "")
                    logger.debug(
                        "[tool:start] session=%s call_id=%s tool=%s input=%s",
                        session_id,
                        call_id,
                        str(tool_name or "tool"),
                        SmartAgent._short_debug_text(tool_input),
                    )
                return call_id

            def on_tool_start(self, serialized, input_str=None, **kwargs):
                tool_name = None
                if isinstance(serialized, dict):
                    tool_name = serialized.get("name")
                tool_name = tool_name or "tool"
                run_id = kwargs.get("run_id") or kwargs.get("parent_run_id") or f"{tool_name}-{time.time_ns()}"
                tool_input = input_str
                if tool_input is None:
                    tool_input = kwargs.get("input")
                if tool_input is None:
                    tool_input = kwargs.get("tool_input")
                if tool_input is None and isinstance(serialized, dict):
                    tool_input = serialized.get("input")
                self._start_if_needed(run_id, tool_name, tool_input)

            def on_agent_action(self, action, **kwargs):
                tool_name = getattr(action, "tool", None) or kwargs.get("tool") or "tool"
                tool_input = (
                    getattr(action, "tool_input", None)
                    or kwargs.get("tool_input")
                    or kwargs.get("input")
                    or ""
                )
                run_id = kwargs.get("run_id") or kwargs.get("parent_run_id") or f"{tool_name}-{time.time_ns()}"
                self._start_if_needed(run_id, tool_name, tool_input)

            def on_tool_end(self, output, **kwargs):
                run_id = kwargs.get("run_id") or kwargs.get("parent_run_id")
                if not run_id:
                    return
                call_id = str(run_id)
                self._starts.pop(call_id, None)
                record_tool_end(session_id, call_id, output)
                logger.debug(
                    "[tool:end] session=%s call_id=%s output=%s",
                    session_id,
                    call_id,
                    SmartAgent._short_debug_text(output),
                )

            def on_tool_error(self, error, **kwargs):
                run_id = kwargs.get("run_id") or kwargs.get("parent_run_id")
                if not run_id:
                    return
                call_id = str(run_id)
                self._starts.pop(call_id, None)
                record_tool_end(session_id, call_id, f"Error: {error}")
                logger.debug(
                    "[tool:error] session=%s call_id=%s error=%s",
                    session_id,
                    call_id,
                    SmartAgent._short_debug_text(error),
                )

        return ToolUsageCallback()

    @staticmethod
    def _extract_answer_and_tokens(result: dict) -> tuple[str, int]:
        messages = result.get("messages", []) if isinstance(result, dict) else []

        answer = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                reasoning = message.additional_kwargs.get("reasoning")
                if isinstance(reasoning, str):
                    answer += "<think>" + reasoning.strip() + "</think>\n"
                if isinstance(message.content, str):
                    answer += message.content
                elif isinstance(message.content, list):
                    text_parts = []
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    answer += "\n".join(text_parts).strip()
                if answer:
                    break

        total_tokens = 0
        for message in messages:
            usage_meta = getattr(message, "usage_metadata", None)
            if isinstance(usage_meta, dict):
                total_tokens += usage_meta.get("total_tokens", 0) or 0
                continue

            response_meta = getattr(message, "response_metadata", None)
            if isinstance(response_meta, dict):
                token_usage = response_meta.get("token_usage")
                if isinstance(token_usage, dict):
                    total_tokens += token_usage.get("total_tokens", 0) or 0

        return answer, total_tokens

    def chat(
        self,
        user_input: str,
        image_urls: Optional[list[str]] = None,
        rag_db_names: Optional[list[str]] = None,
        force_agent: bool = False,
        allowed_mcp_tools: Optional[list[str]] = None,
        conversation_path: Optional[list[str]] = None,
    ) -> str:
        self.memory.add_user_message(user_input)

        messages_override = None
        caps = get_capabilities()
        if image_urls and caps.multimodal_supported is True:
            content = [{"type": "text", "text": user_input}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": url})
            # Ensure content is of type list[str | dict[Any, Any]]
            messages_override = self.memory.get_messages()[:-1] + [
                HumanMessage(content=content)  # type: ignore
            ]

        original_db_name, original_db_names = self._apply_rag_scope(rag_db_names)
        session_token = set_current_session_id(self.session_id)
        scope_token = set_current_scope_key(self._build_tool_scope_key(conversation_path))
        try:
            if not force_agent:
                rag_answer = self._try_rag_shortcut(user_input, rag_db_names=rag_db_names)
                if rag_answer is not None:
                    self.memory.add_ai_message(rag_answer)
                    return rag_answer

            callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
            tool_callback = self._tool_callback()
            active_agent = self._create_agent(self._select_tools_for_request(allowed_mcp_tools))
            try:
                result = self._invoke_with_retry(
                    cast(list[Any], messages_override or self.memory.get_messages()),
                    [callback, tool_callback],
                    active_agent,
                )
                answer, total_tokens = self._extract_answer_and_tokens(cast(dict, result))
                logger.info("Token usage: %s", total_tokens)
                agent_token_usage.inc(total_tokens)
            except Exception as exc:
                logger.exception("Agent execution failed")
                if GraphRecursionError is not None and isinstance(exc, GraphRecursionError):
                    msg = (
                        "Agent recursion detected (GraphRecursionError). "
                        "Review tool implementations for recursive agent calls or increase 'recursion_limit' config."
                    )
                    if config.settings.ENV == "prod":
                        answer = "Request failed: agent recursion." 
                    else:
                        answer = f"Request failed: {msg}"
                else:
                    if config.settings.ENV == "prod":
                        answer = "Request failed."
                    else:
                        answer = f"Request failed: {str(exc)[:100]}"
        finally:
            self._restore_rag_scope(original_db_name, original_db_names)
            reset_current_scope_key(scope_token)
            reset_current_session_id(session_token)

        self.memory.add_ai_message(answer)
        return answer

    async def achat(
        self,
        user_input: str,
        image_urls: Optional[list[str]] = None,
        rag_db_names: Optional[list[str]] = None,
        force_agent: bool = False,
        allowed_mcp_tools: Optional[list[str]] = None,
        messages: Optional[list[dict]] = None,  # 新增参数
        conversation_path: Optional[list[str]] = None,
    ) -> str:
        # 如果提供了 messages，则直接使用它们构建输入
        if messages is not None:
            lc_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    # 如果 content 是字符串，直接使用；如果是列表（多模态），也直接使用
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                elif role == "system":
                    lc_messages.append(SystemMessage(content=content))
            # 忽略 user_input 和 image_urls，直接使用 lc_messages 作为输入
            messages_override = lc_messages
        else:
            # 原有逻辑：基于 self.memory 构建消息
            self.memory.add_user_message(user_input)
            messages_override = None
            caps = get_capabilities()
            if image_urls and caps.multimodal_supported is True:
                content = [{"type": "text", "text": user_input}]
                for url in image_urls:
                    content.append({"type": "image_url", "image_url": url})
                messages_override = self.memory.get_messages()[:-1] + [
                    HumanMessage(content=content) # type: ignore
                ]

        original_db_name, original_db_names = self._apply_rag_scope(rag_db_names)
        session_token = set_current_session_id(self.session_id)
        scope_token = set_current_scope_key(self._build_tool_scope_key(conversation_path))
        try:
            if not force_agent:
                rag_answer = await asyncio.to_thread(
                    self._try_rag_shortcut,
                    user_input,
                    rag_db_names,
                )
                if rag_answer is not None:
                    self.memory.add_ai_message(rag_answer)
                    return rag_answer

            callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
            tool_callback = self._tool_callback()
            active_agent = self._create_agent(self._select_tools_for_request(allowed_mcp_tools))
            try:
                token = set_agent_active()
                try:
                    # 使用 messages_override（如果存在）或 self.memory.get_messages()
                    input_messages = cast(list[Any], messages_override or self.memory.get_messages())
                    result = await self._ainvoke_with_retry(
                        input_messages,
                        [callback, tool_callback],
                        active_agent,
                    )
                finally:
                    reset_agent_active(token)
                answer, total_tokens = self._extract_answer_and_tokens(cast(dict, result))
                logger.info("Token usage: %s", total_tokens)
                agent_token_usage.inc(total_tokens)
            except Exception as exc:
                logger.exception("Agent execution failed")
                if GraphRecursionError is not None and isinstance(exc, GraphRecursionError):
                    msg = (
                        "Agent recursion detected (GraphRecursionError). "
                        "Review tool implementations for recursive agent calls or increase 'recursion_limit' config."
                    )
                    if config.settings.ENV == "prod":
                        answer = "Request failed: agent recursion."
                    else:
                        answer = f"Request failed: {msg}"
                else:
                    if config.settings.ENV == "prod":
                        answer = "Request failed."
                    else:
                        answer = f"Request failed: {str(exc)[:100]}"
        finally:
            self._restore_rag_scope(original_db_name, original_db_names)
            reset_current_scope_key(scope_token)
            reset_current_session_id(session_token)

        self.memory.add_ai_message(answer)
        return answer

    async def astream(
        self,
        user_input: str,
        image_urls: Optional[list[str]] = None,
        rag_db_names: Optional[list[str]] = None,
        force_agent: bool = False,
        allowed_mcp_tools: Optional[list[str]] = None,
        messages: Optional[list[dict]] = None,
        conversation_path: Optional[list[str]] = None,
    ):
        # 与 achat 类似的逻辑，构建 input_messages
        if messages is not None:
            lc_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                elif role == "system":
                    lc_messages.append(SystemMessage(content=content))
            messages_override = lc_messages
        else:
            self.memory.add_user_message(user_input)
            messages_override = None
            caps = get_capabilities()
            if image_urls and caps.multimodal_supported is True:
                content = [{"type": "text", "text": user_input}]
                for url in image_urls:
                    content.append({"type": "image_url", "image_url": url})
                messages_override = self.memory.get_messages()[:-1] + [
                    HumanMessage(content=content) # type: ignore
                ]

        original_db_name, original_db_names = self._apply_rag_scope(rag_db_names)
        session_token = set_current_session_id(self.session_id)
        scope_token = set_current_scope_key(self._build_tool_scope_key(conversation_path))
        try:
            if not force_agent:
                rag_answer = await asyncio.to_thread(
                    self._try_rag_shortcut,
                    user_input,
                    rag_db_names,
                )
                if rag_answer is not None:
                    self.memory.add_ai_message(rag_answer)
                    yield rag_answer
                    return

            queue: asyncio.Queue[str] = asyncio.Queue()
            done_event = asyncio.Event()
            token_buffer: list[str] = []
            error_text: Optional[str] = None
            final_answer: Optional[str] = None
            stream_started_at = time.perf_counter()
            agent_session_id = self.session_id

            @DeprecationWarning
            def _tool_markup(call: dict[str, Any]) -> str:
                call_id = html.escape(str(call.get("call_id") or ""), quote=False)
                tool_name = html.escape(str(call.get("tool_name") or "tool"), quote=False)
                tool_input = html.escape(str(call.get("tool_input") or ""), quote=False)
                tool_output = html.escape(str(call.get("tool_output") or ""), quote=False)
                parts = [
                    "<tool>",
                    f"<id>{call_id}</id>",
                    f"<type>{tool_name}</type>",
                    f"<query>{tool_input}</query>",
                ]
                if tool_output:
                    parts.append(f"<result>{tool_output}</result>")
                parts.append("</tool>")
                return "".join(parts)

            class StreamCallback(BaseCallbackHandler):
                def __init__(self, queue: asyncio.Queue[str]):
                    self.queue = queue
                    self.thinking = False
                    self._first_emit_at: float | None = None
                    self._tool_started_at: dict[str, float] = {}

                @staticmethod
                def _escape_xml(value: Any) -> str:
                    return html.escape(str(value or ""), quote=False)

                @staticmethod
                def _json_pretty(value: Any) -> str:
                    return json.dumps(value, ensure_ascii=False, indent=2)

                @staticmethod
                def _parse_json_or_literal(text: str) -> Any:
                    stripped = (text or "").strip()
                    if not stripped:
                        return None
                    try:
                        return json.loads(stripped)
                    except Exception:
                        pass
                    try:
                        return ast.literal_eval(stripped)
                    except Exception:
                        return None

                @staticmethod
                def _extract_content_literal(text: str) -> Optional[str]:
                    if not text:
                        return None
                    match = re.search(r"content=(['\"])([\s\S]*?)\1", text)
                    if not match:
                        return None
                    quoted_literal = f"{match.group(1)}{match.group(2)}{match.group(1)}"
                    try:
                        value = ast.literal_eval(quoted_literal)
                    except Exception:
                        value = match.group(2)
                    return str(value)

                def _format_tool_payload(self, payload: Any) -> str:
                    if payload is None:
                        return ""
                    if isinstance(payload, (dict, list, tuple)):
                        return self._json_pretty(payload)

                    content_attr = getattr(payload, "content", None)
                    if content_attr is not None and content_attr is not payload:
                        formatted = self._format_tool_payload(content_attr)
                        if formatted:
                            return formatted

                    text = str(payload).strip()
                    if not text:
                        return ""

                    parsed = self._parse_json_or_literal(text)
                    if isinstance(parsed, (dict, list, tuple)):
                        return self._json_pretty(parsed)
                    if isinstance(parsed, str) and parsed != text:
                        reparsed = self._parse_json_or_literal(parsed)
                        if isinstance(reparsed, (dict, list, tuple)):
                            return self._json_pretty(reparsed)
                        return parsed

                    embedded_content = self._extract_content_literal(text)
                    if embedded_content is not None:
                        embedded_parsed = self._parse_json_or_literal(embedded_content)
                        if isinstance(embedded_parsed, (dict, list, tuple)):
                            return self._json_pretty(embedded_parsed)
                        return embedded_content

                    return text

                def _write_to_front_end(self, message: str) -> None:
                    if self._first_emit_at is None:
                        self._first_emit_at = time.perf_counter()
                        logger.info(
                            "astream first_emit: session=%s first_emit_ms=%.1f",
                            agent_session_id,
                            (self._first_emit_at - stream_started_at) * 1000.0,
                        )
                    self.queue.put_nowait(message)

                def on_llm_new_token(self, token: str, **kwargs):
                    """流式输出 token"""
                    chunk = kwargs.get("chunk")
                    chunk_message = getattr(chunk, "message", None)
                    if not isinstance(chunk_message, AIMessageChunk):
                        return

                    reasoning = (getattr(chunk_message, "additional_kwargs", None) or {}).get("reasoning")
                    assert not ((reasoning is not None and len(reasoning) > 0) and (len(token) > 0)), \
                        f"Token({token})(length:{len(token) if isinstance(token, str) else 'N/A'}) and reasoning({reasoning})(length:{len(reasoning) if isinstance(reasoning, str) else 'N/A'}) cannot both be present"
                    if reasoning is not None and len(reasoning) > 0:
                        if not self.thinking:
                            self.thinking = True
                            reasoning = f"<think>{reasoning}"
                        try:
                            self._write_to_front_end(reasoning)
                        except Exception:
                            pass
                    if len(token) > 0:
                        if self.thinking:
                            self.thinking = False
                            token = f"</think>{token}"
                        try:
                            self._write_to_front_end(token)
                        except Exception:
                            pass
                
                def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, **kwargs: Any) -> Any:
                    if self.thinking:
                        self.thinking = False
                        try:
                            self._write_to_front_end("</think>")
                        except Exception:
                            pass
                    return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)

                def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
                    tool_name = serialized.get("name", "tool")
                    run_id = kwargs.get("run_id")
                    if run_id:
                        call_id = str(run_id)
                        self._tool_started_at[call_id] = time.perf_counter()
                        safe_input = self._escape_xml(self._format_tool_payload(input_str))
                        safe_tool_name = self._escape_xml(tool_name)
                        # 发送开始部分，不闭合标签
                        tool_xml = f"<tool><id>{call_id}</id><type>{safe_tool_name}</type><query>{safe_input}</query>"
                        if self.thinking:
                            tool_xml = f"</think>{tool_xml}"
                        try:
                            self._write_to_front_end(tool_xml)
                        except Exception:
                            pass

                def on_tool_end(self, output: Any, **kwargs):
                    run_id = kwargs.get("run_id")
                    if run_id:
                        call_id = str(run_id)
                        started = self._tool_started_at.pop(call_id, None)
                        safe_output = self._escape_xml(self._format_tool_payload(output))
                        if started is not None:
                            logger.info(
                                "astream tool_done: session=%s call_id=%s ms=%.1f output_chars=%s",
                                agent_session_id,
                                call_id,
                                (time.perf_counter() - started) * 1000.0,
                                len(safe_output),
                            )
                        # 发送结果和闭合标签
                        tool_xml = f"<result>{safe_output}</result></tool>"
                        if self.thinking:
                            tool_xml += "<think>"
                        try:
                            self._write_to_front_end(tool_xml)
                        except Exception:
                            pass

                def on_tool_error(self, error: BaseException, **kwargs):
                    run_id = kwargs.get("run_id")
                    if run_id:
                        safe_error = self._escape_xml(error)
                        tool_xml = f"<result>Error: {safe_error}</result></tool>"
                        try:
                            self._write_to_front_end(tool_xml)
                        except Exception:
                            pass
                                
            stream_callback = StreamCallback(queue)
            callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
            tool_callback = self._tool_callback()
            active_agent = self._create_agent(self._select_tools_for_request(allowed_mcp_tools))

            async def run_agent():
                nonlocal error_text, final_answer
                token = set_agent_active()
                try:
                    input_messages = cast(list[Any], messages_override or self.memory.get_messages())
                    result = await self._ainvoke_with_retry(
                        input_messages,
                        [callback, stream_callback, tool_callback],
                        active_agent,
                    )
                    answer, total_tokens = self._extract_answer_and_tokens(cast(dict, result))
                    final_answer = answer
                    if answer:
                        self.memory.add_ai_message(answer)
                    logger.info("Token usage: %s", total_tokens)
                    agent_token_usage.inc(total_tokens)
                except Exception as exc:
                    logger.exception("Agent execution failed")
                    if GraphRecursionError is not None and isinstance(exc, GraphRecursionError):
                        msg = (
                            "Agent recursion detected (GraphRecursionError). "
                            "Review tool implementations for recursive agent calls or increase 'recursion_limit' config."
                        )
                        if config.settings.ENV == "prod":
                            error_text = "Request failed: agent recursion."
                        else:
                            error_text = f"Request failed: {msg}"
                    else:
                        if config.settings.ENV == "prod":
                            error_text = "Request failed."
                        else:
                            error_text = f"Request failed: {str(exc)[:100]}"
                finally:
                    reset_agent_active(token)
                    done_event.set()

            task = asyncio.create_task(run_agent())

            while True:
                if done_event.is_set() and queue.empty():
                    break
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                token_buffer.append(token)
                yield token

            await task

            try:
                logger.debug(
                    "astream finished, token_buffer length=%s final_answer=%r error_text=%r",
                    len(token_buffer),
                    final_answer,
                    error_text,
                )
                logger.info(
                    "astream done: session=%s total_ms=%.1f emitted_chunks=%s",
                    self.session_id,
                    (time.perf_counter() - stream_started_at) * 1000.0,
                    len(token_buffer),
                )
            except Exception:
                pass

            if error_text:
                self.memory.add_ai_message(error_text)
                if not token_buffer:
                    yield error_text
            elif not final_answer and token_buffer:
                fallback_answer = "".join(token_buffer)
                self.memory.add_ai_message(fallback_answer)
        finally:
            self._restore_rag_scope(original_db_name, original_db_names)
            reset_current_scope_key(scope_token)
            reset_current_session_id(session_token)

    def _try_rag_shortcut(
        self,
        user_input: str,
        rag_db_names: Optional[list[str]] = None,
    ) -> Optional[str]:
        if self.tools:
            return None
        if not config.settings.ENABLE_RAG:
            return None
        try:
            rag_result = RAGEngine().query(user_input, db_names=rag_db_names)
        except Exception:
            return None
        sources = rag_result.get("sources", []) if rag_result else []
        if len(sources) < config.settings.MIN_RAG_SOURCES:
            return None
        top_score = sources[0].get("score")
        if top_score is None:
            return None
        if top_score > config.settings.RAG_CONFIDENCE_THRESHOLD:
            return rag_result.get("answer", "")
        return None
