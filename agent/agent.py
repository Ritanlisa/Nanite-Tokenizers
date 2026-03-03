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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
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
from tool_usage import (
    record_tool_end,
    record_tool_start,
    set_current_session_id,
    reset_current_session_id,
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
            timeout=60,
            streaming=True,
        )
        caps = get_capabilities()
        if caps.tool_calling_supported is False:
            self.tools = []
        else:
            self.tools = tools
        self.agent = self._create_agent()

    def _create_agent(self):
        custom_prompt = (config.settings.SYSTEM_PROMPT or "").strip()
        if custom_prompt:
            system_prompt = custom_prompt
        elif self.tools:
            system_prompt = (
                "You are a helpful assistant with access to retrieval and execution tools.\n"
                "Strategy:\n"
                "1) Use rag_search first for internal knowledge.\n"
                "2) If rag_search is not useful, use fetch_webpage, skill_web_visit, or skill_search.\n"
                "3) Use skill_file_io for workspace-local file read/write tasks.\n"
                "4) Use skill_shell for safe command execution when needed.\n"
                "5) Ground answers on retrieved content and tool output.\n"
                "6) If tools fail, retry or explain limitations."
            )
        else:
            system_prompt = (
                "You are a helpful assistant.\n"
                "Tools are unavailable for this model; answer directly."
            )
        return create_agent(
            model=self.llm,
            tools=self.tools,
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
        try:
            if not force_agent:
                rag_answer = self._try_rag_shortcut(user_input, rag_db_names=rag_db_names)
                if rag_answer is not None:
                    self.memory.add_ai_message(rag_answer)
                    return rag_answer

            callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
            tool_callback = self._tool_callback()
            try:
                # run agent.invoke in thread with timeout to protect against hangs
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        self.agent.invoke,
                        cast(Any, {"messages": messages_override or self.memory.get_messages()}),
                        cast(Any, {"callbacks": [callback, tool_callback], "recursion_limit": config.settings.RECURSION_LIMIT}),
                    )
                    try:
                        result = fut.result(timeout=getattr(config.settings, "AGENT_INVOKE_TIMEOUT", 60))
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        raise
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
            reset_current_session_id(session_token)

        self.memory.add_ai_message(answer)
        return answer

    async def achat(
        self,
        user_input: str,
        image_urls: Optional[list[str]] = None,
        rag_db_names: Optional[list[str]] = None,
        force_agent: bool = False,
        messages: Optional[list[dict]] = None,  # 新增参数
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
            try:
                token = set_agent_active()
                try:
                    # 使用 messages_override（如果存在）或 self.memory.get_messages()
                    input_messages = messages_override or self.memory.get_messages()
                    result = await self.agent.ainvoke(
                        cast(Any, {"messages": input_messages}),
                        cast(
                            Any,
                            {
                                "callbacks": [callback, tool_callback],
                                "recursion_limit": config.settings.RECURSION_LIMIT,
                            },
                        ),
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
            reset_current_session_id(session_token)

        self.memory.add_ai_message(answer)
        return answer

    async def astream(
        self,
        user_input: str,
        image_urls: Optional[list[str]] = None,
        rag_db_names: Optional[list[str]] = None,
        force_agent: bool = False,
        messages: Optional[list[dict]] = None,
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
                    print(message, flush=True, end="")  # for debugging
                    self.queue.put_nowait(message)
                    
                def on_llm_new_token(self, token: str, **kwargs):
                    """流式输出 token"""
                    reasoning = kwargs['chunk'].message.additional_kwargs.get("reasoning")
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
                        safe_output = self._escape_xml(self._format_tool_payload(output))
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

            async def run_agent():
                nonlocal error_text, final_answer
                token = set_agent_active()
                try:
                    input_messages = messages_override or self.memory.get_messages()
                    result = await self.agent.ainvoke(
                        cast(Any, {"messages": input_messages}),
                        cast(Any, {
                            "callbacks": [callback, stream_callback, tool_callback],
                            "recursion_limit": config.settings.RECURSION_LIMIT,
                        }),
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
