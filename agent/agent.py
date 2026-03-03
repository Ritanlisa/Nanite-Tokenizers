from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, cast

from langchain.agents import create_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.memory import get_chat_memory
from agent.tools import tools
import config
from exceptions import TokenLimitExceeded
from rag.engine import RAGEngine
from monitoring import agent_token_usage

logger = logging.getLogger(__name__)


class TokenLimitCallback(BaseCallbackHandler):
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if response.llm_output and "token_usage" in response.llm_output:
            self.total_tokens += response.llm_output["token_usage"].get("total_tokens", 0)
            if self.total_tokens > self.max_tokens:
                raise TokenLimitExceeded(f"Token usage exceeded {self.max_tokens}")


class SmartAgent:
    def __init__(
        self,
        session_id: str = "default",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.memory = get_chat_memory(session_id)
        resolved_api_key = api_key or config.settings.OPENAI_API_KEY
        self.llm = ChatOpenAI(
            model=model or config.settings.LLM_MODEL,
            temperature=(
                temperature if temperature is not None else config.settings.TEMPERATURE
            ),
            api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
            base_url=api_url or config.settings.OPENAI_API_URL,
            timeout=60,
        )
        self.tools = tools
        self.agent = self._create_agent()

    def _create_agent(self):
        system_prompt = (
            "You are a helpful assistant with access to rag_search and fetch_webpage.\n"
            "Strategy:\n"
            "1) Use rag_search first for internal knowledge.\n"
            "2) If rag_search is not useful, use fetch_webpage.\n"
            "3) Ground answers on retrieved content.\n"
            "4) If tools fail, retry or explain limitations."
        )
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=config.settings.AGENT_VERBOSE,
            name="smart_agent",
        )

    @staticmethod
    def _extract_answer_and_tokens(result: dict) -> tuple[str, int]:
        messages = result.get("messages", []) if isinstance(result, dict) else []

        answer = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                if isinstance(message.content, str):
                    answer = message.content
                elif isinstance(message.content, list):
                    text_parts = []
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    answer = "\n".join(text_parts).strip()
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

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)

        rag_answer = self._try_rag_shortcut(user_input)
        if rag_answer is not None:
            self.memory.add_ai_message(rag_answer)
            return rag_answer

        callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
        try:
            result = self.agent.invoke(
                cast(Any, {"messages": self.memory.get_messages()}),
                cast(
                    Any,
                    {
                        "callbacks": [callback],
                        "recursion_limit": max(10, config.settings.MAX_ITERATIONS * 2),
                    },
                ),
            )
            answer, total_tokens = self._extract_answer_and_tokens(cast(dict, result))
            logger.info("Token usage: %s", total_tokens)
            agent_token_usage.inc(total_tokens)
        except Exception as exc:
            logger.exception("Agent execution failed")
            if config.settings.ENV == "prod":
                answer = "Request failed."
            else:
                answer = f"Request failed: {str(exc)[:100]}"

        self.memory.add_ai_message(answer)
        return answer

    async def achat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)

        rag_answer = await asyncio.to_thread(self._try_rag_shortcut, user_input)
        if rag_answer is not None:
            self.memory.add_ai_message(rag_answer)
            return rag_answer

        callback = TokenLimitCallback(config.settings.MAX_TOTAL_TOKENS)
        try:
            result = await self.agent.ainvoke(
                cast(Any, {"messages": self.memory.get_messages()}),
                cast(
                    Any,
                    {
                        "callbacks": [callback],
                        "recursion_limit": max(10, config.settings.MAX_ITERATIONS * 2),
                    },
                ),
            )
            answer, total_tokens = self._extract_answer_and_tokens(cast(dict, result))
            logger.info("Token usage: %s", total_tokens)
            agent_token_usage.inc(total_tokens)
        except Exception as exc:
            logger.exception("Agent execution failed")
            if config.settings.ENV == "prod":
                answer = "Request failed."
            else:
                answer = f"Request failed: {str(exc)[:100]}"

        self.memory.add_ai_message(answer)
        return answer

    def _try_rag_shortcut(self, user_input: str) -> Optional[str]:
        try:
            rag_result = RAGEngine().query(user_input)
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
