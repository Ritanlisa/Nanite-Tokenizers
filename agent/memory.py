from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict
import redis

import config

logger = logging.getLogger(__name__)


class PersistentChatMemory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history = InMemoryChatMessageHistory()
        self.redis_client: Optional[redis.Redis] = None
        if config.settings.CACHE_TYPE == "redis" and config.settings.REDIS_URL:
            try:
                self.redis_client = redis.Redis.from_url(
                    config.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=float(getattr(config.settings, "REDIS_CONNECT_TIMEOUT", 1.0)),
                    socket_timeout=float(getattr(config.settings, "REDIS_SOCKET_TIMEOUT", 2.0)),
                    retry_on_timeout=False,
                    health_check_interval=30,
                )
                # 显式 ping，一旦不可用就快速回退，避免首次读写触发长时间阻塞
                self.redis_client.ping()
                self._load()
            except Exception as exc:
                logger.warning("Redis unavailable, using in-memory history: %s", exc)
                self.redis_client = None

    def _load(self) -> None:
        if not self.redis_client:
            return
        key = f"chat_history:{self.session_id}"
        data = self.redis_client.get(key)
        if isinstance(data, (str, bytes, bytearray)):
            try:
                msgs = messages_from_dict(json.loads(data))
                for message in msgs:
                    self.history.add_message(message)
            except Exception as exc:
                logger.error("Failed to load history: %s", exc)

    def _save(self) -> None:
        if not self.redis_client:
            return
        key = f"chat_history:{self.session_id}"
        data = json.dumps(messages_to_dict(self.history.messages))
        self.redis_client.setex(key, 86400, data)

    def _trim_history(self) -> None:
        max_messages = config.settings.MAX_HISTORY_ROUNDS * 2
        if len(self.history.messages) > max_messages:
            self.history.messages = self.history.messages[-max_messages:]

    def add_user_message(self, content) -> None:
        if isinstance(content, list):
            self.history.add_message(HumanMessage(content=content))
        else:
            self.history.add_user_message(str(content))
        self._trim_history()
        self._save()

    def add_ai_message(self, content: str) -> None:
        self.history.add_ai_message(content)
        self._trim_history()
        self._save()

    def get_messages(self):
        return self.history.messages

    def snapshot(self) -> list[dict[str, Any]]:
        return messages_to_dict(self.history.messages)

    def restore(self, messages: list[dict[str, Any]]) -> None:
        self.history.clear()
        for message in messages_from_dict(messages or []):
            self.history.add_message(message)
        self._trim_history()
        self._save()

    def clear(self) -> None:
        self.history.clear()
        if self.redis_client:
            self.redis_client.delete(f"chat_history:{self.session_id}")


_memories: dict[str, PersistentChatMemory] = {}


def get_chat_memory(session_id: str) -> PersistentChatMemory:
    memory = _memories.get(session_id)
    if memory is None:
        memory = PersistentChatMemory(session_id)
        _memories[session_id] = memory
    return memory
