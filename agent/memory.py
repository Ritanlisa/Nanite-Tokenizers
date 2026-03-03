from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict
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
                    config.settings.REDIS_URL, decode_responses=True
                )
                self._load()
            except Exception as exc:
                logger.warning("Redis unavailable, using in-memory history: %s", exc)

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

    def add_user_message(self, content: str) -> None:
        self.history.add_user_message(content)
        self._trim_history()
        self._save()

    def add_ai_message(self, content: str) -> None:
        self.history.add_ai_message(content)
        self._trim_history()
        self._save()

    def get_messages(self):
        return self.history.messages

    def clear(self) -> None:
        self.history.clear()
        if self.redis_client:
            self.redis_client.delete(f"chat_history:{self.session_id}")


_memory: Optional[PersistentChatMemory] = None


def get_chat_memory(session_id: str) -> PersistentChatMemory:
    global _memory
    if _memory is None or _memory.session_id != session_id:
        _memory = PersistentChatMemory(session_id)
    return _memory
