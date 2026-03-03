from __future__ import annotations

from contextvars import ContextVar, Token

_CURRENT_LANGUAGE: ContextVar[str] = ContextVar("current_language", default="zh")


def normalize_language(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    return "zh"


def set_current_language(language: str | None) -> Token:
    return _CURRENT_LANGUAGE.set(normalize_language(language))


def reset_current_language(token: Token) -> None:
    _CURRENT_LANGUAGE.reset(token)


def get_current_language() -> str:
    return _CURRENT_LANGUAGE.get()
