# -*- coding: utf-8 -*-
"""Tests for ``_build_extraction_prompt`` — kg-meta-architecture plan T5.

Covers the dynamic extraction prompt generator on ``KGBuildAgent``:

- No meta schema set → returns the static ``EXTRACTION_CANDIDATES_PROMPT``
  character-for-character (fallback contract).
- Meta schema present → returns a dynamically built prompt containing the
  granularity description, allowed entity types, relation patterns,
  constraints and root nodes; the output-format section of the static prompt
  is preserved.
- MCP call raises → falls back to the static prompt (no exception propagates).

Import hygiene: importing ``agent.kg_build_agent`` executes ``agent/__init__.py``
→ module-level ``RAGEngine()`` instantiation (expensive). The shared
``no_expensive_rag`` stub pattern from tests/test_meta_phase.py is reused.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock

import pytest


def _shared_stub_or_local():
    """test_import_contract.StubRAGEngine when loaded, else a local twin."""
    for name in ("test_import_contract", "tests.test_import_contract"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "StubRAGEngine", None) is not None:
            return mod.StubRAGEngine

    class StubRAGEngine:
        _instance = None
        _initialized = True

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self, *args, **kwargs):
            pass

    return StubRAGEngine


@pytest.fixture
def no_expensive_rag(monkeypatch):
    """Patch rag.engine.RAGEngine to a cheap singleton stub before the
    first `import agent.*`."""
    import rag.engine as engine_module

    stub = _shared_stub_or_local()
    monkeypatch.setattr(engine_module, "RAGEngine", stub)
    return stub


def _make_agent(mcp_return):
    """KGBuildAgent without __init__ side effects; only _mcp_session stubbed."""
    from agent.kg_build_agent import KGBuildAgent

    agent = object.__new__(KGBuildAgent)
    agent._mcp_session = AsyncMock()
    agent._mcp_session.call_tool.return_value = mcp_return
    return agent


def _meta_schema() -> dict:
    """Realistic MetaArchitecture.to_dict() shape (entity_types are dicts)."""
    return {
        "entity_types": [
            {"name": "PartDef", "parent": None, "level": 1,
             "description": "\u90e8\u4ef6/\u6a21\u5757"},
            {"name": "AttributeDef", "parent": "PartDef", "level": 2,
             "description": "\u5c5e\u6027/\u53c2\u6570"},
        ],
        "relation_patterns": [
            {"source_type": "PartDef", "target_type": "PartDef",
             "relation_type": "allocation", "desc": "\u7ec4\u6210\u5173\u7cfb"},
        ],
        "constraints": ["\u5ffd\u7565\u6e29\u5ea6\u53c2\u6570", "\u4e0d\u63d0\u53d6\u547d\u4ee4"],
        "root_nodes": [
            {"name": "\u7cfb\u7edf\u6982\u89c8", "type": "PartDef",
             "description": "\u9876\u5c42\u7cfb\u7edf\u7ec4\u6210"},
        ],
        "granularity_description": "\u8be6\u7ec6\u5230\u7aef\u53e3\u548c\u547d\u4ee4\u7ea7\u522b",
    }


def test_no_meta_returns_static_prompt_verbatim(no_expensive_rag):
    """sysml_get_meta_schema → {"ok": False} → prompt must equal the static
    constant character-for-character."""
    from agent.kg_build_agent import EXTRACTION_CANDIDATES_PROMPT

    agent = _make_agent(
        json.dumps({"ok": False, "error": "no meta schema set"}))
    prompt = asyncio.run(agent._build_extraction_prompt())

    assert prompt == EXTRACTION_CANDIDATES_PROMPT
    assert prompt is not None and len(prompt) > 100
    agent._mcp_session.call_tool.assert_awaited_once_with(
        "sysml_get_meta_schema", {})


def test_meta_schema_builds_dynamic_prompt(no_expensive_rag):
    """schema present → prompt contains granularity description, allowed
    entity types, relation patterns, constraints, root nodes; output-format
    section preserved; differs from the static constant."""
    from agent.kg_build_agent import EXTRACTION_CANDIDATES_PROMPT

    agent = _make_agent(json.dumps({"ok": True, "schema": _meta_schema()}))
    prompt = asyncio.run(agent._build_extraction_prompt())

    # \u7c92\u5ea6\u63cf\u8ff0
    assert "\u8be6\u7ec6\u5230\u7aef\u53e3\u548c\u547d\u4ee4\u7ea7\u522b" in prompt
    # \u5141\u8bb8\u7684\u5b9e\u4f53\u7c7b\u578b\u533a\u57df + \u7c7b\u578b\u540d
    assert "## \u5141\u8bb8\u7684\u5b9e\u4f53\u7c7b\u578b" in prompt
    assert "- PartDef" in prompt
    assert "- AttributeDef" in prompt
    assert "\u7236\u7c7b\u578b: PartDef" in prompt
    # \u5173\u7cfb\u6a21\u5f0f
    assert "## \u5141\u8bb8\u7684\u5173\u7cfb\u6a21\u5f0f" in prompt
    assert "PartDef \u2192 allocation \u2192 PartDef" in prompt
    assert "\u7ec4\u6210\u5173\u7cfb" in prompt
    # \u7ea6\u675f
    assert "\u5ffd\u7565\u6e29\u5ea6\u53c2\u6570" in prompt
    assert "\u4e0d\u63d0\u53d6\u547d\u4ee4" in prompt
    # \u6839\u8282\u70b9
    assert "## \u6839\u8282\u70b9" in prompt
    assert "\u7cfb\u7edf\u6982\u89c8 (PartDef)" in prompt
    assert "\u9876\u5c42\u7cfb\u7edf\u7ec4\u6210" in prompt
    # \u8f93\u51fa\u683c\u5f0f\u6bb5\u4fdd\u7559
    assert "## \u8f93\u51fa\u683c\u5f0f" in prompt
    assert '\u5b9e\u4f53\u683c\u5f0f: {"type":"PartDef"' in prompt
    assert '{"relation":true,"type":"Connection"' in prompt
    assert "\u8f93\u5165\u6587\u672c\u65e0\u4efb\u4f55\u7cfb\u7edf\u67b6\u6784\u5185\u5bb9\u65f6\u8f93\u51fa: []" in prompt
    # \u9759\u6001\u5e38\u91cf\u4e2d\u7684\u65e7\u5173\u7cfb\u5217\u8868\u4e0d\u5e94\u51fa\u73b0\u5728\u52a8\u6001 prompt \u91cc
    assert prompt != EXTRACTION_CANDIDATES_PROMPT
    agent._mcp_session.call_tool.assert_awaited_once_with(
        "sysml_get_meta_schema", {})


def test_call_tool_exception_falls_back(no_expensive_rag):
    """MCP call raises → falls back to the static prompt, no exception
    propagates to the caller."""
    from agent.kg_build_agent import EXTRACTION_CANDIDATES_PROMPT

    agent = _make_agent(None)
    agent._mcp_session.call_tool.side_effect = RuntimeError("boom")
    prompt = asyncio.run(agent._build_extraction_prompt())

    assert prompt == EXTRACTION_CANDIDATES_PROMPT
