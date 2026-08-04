"""Tests for the meta-phase in ``build_kg_recursive`` — kg-meta-architecture plan T4.

Covers the optional meta-architecture phase inserted before Phase 0 root
identification in ``agent/kg_build_agent.py``:

- Mock scenario: GranularityAgent + sysml_set_meta_schema called, state saved
  with meta_schema version=1, pipeline continues.
- Resume scenario: build dict already carries meta_schema version=1 →
  meta-phase skipped (GranularityAgent never instantiated).
- Version mismatch: meta_schema version=2 → regenerated (version check).
- Failure degradation: GranularityAgent raises → warning logged, pipeline
  continues without meta (no exception propagates).
- No granularity: granularity_description=None → meta-phase fully skipped
  (identical to pre-T4 behaviour).

Import hygiene: importing ``agent.kg_build_agent`` executes ``agent/__init__.py``
→ module-level ``RAGEngine()`` instantiation (expensive). Same shared-stub-or-
local pattern as tests/test_kg_granularity.py is reused via ``no_expensive_rag``.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
    first `import agent.*` (required by test_import_contract's contract)."""
    import rag.engine as engine_module

    stub = _shared_stub_or_local()
    monkeypatch.setattr(engine_module, "RAGEngine", stub)
    return stub


# ── Test doubles ──────────────────────────────────────────────

_FAKE_TREE = SimpleNamespace(total_pages=3, nodes={}, mark_processed=Mock())

_SECTIONS = {"n2": []}

_DOC_NAME = "测试文档"


def _rag_doc():
    return SimpleNamespace(doc_name=_DOC_NAME)


def _make_agent(build_state: dict):
    """KGBuildAgent without __init__ side effects; all pipeline steps mocked."""
    from agent.kg_build_agent import KGBuildAgent

    agent = object.__new__(KGBuildAgent)
    agent._mcp_session = AsyncMock()
    agent._load_build_state = Mock(return_value=build_state)
    agent.initialize = AsyncMock()
    agent._summary = AsyncMock(
        return_value={"total_entities": 0, "total_relations": 0})
    agent._build_section_map = Mock(return_value=_SECTIONS)
    agent._build_inverted_index = Mock(return_value={})
    agent._identify_root = AsyncMock(
        return_value=({"name": "root"}, ["n2"]))
    agent._process_root_sections = AsyncMock(return_value=[])
    agent._unload_light_model = AsyncMock()
    agent._propagate_entities = AsyncMock(return_value=[])
    agent._process_queued_section = AsyncMock(return_value=[])
    agent._trigger_save = AsyncMock()
    agent._save_recursive_state = Mock()
    agent._deduplicate_entities = AsyncMock()
    agent._aggregate_graph = AsyncMock()
    return agent


def _meta_architecture():
    """Prebuilt MetaArchitecture returned by the mocked GranularityAgent."""
    from agent.kg_granularity import MetaArchitecture

    return MetaArchitecture(
        entity_types=[
            {"name": "PartDef", "parent": None, "level": 1,
             "description": "部件/模块"},
            {"name": "AttributeDef", "parent": "PartDef", "level": 2,
             "description": "属性/参数"},
        ],
        root_nodes=[
            {"name": "系统概览", "type": "PartDef", "description": "顶层系统组成"},
        ],
        relation_patterns=[
            {"source_type": "PartDef", "target_type": "PartDef",
             "relation_type": "allocation", "desc": "组成关系"},
        ],
        constraints=["忽略温度参数"],
        granularity_description="详细到端口和命令级别",
    )


def _meta_schema_calls(mcp_session):
    return [
        c for c in mcp_session.call_tool.await_args_list
        if c.args and c.args[0] == "sysml_set_meta_schema"
    ]


def _meta_state_calls(save_mock):
    return [
        c for c in save_mock.call_args_list
        if c.kwargs.get("meta_schema") is not None
    ]


def _phase_of(call) -> str:
    """_save_recursive_state 的 phase 是位置参数（doc_name, phase, ...）。"""
    if len(call.args) > 1:
        return call.args[1]
    return call.kwargs.get("phase")


# ── Meta-phase tests ──────────────────────────────────────────


def test_meta_phase_runs_and_saves(no_expensive_rag):
    """granularity_description given + no stored meta → GranularityAgent runs,
    sysml_set_meta_schema called, state saved with meta_schema version=1,
    pipeline continues and returns stats."""
    agent = _make_agent(build_state={})
    meta = _meta_architecture()

    with patch("agent.kg_build_agent.DocumentTreeState",
               return_value=_FAKE_TREE):
        with patch("agent.kg_granularity.GranularityAgent") as ga_cls:
            ga_cls.return_value.determine_meta_architecture = AsyncMock(
                return_value=meta)
            stats = asyncio.run(agent.build_kg_recursive(
                _rag_doc(), granularity_description="详细到端口和命令级别"))

    ga_cls.assert_called_once_with(agent, _DOC_NAME)
    ga_cls.return_value.determine_meta_architecture.assert_awaited_once()

    schema_calls = _meta_schema_calls(agent._mcp_session)
    assert len(schema_calls) == 1
    schema_payload = schema_calls[0].args[1]
    assert schema_payload["schema_json"]["version"] == 1
    assert len(schema_payload["schema_json"]["entity_types"]) == 2

    state_calls = _meta_state_calls(agent._save_recursive_state)
    assert len(state_calls) == 1
    saved = state_calls[0].kwargs["meta_schema"]
    assert saved["version"] == 1
    assert state_calls[0].args[1] == "recursive_phase0"

    # 管线继续：最终 done 状态保存 + stats 返回
    assert isinstance(stats, dict)
    assert any(
        _phase_of(c) == "done"
        for c in agent._save_recursive_state.call_args_list
    )


def test_meta_phase_resume_skips(no_expensive_rag):
    """build dict already has meta_schema version=1 → meta-phase skipped
    (GranularityAgent never instantiated, no meta save), and the stored
    meta_schema is restored into the MCP session (F2-F3: 跨进程 resume 时
    新进程 MCP manager 的 _meta_schema 为 None，软过滤/动态 prompt 会静默
    回退——已存 schema 必须写回)."""
    agent = _make_agent(build_state={
        "meta_schema": {"version": 1, "entity_types": [{"name": "PartDef"}]},
    })

    with patch("agent.kg_build_agent.DocumentTreeState",
               return_value=_FAKE_TREE):
        with patch("agent.kg_granularity.GranularityAgent") as ga_cls:
            stats = asyncio.run(agent.build_kg_recursive(
                _rag_doc(), granularity_description="详细到端口和命令级别"))

    ga_cls.assert_not_called()  # 不重新生成
    # F2-F3: 已存 schema 原样写回 MCP（含 version 与 entity_types）
    schema_calls = _meta_schema_calls(agent._mcp_session)
    assert len(schema_calls) == 1
    assert schema_calls[0].args[1]["schema_json"] == {
        "version": 1, "entity_types": [{"name": "PartDef"}]}
    assert _meta_state_calls(agent._save_recursive_state) == []
    assert isinstance(stats, dict)


def test_meta_phase_version_mismatch_regenerates(no_expensive_rag):
    """stored meta_schema version != 1 → regenerated via GranularityAgent
    (version check invalidates stale schema)."""
    agent = _make_agent(build_state={
        "meta_schema": {"version": 2, "entity_types": []},
    })
    meta = _meta_architecture()

    with patch("agent.kg_build_agent.DocumentTreeState",
               return_value=_FAKE_TREE):
        with patch("agent.kg_granularity.GranularityAgent") as ga_cls:
            ga_cls.return_value.determine_meta_architecture = AsyncMock(
                return_value=meta)
            asyncio.run(agent.build_kg_recursive(
                _rag_doc(), granularity_description="详细到端口和命令级别"))

    ga_cls.assert_called_once()
    state_calls = _meta_state_calls(agent._save_recursive_state)
    assert len(state_calls) == 1
    assert state_calls[0].kwargs["meta_schema"]["version"] == 1


def test_meta_phase_failure_degrades(no_expensive_rag):
    """GranularityAgent raises → warning path, pipeline continues without
    meta (no exception propagates, no schema write)."""
    agent = _make_agent(build_state={})

    with patch("agent.kg_build_agent.DocumentTreeState",
               return_value=_FAKE_TREE):
        with patch("agent.kg_granularity.GranularityAgent") as ga_cls:
            ga_cls.return_value.determine_meta_architecture = AsyncMock(
                side_effect=RuntimeError("LLM unavailable"))
            stats = asyncio.run(agent.build_kg_recursive(
                _rag_doc(), granularity_description="详细到端口和命令级别"))

    assert isinstance(stats, dict)
    assert _meta_schema_calls(agent._mcp_session) == []
    assert _meta_state_calls(agent._save_recursive_state) == []
    # 管线继续完成（done 状态保存）
    assert any(
        _phase_of(c) == "done"
        for c in agent._save_recursive_state.call_args_list
    )


def test_meta_phase_skipped_without_description(no_expensive_rag):
    """granularity_description=None → meta-phase fully skipped, identical to
    pre-T4 behaviour (GranularityAgent never instantiated)."""
    agent = _make_agent(build_state={})

    with patch("agent.kg_build_agent.DocumentTreeState",
               return_value=_FAKE_TREE):
        with patch("agent.kg_granularity.GranularityAgent") as ga_cls:
            stats = asyncio.run(agent.build_kg_recursive(_rag_doc()))

    ga_cls.assert_not_called()
    assert _meta_schema_calls(agent._mcp_session) == []
    assert _meta_state_calls(agent._save_recursive_state) == []
    assert isinstance(stats, dict)
