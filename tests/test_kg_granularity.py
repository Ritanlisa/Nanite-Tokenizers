"""Tests for agent/kg_granularity.py — kg-meta-architecture plan T1.

Covers:
- ``MetaArchitecture`` JSON roundtrip (to_dict → json.dumps → loads → from_dict),
  no LLM involved.
- ``GranularityAgent.determine_meta_architecture`` three-step flow with mocked
  light_llm / llm (no real Ollama calls): TOC analysis → section sampling →
  meta architecture generation.

Import hygiene: importing ``agent.kg_granularity`` executes ``agent/__init__.py``
→ ``agent.agent`` → ``agent.tools``, whose module level
``rag_engine = RAGEngine()`` (agent/tools/rag_tools.py:32) would instantiate the
REAL (expensive) engine. tests/test_import_contract.py asserts the singleton
identity ``agent.tools.rag_engine is RAGEngine()`` under ITS stub, so
rag.engine.RAGEngine must be patched BEFORE the first ``import agent.*`` in the
process — the same contract test_extract_page_candidates.py documents. We reuse
its shared-stub-or-local pattern: full-suite runs share
test_import_contract.StubRAGEngine; standalone runs fall back to a local twin
(the identity assertion cannot run without test_import_contract collected).
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


class _FakeLLMResponse:
    """Minimal stand-in for a langchain AI message: exposes .content."""

    def __init__(self, content: str):
        self.content = content


def _fake_node(text: str, title: str = "", page: int = 0):
    return SimpleNamespace(text=text, title=title, page_start=page)


class _FakeTreeState:
    """Minimal tree_state stand-in: get_tree_structure() + nodes lookup."""

    def __init__(self, structure: dict, nodes: dict):
        self._structure = structure
        self.nodes = nodes

    def get_tree_structure(self):
        return self._structure


# ── Test data ──────────────────────────────────────────────────

_TOC = {
    "document": "测试文档",
    "total_pages": 20,
    "processed_pages": 0,
    "remaining_pages": 20,
    "structure": [
        {
            "node_id": "n1", "title": "系统概览", "level": 1, "page": "1-3",
            "children": [
                {"node_id": "n2", "title": "总体架构", "level": 2, "page": "1-2"},
                {"node_id": "n3", "title": "电源设计", "level": 2, "page": "2-3"},
            ],
        },
        {"node_id": "n5", "title": "接口定义", "level": 1, "page": "10-12"},
        {"node_id": "n8", "title": "命令表", "level": 1, "page": "14-15"},
    ],
}

_SECTIONS = {
    "n2": [_fake_node("总体架构：系统由电源模块与主控模块组成。", "总体架构", 1)],
    "n3": [_fake_node("电源设计：输入 220V 交流，输出 12V 直流。", "电源设计", 2)],
    "n5": [_fake_node("接口定义：串口与以太网接口，波特率 115200。", "接口定义", 10)],
    "n8": [_fake_node("命令表：GET_STATUS、SET_POWER。", "命令表", 14)],
}

_SAMPLE_RESPONSE = json.dumps({"sample_node_ids": ["n2", "n5", "n8"]}, ensure_ascii=False)

_META_RESPONSE = json.dumps({
    "entity_types": [
        {"name": "PartDef", "parent": None, "level": 1, "description": "部件/模块"},
        {"name": "AttributeDef", "parent": "PartDef", "level": 2, "description": "属性/参数"},
    ],
    "root_nodes": [
        {"name": "系统概览", "type": "PartDef", "description": "顶层系统组成"},
    ],
    "relation_patterns": [
        {"source_type": "PartDef", "target_type": "PartDef",
         "relation_type": "allocation", "desc": "组成关系"},
    ],
    "constraints": ["忽略温度参数", "不提取命令"],
}, ensure_ascii=False)


# ── Tests ──────────────────────────────────────────────────────

def test_meta_architecture_roundtrip(no_expensive_rag):
    """构造实例 → to_dict → json.dumps → loads → from_dict → 字段一致（无 LLM）。"""
    from agent.kg_granularity import MetaArchitecture

    ma = MetaArchitecture(
        entity_types=[{"name": "PartDef", "parent": None, "level": 1,
                       "description": "部件/模块"}],
        root_nodes=[{"name": "系统", "type": "PartDef", "description": "顶层"}],
        relation_patterns=[{"source_type": "PartDef", "target_type": "PartDef",
                            "relation_type": "allocation", "desc": "组成"}],
        constraints=["忽略温度参数", "不提取命令"],
        granularity_description="只提取顶层架构",
        relation_network={"entities": 10, "relations": 3},
    )

    payload = json.dumps(ma.to_dict(), ensure_ascii=False)
    ma2 = MetaArchitecture.from_dict(json.loads(payload))

    assert ma2 == ma
    assert ma2.to_dict() == ma.to_dict()
    assert ma2.granularity_description == "只提取顶层架构"
    assert ma2.entity_types[0]["name"] == "PartDef"
    assert ma2.entity_types[0]["parent"] is None
    assert ma2.root_nodes[0]["name"] == "系统"
    assert ma2.relation_patterns[0]["relation_type"] == "allocation"
    assert ma2.constraints == ["忽略温度参数", "不提取命令"]
    assert ma2.relation_network == {"entities": 10, "relations": 3}

    # JSON 字符串直接往返（to_json/from_json）
    ma3 = MetaArchitecture.from_json(ma.to_json())
    assert ma3 == ma


def test_granularity_agent_mocked(no_expensive_rag):
    """mock light_llm/llm.ainvoke → 三步流程返回正确 MetaArchitecture（无真实 Ollama）。"""
    from agent.kg_build_agent import KGBuildAgent
    from agent.kg_granularity import GranularityAgent, MetaArchitecture

    light_mock = MagicMock()
    light_mock.ainvoke = AsyncMock(return_value=_FakeLLMResponse(_SAMPLE_RESPONSE))
    llm_mock = MagicMock()
    llm_mock.ainvoke = AsyncMock(return_value=_FakeLLMResponse(_META_RESPONSE))

    # 不触发 KGBuildAgent.__init__（避免 config/MCP 副作用）；直接设 backing fields
    agent = object.__new__(KGBuildAgent)
    agent._light_llm = light_mock
    agent._llm = llm_mock

    ga = GranularityAgent(agent, doc_name="测试文档")
    tree_state = _FakeTreeState(_TOC, {nid: nodes for nid, nodes in _SECTIONS.items()})

    meta = asyncio.run(ga.determine_meta_architecture(
        granularity_description="详细到端口和命令级别", tree_state=tree_state, sections=_SECTIONS
    ))

    assert isinstance(meta, MetaArchitecture)
    assert meta.granularity_description == "详细到端口和命令级别"
    assert [t["name"] for t in meta.entity_types] == ["PartDef", "AttributeDef"]
    assert meta.entity_types[1]["parent"] == "PartDef"
    assert meta.root_nodes[0]["name"] == "系统概览"
    assert meta.relation_patterns[0]["relation_type"] == "allocation"
    assert meta.constraints == ["忽略温度参数", "不提取命令"]

    # 每步 LLM 恰好调用一次
    light_mock.ainvoke.assert_awaited_once()
    llm_mock.ainvoke.assert_awaited_once()

    # Step 2 prompt 应包含 TOC 摘要（文档名 + 目录条目）
    step2_human = light_mock.ainvoke.await_args.args[0][1]
    assert "测试文档" in step2_human.content
    assert "n2" in step2_human.content and "总体架构" in step2_human.content

    # Step 3 prompt 应包含采样文本（证明采样结果真正喂给了生成阶段）
    step3_human = llm_mock.ainvoke.await_args.args[0][1]
    assert "总体架构：系统由电源模块与主控模块组成" in step3_human.content
    assert "接口定义：串口与以太网接口" in step3_human.content
    assert "命令表：GET_STATUS、SET_POWER" in step3_human.content
    assert "详细到端口和命令级别" in step3_human.content


# ── T2: relation network statistics ───────────────────────────


def test_analyze_relation_network_basic(no_expensive_rag):
    """三角形 + 链：类型分布 / hub / 连通分量 / max_depth。"""
    from agent.kg_granularity import analyze_relation_network

    relations = [
        {"source": "a", "target": "b", "type": "ConnectionUsage", "name": "r1"},
        {"source": "b", "target": "c", "type": "ConnectionUsage", "name": "r2"},
        {"source": "c", "target": "a", "type": "ConnectionUsage", "name": "r3"},
        {"source": "c", "target": "d", "type": "AllocationUsage", "name": "r4"},
    ]
    report = analyze_relation_network(relations)

    assert report["relation_type_distribution"] == {"ConnectionUsage": 3, "AllocationUsage": 1}
    assert report["num_relations"] == 4
    assert report["num_entities"] == 4
    assert report["self_loops"] == 0
    assert report["hub_nodes"][0] == {"name": "c", "degree": 3}
    assert report["num_components"] == 1
    assert report["component_sizes"] == [4]
    assert report["max_depth"] == 2


def test_analyze_relation_network_empty(no_expensive_rag):
    """空列表：全零结构，不抛异常。"""
    from agent.kg_granularity import analyze_relation_network

    report = analyze_relation_network([])

    assert report == {
        "relation_type_distribution": {},
        "num_relations": 0,
        "num_entities": 0,
        "hub_nodes": [],
        "num_components": 0,
        "component_sizes": [],
        "max_depth": 0,
        "self_loops": 0,
    }


def test_analyze_relation_network_self_loop(no_expensive_rag):
    """仅自环 {a→a}：self_loops=1、单节点单分量、max_depth=0。"""
    from agent.kg_granularity import analyze_relation_network

    report = analyze_relation_network(
        [{"source": "a", "target": "a", "type": "ConnectionUsage", "name": "r1"}]
    )

    assert report["self_loops"] == 1
    assert report["num_relations"] == 1
    assert report["num_entities"] == 1
    assert report["num_components"] == 1
    assert report["component_sizes"] == [1]
    assert report["max_depth"] == 0
    assert report["hub_nodes"] == [{"name": "a", "degree": 1}]


def test_analyze_relation_network_multi_component(no_expensive_rag):
    """两个不相连子图：num_components=2、component_sizes 降序。"""
    from agent.kg_granularity import analyze_relation_network

    relations = [
        {"source": "a", "target": "b", "type": "ConnectionUsage", "name": "r1"},
        {"source": "b", "target": "c", "type": "ConnectionUsage", "name": "r2"},
        {"source": "c", "target": "a", "type": "ConnectionUsage", "name": "r3"},
        {"source": "d", "target": "e", "type": "InterfaceUsage", "name": "r4"},
        {"source": "e", "target": "f", "type": "InterfaceUsage", "name": "r5"},
        {"source": "f", "target": "g", "type": "InterfaceUsage", "name": "r6"},
        {"source": "g", "target": "h", "type": "InterfaceUsage", "name": "r7"},
    ]
    report = analyze_relation_network(relations)

    assert report["num_components"] == 2
    assert report["component_sizes"] == [5, 3]  # 链 d-e-f-g-h (5) + 三角形 a-b-c (3)
    assert report["max_depth"] == 4  # 链 d→h 最短路径 4 跳


def test_analyze_relation_network_undirected_dedup(no_expensive_rag):
    """无向去重：a→b 与 b→a 视为同一条边，degree 不重复计数。"""
    from agent.kg_granularity import analyze_relation_network

    relations = [
        {"source": "a", "target": "b", "type": "ConnectionUsage", "name": "r1"},
        {"source": "b", "target": "a", "type": "ConnectionUsage", "name": "r2"},
    ]
    report = analyze_relation_network(relations)

    assert report["num_relations"] == 2
    assert report["num_entities"] == 2
    assert report["hub_nodes"] == [
        {"name": "a", "degree": 1},
        {"name": "b", "degree": 1},
    ]
    assert report["num_components"] == 1
    assert report["max_depth"] == 1
