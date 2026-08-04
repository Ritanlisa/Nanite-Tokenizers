# -*- coding: utf-8 -*-
"""T6 tests: meta-schema soft filtering in _process_candidates.

- meta_schema allows only PartDef → candidate type=AttributeDef is SKIPPED
  (log records the skip), type=PartDef is created normally; a candidate with
  no explicit type falls back to PartDef and is still created.
- An EXISTING entity is never filtered: even with a disallowed type it still
  gets its source_sections appended (soft filter only guards NEW entities).
- meta_schema returns ok=False → no filtering: every candidate is created
  (identical to the pre-filter behavior).
- meta_schema call raises → no filtering, no crash.
- entity_types given as dicts (real MetaArchitecture.to_dict() shape with
  parent/level hierarchy) → every level name is allowed (hierarchy expansion).

Import hygiene mirrors tests/test_kg_process_candidates_merge.py
(no_expensive_rag stub for rag.engine.RAGEngine).
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest


class _StubRAGEngine:
    """Singleton-preserving cheap stand-in for rag.engine.RAGEngine."""

    _instance = None
    _initialized = True

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def no_expensive_rag(monkeypatch):
    import rag.engine as engine_module

    monkeypatch.setattr(engine_module, "RAGEngine", _StubRAGEngine)


class MockMCP:
    """In-memory SysML store + call recorder, with a configurable
    sysml_get_meta_schema response (dict, or an Exception to raise)."""

    def __init__(self, meta_response=None):
        self.calls = []  # list[(tool, arguments)]
        self.qn_by_name = {}
        self.entities = {}
        self.relations = []
        self.meta_response = meta_response if meta_response is not None else {"ok": False}

    def _seed_entity(self, name, qn, aliases=()):
        self.qn_by_name[name] = qn
        self.entities[qn] = {
            "name": name,
            "qualified_name": qn,
            "aliases": list(aliases),
            "entity_type": "PartDef",
        }

    def _match(self, query):
        for name, qn in self.qn_by_name.items():
            if query == name:
                return {"total_matches": 1, "matches": [{"qualified_name": qn, "confidence": 0.9}]}
        return {"total_matches": 0, "matches": []}

    def _add_entity(self, args):
        name = args["name"]
        if name not in self.qn_by_name:
            qn = f"root::{name}"
            self.qn_by_name[name] = qn
            self.entities[qn] = {
                "name": name,
                "qualified_name": qn,
                "aliases": list(args.get("aliases") or []),
                "entity_type": args.get("entity_type"),
            }
        return {"ok": True, "qualified_name": self.qn_by_name[name]}

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        return await self._dispatch(tool, arguments)

    async def _dispatch(self, tool, arguments):
        if tool == "sysml_batch":
            results = []
            for idx, op in enumerate(arguments.get("operations", [])):
                raw = await self._dispatch(op["tool"], op.get("arguments", {}))
                results.append({"index": idx, "ok": True, "result": json.loads(raw)})
            return json.dumps({"results": results})
        if tool == "sysml_get_meta_schema":
            if isinstance(self.meta_response, Exception):
                raise self.meta_response
            return json.dumps(self.meta_response)
        if tool == "sysml_search_entity":
            return json.dumps(self._match(arguments.get("query", "")))
        if tool == "sysml_add_entity":
            return json.dumps(self._add_entity(arguments))
        if tool == "sysml_update_entity":
            return json.dumps({"ok": True})
        if tool == "sysml_add_alias":
            return json.dumps({"ok": True})
        if tool == "sysml_add_relation":
            self.relations.append(arguments.get("relation_type"))
            return json.dumps({"ok": True})
        raise AssertionError("unexpected tool: %s" % tool)


def _build_agent(mcp):
    """Construct a KGBuildAgent without touching __init__/initialize."""
    from agent.kg_build_agent import KGBuildAgent

    agent = object.__new__(KGBuildAgent)
    agent._mcp_session = mcp
    return agent


def _candidates():
    return [
        {"name": "Valve", "type": "PartDef", "description": "allowed part"},
        {"name": "Pressure", "type": "AttributeDef", "description": "disallowed attr"},
        {"name": "NoType", "description": "no explicit type (defaults to PartDef)"},
    ]


async def _run(meta_response, batch=True):
    mcp = MockMCP(meta_response)
    res = await _build_agent(mcp)._process_candidates(
        _candidates(), [], "doc", 1, "Section A",
        source_pages=["p1 Section A"], batch=batch,
    )
    return mcp, res


def _created_names(mcp):
    return {e["name"] for e in mcp.entities.values()}


@pytest.mark.parametrize("batch", [True, False])
def test_disallowed_type_skipped_allowed_created(no_expensive_rag, caplog, batch):
    """meta allows only PartDef → AttributeDef candidate skipped, PartDef
    (and the no-type candidate falling back to PartDef) created."""
    caplog.set_level(logging.DEBUG)
    meta = {"ok": True, "schema": {"entity_types": ["PartDef"]}}
    mcp, res = asyncio.run(_run(meta, batch=batch))

    assert res["entities"] == 2, res  # Valve + NoType only
    assert _created_names(mcp) == {"Valve", "NoType"}
    assert "Pressure" not in _created_names(mcp)

    text = caplog.text
    assert "Skipping entity 'Pressure' type=AttributeDef (not in meta-schema)" in text
    assert "Skipping entity 'Valve'" not in text


@pytest.mark.parametrize("batch", [True, False])
def test_existing_entity_never_filtered(no_expensive_rag, caplog, batch):
    """An existing entity with a disallowed type still gets source_sections
    appended — soft filter only guards NEW entities."""
    caplog.set_level(logging.DEBUG)
    meta = {"ok": True, "schema": {"entity_types": ["PartDef"]}}
    mcp = MockMCP(meta)
    mcp._seed_entity("Pressure", "root::Pressure")  # exists as AttributeDef already

    candidates = [{"name": "Pressure", "type": "AttributeDef", "description": "existing"}]
    res = asyncio.run(_build_agent(mcp)._process_candidates(
        candidates, [], "doc", 1, "Section A",
        source_pages=["p1 Section A"], batch=batch,
    ))

    assert res["entities"] == 0, res  # nothing created
    tools = {t for t, _ in mcp.calls}
    update_seen = "sysml_update_entity" in tools
    if not update_seen:
        # batch path: sysml_update_entity is nested inside the sysml_batch op
        for t, args in mcp.calls:
            if t == "sysml_batch":
                ops = [op["tool"] for op in args.get("operations", [])]
                update_seen = "sysml_update_entity" in ops
    assert update_seen, mcp.calls  # source_sections appended, not created
    text = caplog.text
    assert "Skipping entity 'Pressure'" not in text


@pytest.mark.parametrize("batch", [True, False])
def test_meta_ok_false_no_filtering(no_expensive_rag, batch):
    """ok=False → no filtering: every candidate is created unchanged."""
    mcp, res = asyncio.run(_run({"ok": False}, batch=batch))
    assert res["entities"] == 3, res
    assert _created_names(mcp) == {"Valve", "Pressure", "NoType"}


@pytest.mark.parametrize("batch", [True, False])
def test_meta_call_exception_no_filtering(no_expensive_rag, batch):
    """sysml_get_meta_schema raises → no filtering, no crash."""
    mcp, res = asyncio.run(_run(RuntimeError("meta unavailable"), batch=batch))
    assert res["entities"] == 3, res
    assert _created_names(mcp) == {"Valve", "Pressure", "NoType"}


@pytest.mark.parametrize("batch", [True, False])
def test_dict_entity_types_hierarchy_expansion(no_expensive_rag, batch):
    """entity_types as dicts (real MetaArchitecture.to_dict() shape) — every
    level name is collected into the allowed set."""
    meta = {
        "ok": True,
        "schema": {
            "entity_types": [
                {"name": "PartDef", "parent": None, "level": 1, "description": "part"},
                {"name": "AttributeDef", "parent": "PartDef", "level": 2, "description": "attr"},
            ]
        },
    }
    mcp, res = asyncio.run(_run(meta, batch=batch))
    assert res["entities"] == 3, res
    assert _created_names(mcp) == {"Valve", "Pressure", "NoType"}


@pytest.mark.parametrize("batch", [True, False])
def test_empty_entity_types_no_filtering(no_expensive_rag, batch):
    """schema 存在但 entity_types=[] → 空集不再屏蔽所有新建实体（F2-F1）：
    空 entity_types 视为无 meta，全部候选正常创建。"""
    meta = {"ok": True, "schema": {"entity_types": [], "relation_patterns": []}}
    mcp, res = asyncio.run(_run(meta, batch=batch))
    assert res["entities"] == 3, res
    assert _created_names(mcp) == {"Valve", "Pressure", "NoType"}
