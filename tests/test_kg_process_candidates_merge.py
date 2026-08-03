"""T8 differential tests: merged _process_candidates serial vs batch paths.

Both paths must produce identical results (created entities/relations and
the resolved entity_map) for the same input, while the mock MCP session
records every top-level call_tool invocation.

The mock implements a tiny in-memory SysML database so sysml_batch and the
single-tool calls observe the SAME semantics: searching an existing entity
matches, add_entity registers its QN, add_relation records the edge.
Running the serial path and the batch path against separate fresh databases
must yield identical entity/relation sets, counts and resolved QNs.
"""

from __future__ import annotations

import asyncio
import json

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
    """In-memory SysML store + call recorder. call_tool records TOP-level
    invocations only; sysml_batch dispatches its operations internally."""

    def __init__(self):
        self.calls = []  # list[(tool, arguments)]
        self.qn_by_name = {}
        self.entities = {}
        self.relations = []

    def _seed_entity(self, name, qn, aliases=()):
        self.qn_by_name[name] = qn
        self.entities[qn] = {"name": name, "qualified_name": qn, "aliases": list(aliases)}

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
            self.entities[qn] = {"name": name, "qualified_name": qn, "aliases": list(args.get("aliases") or [])}
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
        if tool == "sysml_search_entity":
            return json.dumps(self._match(arguments.get("query", "")))
        if tool == "sysml_add_entity":
            return json.dumps(self._add_entity(arguments))
        if tool == "sysml_update_entity":
            return json.dumps({"ok": True})
        if tool == "sysml_add_alias":
            return json.dumps({"ok": True})
        if tool == "sysml_add_relation":
            self.relations.append(
                {
                    "relation_type": arguments.get("relation_type"),
                    "source": arguments.get("source"),
                    "target": arguments.get("target"),
                    "name": arguments.get("name"),
                    "description": arguments.get("description"),
                }
            )
            return json.dumps({"ok": True})
        raise AssertionError("unexpected tool: %s" % tool)


def _build_agent(mcp):
    """Construct a KGBuildAgent without touching __init__/initialize."""
    from agent.kg_build_agent import KGBuildAgent

    agent = object.__new__(KGBuildAgent)
    agent._mcp_session = mcp
    return agent


def _input():
    candidates = [
        {"name": "Valve Assembly", "type": "PartDef", "description": "existing part", "aliases": ["ValveUnit"]},
        {"name": "PumpBody", "type": "PartDef", "description": "new part 1", "aliases": []},
        {"name": "Actuator", "type": "PartDef", "description": "new part 2", "aliases": ["ActuatorUnit"]},
        {"name": "", "type": "PartDef", "description": "skipped", "aliases": []},
        {"name": "Sensor", "type": "PartDef", "description": "new part 3", "aliases": []},
    ]
    relations = [
        {"type": "connection", "source": "PumpBody", "target": "Valve Assembly", "description": "pumps"},
        {"type": "usage", "source": "Actuator", "target": "GearBox", "description": "uses gearbox"},
        {"type": "flow", "source": "", "target": "Sensor", "description": "skipped"},
    ]
    return candidates, relations


def _fresh_mcp():
    mcp = MockMCP()
    mcp._seed_entity("Valve Assembly", "root::ValveAssembly")
    return mcp


def _assert_equivalent(serial_res, batch_res, serial_mcp, batch_mcp):
    assert serial_res["entities"] == batch_res["entities"], (serial_res, batch_res)
    assert serial_res["relations"] == batch_res["relations"], (serial_res, batch_res)
    assert serial_res.get("entity_map") == batch_res.get("entity_map"), (
        serial_res.get("entity_map"),
        batch_res.get("entity_map"),
    )
    assert sorted(serial_mcp.qn_by_name) == sorted(batch_mcp.qn_by_name)
    assert sorted(serial_mcp.entities) == sorted(batch_mcp.entities)
    norm = lambda rels: sorted(
        (r["relation_type"], r["source"], r["target"], r["name"], r["description"]) for r in rels
    )
    assert norm(serial_mcp.relations) == norm(batch_mcp.relations), (
        norm(serial_mcp.relations),
        norm(batch_mcp.relations),
    )


async def _run_both_paths():
    candidates, relations = _input()

    serial_mcp = _fresh_mcp()
    serial_res = await _build_agent(serial_mcp)._process_candidates(
        candidates, relations, "doc", 3, "Section A",
        source_pages=["p3 Section A", "p4 Section B"],
        batch=False,
    )

    batch_mcp = _fresh_mcp()
    batch_res = await _build_agent(batch_mcp)._process_candidates(
        candidates, relations, "doc", 3, "Section A",
        source_pages=["p3 Section A", "p4 Section B"],
        batch=True,
    )

    _assert_equivalent(serial_res, batch_res, serial_mcp, batch_mcp)

    # Expected totals: 3 new candidates + 1 auto-created endpoint (GearBox).
    assert batch_res["entities"] == 4, batch_res
    assert batch_res["relations"] == 2, batch_res
    assert batch_res["entity_map"]["GearBox"] == "root::GearBox"
    assert batch_res["entity_map"]["Valve Assembly"] == "root::ValveAssembly"
    assert batch_res["entity_map"]["PumpBody"] == "root::PumpBody"


def test_serial_and_batch_paths_equivalent(no_expensive_rag):
    asyncio.run(_run_both_paths())


def test_batch_is_default_path(no_expensive_rag):
    async def run():
        mcp = _fresh_mcp()
        res = await _build_agent(mcp)._process_candidates(*_input(), "doc", 3, "Section A")
        assert res["entities"] == 4 and res["relations"] == 2, res
        tools = {t for t, _ in mcp.calls}
        assert tools == {"sysml_batch"}, tools  # no solo tools on default path
    asyncio.run(run())


def test_serial_path_still_reachable(no_expensive_rag):
    async def run():
        mcp = _fresh_mcp()
        res = await _build_agent(mcp)._process_candidates(*_input(), "doc", 3, "Section A", batch=False)
        assert res["entities"] == 4 and res["relations"] == 2, res
        tools = {t for t, _ in mcp.calls}
        assert "sysml_batch" not in tools
        assert {"sysml_add_entity", "sysml_add_relation", "sysml_search_entity"} <= tools
    asyncio.run(run())