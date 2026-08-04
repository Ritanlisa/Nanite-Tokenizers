"""Regression test for CandidateExtractionEngine.extract_page_candidates.

T9 verification found a pre-existing bug (refactor-independent): the success
path of extract_page_candidates references the unassigned variable `raw` in
logger.debug (`len(raw)`, `raw[:500]`). The raised UnboundLocalError (a
NameError subclass) was swallowed by the surrounding `except Exception`,
which logged a warning and `continue`d past the chunk, silently dropping the
whole page's candidates - disabling build_kg_from_document Phase 1 extraction.

Fix: assign `raw` from the ainvoke response before the debug logs:
    raw = str(response.content) if hasattr(response, "content") else str(response)

This test mocks light_llm.ainvoke to return a successful response with a
`content` attribute and asserts the page candidates ARE parsed (red before
the fix, green after).

Import hygiene: importing `agent.kg_build_agent` transitively imports
`agent.tools` (via `agent.chatOpenAIWithReasoning`), and `agent.tools`
executes `rag_engine = RAGEngine()` at module level (agent/tools.py:75).
tests/test_import_contract.py asserts the singleton identity
`agent.tools.rag_engine is RAGEngine()` under ITS stub, so the stub must be
applied BEFORE the first `import agent.*` (the same contract that
test_import_contract documents). We therefore patch rag.engine.RAGEngine to
test_import_contract.StubRAGEngine when it is already loaded (full-suite
run: both files then observe the SAME singleton class, preserving the
identity assertion); when this file is run standalone we fall back to an
equivalent local singleton stub (test_import_contract is not collected, so
its identity assertion cannot run anyway).
"""

from __future__ import annotations

import asyncio
import sys
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


def _build_engine(llm):
    """Construct a CandidateExtractionEngine without running any __init__."""
    from agent.kg_build_agent import CandidateExtractionEngine, KGBuildAgent

    agent = object.__new__(KGBuildAgent)
    agent._light_llm = llm  # backing field used by the light_llm property
    return CandidateExtractionEngine(agent)


def test_extract_page_candidates_success_path_parses_candidates(no_expensive_rag):
    payload = (
        '[{"name": "PumpBody", "type": "PartDef", '
        '"description": "water pump", "aliases": []}, '
        '{"relation": true, "type": "connection", '
        '"source": "PumpBody", "target": "Motor", "description": "drives"}]'
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=_FakeLLMResponse(payload))

    candidates, relations = asyncio.run(
        _build_engine(llm).extract_page_candidates(
            "The pump body connects to a motor.", "Pump Section", 3
        )
    )

    # Under a successful LLM response the candidates MUST be parsed.
    # (Before the fix: UnboundLocalError on `raw` was swallowed -> empty.)
    assert candidates, "page candidates must be non-empty on the success path"
    assert [c["name"] for c in candidates] == ["PumpBody"]
    assert len(relations) == 1
