"""Import contract smoke tests -- locks the public module import surface.

Part of the nanite-oop-refactor plan (T3). These tests are pure
observation: they execute the *real* import paths and assert the
contracts downstream code depends on:

1. ``agent.tools`` imports successfully and exposes ``tools`` with
   >= 16 entries (16 active tools + skill tools).
2. ``agent.tools.rag_engine`` is a non-None singleton
   (``rag_engine is RAGEngine()``).
3. ``sysml.sysml_model`` exposes the 18 core SysML classes.
4. ``agent.kg_build_agent`` imports successfully.
5. ``rag.engine`` imports successfully.
6. All four production import sites stay compatible.

The only thing stubbed is RAGEngine's expensive ``__init__``
(embedding/rerank model loading). Imports are never mocked -- they run
for real, module-level side effects included.

CRITICAL: agent.tools executes ``rag_engine = RAGEngine()`` at module
level (agent/tools.py:75), and binds ``RAGEngine`` via
``from rag.engine import RAGEngine`` (agent/tools.py:19). The stub must
therefore be applied *before* any module under the ``agent`` package is
first imported. All imports below happen inside test bodies, after the
``no_expensive_rag`` fixture has patched ``rag.engine.RAGEngine``.

Note on the stub: conftest's ``mock_rag_engine`` fixture replaces
RAGEngine with a stateless stub that has NO ``__new__`` singleton guard,
so ``rag_engine is RAGEngine()`` would be False under it. This file
therefore uses its own stub that preserves the singleton identity
contract (rag/engine.py:344 __new__) while still skipping all model
loading.
"""

from __future__ import annotations

import pytest

# The 18 core SysML classes every caller may import from sysml.sysml_model.
EXPECTED_SYSML_CLASSES = [
    "PartDef", "AttributeDef", "PortDef", "ItemDef", "RequirementDef",
    "PartUsage", "AttributeUsage", "PortUsage", "ItemUsage", "RequirementUsage",
    "ConnectionUsage", "InterfaceUsage", "AllocationUsage", "ConnectionEnd",
    "DirectionKind", "Multiplicity", "Namespace", "Package",
]

MIN_TOOL_COUNT = 16  # 16 active tools + _build_skill_tools()


class StubRAGEngine:
    """Cheap stand-in for rag.engine.RAGEngine.

    Keeps the singleton ``__new__`` guard from the real class
    (rag/engine.py __new__) so the identity contract
    ``rag_engine is RAGEngine()`` remains observable, while skipping
    all expensive work (OpenAI client / embedding / rerank loading).
    """

    _instance = None
    _initialized = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def no_expensive_rag(monkeypatch):
    """Patch rag.engine.RAGEngine to StubRAGEngine before agent.* imports.

    Must run before the first ``import agent.tools``: tools.py binds
    ``RAGEngine`` at module level and instantiates ``rag_engine`` right
    away.
    """
    import rag.engine as engine_module

    monkeypatch.setattr(engine_module, "RAGEngine", StubRAGEngine)
    return StubRAGEngine


def test_agent_tools_import_and_tool_count(no_expensive_rag) -> None:
    import agent.tools

    assert isinstance(agent.tools.tools, list)
    assert len(agent.tools.tools) >= MIN_TOOL_COUNT


def test_rag_engine_is_non_none_singleton(no_expensive_rag) -> None:
    import agent.tools
    import rag.engine as engine_module

    rag_engine = agent.tools.rag_engine
    assert rag_engine is not None
    # Singleton identity: a fresh RAGEngine() call returns the same object.
    assert rag_engine is engine_module.RAGEngine()


def test_sysml_model_18_classes_importable(no_expensive_rag) -> None:
    import sysml.sysml_model as sysml_model

    missing = [name for name in EXPECTED_SYSML_CLASSES
               if not hasattr(sysml_model, name)]
    assert not missing, f"sysml.sysml_model missing classes: {missing}"


def test_agent_kg_build_agent_importable(no_expensive_rag) -> None:
    import agent.kg_build_agent

    assert agent.kg_build_agent is not None


def test_rag_engine_module_importable(no_expensive_rag) -> None:
    import rag.engine as engine_module

    # Module importable, and the stub is in place (no real model loading).
    assert engine_module.RAGEngine is StubRAGEngine


def test_import_points_compatibility(no_expensive_rag) -> None:
    """The four production import sites must keep working.

    - ``from agent.tools import tools, rag_engine``
    - ``from agent.tools import rag_engine, tools``
      (scripts/test_tools_invocation.py:270)
    - ``from agent.tools import tools``
      (agent/agent.py:30, web_server.py:24 as ``tools as registered_tools``,
       scripts/test_search_web_tools.py:65)
    """
    # Form 1: tools, rag_engine order
    from agent.tools import tools, rag_engine
    assert len(tools) >= MIN_TOOL_COUNT
    assert rag_engine is not None

    # Form 2: rag_engine, tools order (test_tools_invocation.py:270)
    from agent.tools import rag_engine as re2, tools as t2
    assert t2 is tools
    assert re2 is rag_engine

    # Form 3: tools only (agent.py:30 / web_server.py:24 / test_search_web_tools.py:65)
    from agent.tools import tools as tools_only
    assert tools_only is tools