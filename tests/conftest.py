"""Shared pytest fixtures for the Nanite-Tokenizers project.

Fixtures here intentionally avoid touching the real database, MCP servers,
or any expensive model initialization (embedding/rerank models).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable (config.py, rag/, agent/, ... live there).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

import config


@pytest.fixture
def settings_override(tmp_path, monkeypatch):
    """Redirect config.settings to a temporary, throwaway environment.

    Writes a temp settings.yaml and monkeypatches the module-level
    config.settings singleton so no test touches the real ./database,
    ./data, or the production settings file.
    """
    temp_db = tmp_path / "database"
    temp_data = tmp_path / "data"

    (tmp_path / "settings.yaml").write_text(
        "ENV: test\n"
        f"PERSIST_DIR: {temp_db}\n"
        f"DATA_DIR: {temp_data}\n",
        encoding="utf-8",
    )

    overridden = config.settings.update(
        ENV="test",
        PERSIST_DIR=str(temp_db),
        DATA_DIR=str(temp_data),
        RAG_DB_NAME=None,
        RAG_DB_NAMES=[],
    )
    monkeypatch.setattr(config, "settings", overridden)
    return overridden


@pytest.fixture
def mock_rag_engine(monkeypatch):
    """Replace rag.engine.RAGEngine with a cheap stub.

    The real RAGEngine lazily loads embedding/rerank models in __init__,
    which is far too expensive for unit tests. Explicit (not autouse):
    request it only when a test actually needs it.
    """
    import rag.engine as engine_module

    class StubRAGEngine:
        _instance = None
        _initialized = True

        def __init__(self, *args, **kwargs):
            pass

        def query(self, *args, **kwargs):
            raise NotImplementedError("StubRAGEngine.query is not implemented")

        def upsert_documents(self, *args, **kwargs):
            raise NotImplementedError("StubRAGEngine.upsert_documents is not implemented")

    monkeypatch.setattr(engine_module, "RAGEngine", StubRAGEngine)
    return StubRAGEngine


@pytest.fixture(autouse=True)
def reset_sysml_global_state():
    """Reset sysml_rag_mcp_server module globals before every test.

    _global_manager / _HV_RESOLVER / _loaded_files are process-global
    singletons shared across ALL test files (including integration-tagged
    test_mcp_server.py / test_kg_api.py). Without a reset between tests,
    data added by one test leaks into the next (e.g. test_k_layer_traversal
    sees extra entities and fails `assert 5 == 4`).
    """
    import scripts.sysml_rag_mcp_server as _mcp

    _mcp._global_manager = None
    _mcp._HV_RESOLVER = None
    _mcp._loaded_files.clear()
    yield
