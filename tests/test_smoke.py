"""Smoke tests: verify the pytest infrastructure itself works.

These tests make no real system calls: they only check that the suite is
collectable, fixtures are importable, and settings can be redirected away
from the real database.
"""

from __future__ import annotations


def test_truth() -> None:
    assert True


def test_config_module_importable() -> None:
    import config

    assert config.settings is not None


def test_settings_override_fixture_importable(settings_override) -> None:
    assert settings_override.ENV == "test"
    # The override must point away from the real ./database.
    assert settings_override.PERSIST_DIR != "./database"
    assert settings_override.RAG_DB_NAME is None


def test_mock_rag_engine_fixture_importable(mock_rag_engine) -> None:
    import rag.engine as engine_module

    assert engine_module.RAGEngine is mock_rag_engine
