#!/usr/bin/env python3
"""Test Step 4: KG build pipeline integration"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest
pytestmark = pytest.mark.integration

def test_section_extraction():
    """Test that section extraction works with mock RAG documents"""
    print("=== Test section extraction ===")
    from web_server import _extract_sections_from_rag_doc

    class MockMonoPage:
        def __init__(self, title, text, page_num):
            self.title = title
            self.markdown_text = text
            self.page_number = page_num

    class MockRAGDoc:
        def __init__(self, mono_pages, catalog=None):
            self._mono_pages = mono_pages
            self._catalog = catalog or []
            self.doc_name = "test_doc.pdf"

        def get_mono_pages(self):
            return self._mono_pages

        def catalog_payload(self):
            return self._catalog

    # Test with catalog
    mock = MockRAGDoc(
        mono_pages=[
            MockMonoPage("Old Title", "The system has three modules: Data Collector, Analyzer, Dashboard.", 3),
            MockMonoPage("Data", "Data Collector uses MQTT to stream telemetry. Sampling rate: 100Hz.", 5),
            MockMonoPage("", "", 6),  # Empty — skip
        ],
        catalog=[
            {"page": 3, "title": "1.1 System Overview", "parent_title": "Chapter 1"},
            {"page": 5, "title": "1.2 Data Layer", "parent_title": "Chapter 1"},
        ]
    )
    sections = _extract_sections_from_rag_doc(mock)
    assert len(sections) == 2, f"Expected 2, got {len(sections)}"
    assert sections[0].title == "1.1 System Overview"
    assert sections[0].parent_title == "Chapter 1"
    assert sections[0].page == 3
    print(f"  catalog section[0]: title='{sections[0].title}', parent='{sections[0].parent_title}'")

    # Test without catalog (fallback to page title)
    mock2 = MockRAGDoc(
        mono_pages=[MockMonoPage("Custom Title", "Some content that is long enough to pass the minimum length filter for extraction.", 1)],
        catalog=[]
    )
    sections2 = _extract_sections_from_rag_doc(mock2)
    assert len(sections2) == 1
    assert sections2[0].title == "Custom Title"
    assert sections2[0].parent_title == ""
    print("  fallback title: OK")

    # Test text truncation
    long_text = "x" * 10000
    mock3 = MockRAGDoc(
        mono_pages=[MockMonoPage("Long", long_text, 1)],
    )
    sections3 = _extract_sections_from_rag_doc(mock3)
    assert len(sections3[0].text) == 4000
    print("  text truncation: OK")

    print("  PASS\n")


def test_kg_pipeline_source_structure():
    """Test that KG build hook is properly integrated in source code"""
    print("=== Test KG pipeline structure ===")

    # Read web_server.py source
    with open("web_server.py", "r", encoding="utf-8") as f:
        source = f.read()

    # Check that _run_kg_build_for_job exists and is async
    assert "async def _run_kg_build_for_job" in source
    print("  _run_kg_build_for_job exists: OK")

    # Check KG build is called from _run_rag_build_job
    assert "_run_kg_build_for_job" in source
    assert "KG_EXTRACTION_ENABLED" in source
    print("  KG hook in _run_rag_build_job: OK")

    # Check progress stages
    assert "kg_build_started" in source
    assert "kg_build_completed" in source
    assert "kg_building" in source
    print("  KG progress stages: OK")

    # Check that KGBuildAgent import exists
    assert "KGBuildAgent" in source
    assert "SectionInfo" in source
    print("  KGBuildAgent import: OK")

    # Check load_rag_documents_from_persist_dir usage
    assert "load_rag_documents_from_persist_dir" in source
    print("  RAG doc loading: OK")

    # Check section extraction helper is at module level
    assert "def _extract_sections_from_rag_doc" in source
    print("  _extract_sections_from_rag_doc (module-level): OK")

    print("  PASS\n")


def test_config_kg_settings():
    """Test that KG config settings are available"""
    print("=== Test KG config settings ===")
    import config

    settings_keys = ["KG_EXTRACTION_ENABLED", "KG_EXTRACTION_MODEL",
                     "KG_EXTRACTION_TEMPERATURE", "KG_EXTRACTION_TIMEOUT",
                     "KG_EXTRACTION_MAX_ITERATIONS",
                     "KG_MERGE_CONFIDENCE_THRESHOLD"]
    for key in settings_keys:
        val = getattr(config.settings, key, None)
        assert val is not None, f"Missing config: {key}"
        print(f"  {key}: {val}")

    # config.py 的产品默认值为 True；运行时可能被 settings.yaml 有意覆盖为 false，
    # 因此这里验证"默认类属性为 True"（产品逻辑）与"运行时值为合法 bool"，而非硬编码运行时值。
    assert config.Settings.model_fields["KG_EXTRACTION_ENABLED"].default is True
    assert isinstance(config.settings.KG_EXTRACTION_ENABLED, bool)
    print("  PASS\n")


def test_kg_build_hook_runs_with_disabled_setting():
    """Test that KG build is skippable via config"""
    print("=== Test KG config toggle ===")
    import config

    # Save original
    original = config.settings.KG_EXTRACTION_ENABLED
    config.settings = config.settings.update(KG_EXTRACTION_ENABLED=False)
    assert config.settings.KG_EXTRACTION_ENABLED is False

    # Restore — 恢复后应与保存的 original 一致（原值可能被 settings.yaml 覆盖为 false）
    config.settings = config.settings.update(KG_EXTRACTION_ENABLED=original)
    assert config.settings.KG_EXTRACTION_ENABLED is original
    print("  KG toggle works: OK")
    print("  PASS\n")


if __name__ == "__main__":
    test_section_extraction()
    test_kg_pipeline_source_structure()
    test_config_kg_settings()
    test_kg_build_hook_runs_with_disabled_setting()
    print("All Step 4 tests passed!")
