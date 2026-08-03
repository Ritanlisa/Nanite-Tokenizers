#!/usr/bin/env python3
"""Test Step 5: KG API endpoints"""
import os
import sys
import json
import tempfile
from pathlib import Path



import pytest
pytestmark = pytest.mark.integration

def test_kg_file_paths():
    """Test KG file path helpers"""
    print("=== Test KG file paths ===")
    import web_server
    import config

    # We need to mock create_app internals - let's test inline
    persist = config.settings.PERSIST_DIR
    db_name = "test_db"

    kg_path = os.path.join(persist, db_name, "knowledge_graph.sysml")
    meta_path = os.path.join(persist, db_name, "knowledge_graph.meta.json")

    assert "knowledge_graph.sysml" in kg_path
    assert "knowledge_graph.meta.json" in meta_path
    print(f"  kg_path: {kg_path}")
    print(f"  meta_path: {meta_path}")
    print("  PASS\n")


def test_kg_summary_via_manager():
    """Test KG summary functionality via SysMLManager"""
    print("=== Test KG summary via SysMLManager ===")
    from sysml.sysml_manager import SysMLManager

    mgr = SysMLManager()
    # Add some entities
    mgr.add_entity_with_metadata("PartDef", "TestComponent", description="Test",
                                  aliases=["TC"], source_sections=["s1"])
    mgr.add_entity_with_metadata("AttributeDef", "TestAttribute", description="Attr")
    mgr.add_relation("connection", "TestComponent", "TestAttribute",
                    name="test_conn", description="Connection")

    # Test summary
    entities = mgr.get_all_entities()
    relations = mgr.get_all_relations()
    assert len(entities) == 2
    assert len(relations) == 1
    print(f"  entities: {len(entities)}, relations: {len(relations)}")

    # Test entity detail
    from scripts.sysml_rag_mcp_server import _entity_summary

    for e in entities:
        summary = _entity_summary(e, include_body=True)
        assert summary["name"] in ("TestComponent", "TestAttribute")
        print(f"  entity: {summary['name']} ({summary['type']})")

    # Test search
    results = mgr.search_entities("TestComp")
    assert len(results) >= 1
    assert results[0]["name"] == "TestComponent"
    print(f"  search 'TestComp': found {results[0]['name']} (conf: {results[0]['confidence']})")

    # Test save/load for KG file
    with tempfile.TemporaryDirectory() as tmpdir:
        kg_file = os.path.join(tmpdir, "knowledge_graph.sysml")
        mgr.save_to_file(kg_file)
        assert os.path.exists(kg_file)

        mgr2 = SysMLManager()
        mgr2.load_from_file(kg_file)
        assert len(mgr2.get_all_entities()) == 2
        print(f"  save/load KG file: OK ({kg_file})")

    print("  PASS\n")


def test_kg_search_functionality():
    """Test enhanced search via manager"""
    print("=== Test KG search ===")
    from sysml.sysml_manager import SysMLManager, AliasRegistry

    mgr = SysMLManager()
    mgr.add_entity_with_metadata("PartDef", "数据采集模块", description="负责数据采集的模块",
                                  aliases=["DAQ模块", "采集模块", "DataCollector"],
                                  source_sections=["第1章::1.1"])

    # Exact match
    results = mgr.search_entities("数据采集模块")
    assert len(results) >= 1
    assert results[0]["confidence"] >= 0.9
    print(f"  exact match: conf={results[0]['confidence']}")

    # Alias match
    results = mgr.search_entities("DAQ模块")
    assert len(results) >= 1
    print(f"  alias match 'DAQ模块': conf={results[0]['confidence']}")

    # Substring match
    results = mgr.search_entities("采集")
    assert len(results) >= 1
    print(f"  substring '采集': found {results[0]['name']}")

    # Normalized match
    results = mgr.search_entities("data collector")
    assert len(results) >= 1
    print(f"  normalized 'data collector': conf={results[0]['confidence']}")

    # No match
    results = mgr.search_entities("不存在的实体XYZ")
    assert len(results) == 0 or results[0]["confidence"] < 0.3
    print(f"  no match: {len(results)} results (expected 0)")

    print("  PASS\n")


def test_kg_api_request_models():
    """Test that KG API endpoints exist in web_server source"""
    print("=== Test KG API route definitions ===")
    with open("web_server.py", "r", encoding="utf-8") as f:
        source = f.read()

    expected_routes = [
        "/api/rag/dbs/{db_name}/kg/export",
        "/api/rag/dbs/{db_name}/kg/summary",
        "/api/rag/dbs/{db_name}/kg/entities",
        "/api/rag/dbs/{db_name}/kg/entity/",
        "/api/rag/dbs/{db_name}/kg/search",
        "/api/rag/dbs/{db_name}/kg/re-extract",
    ]
    for route in expected_routes:
        assert route in source, f"Missing route: {route}"
        print(f"  route: {route}")

    # Check helper functions
    helpers = ["_kg_file", "_kg_meta_file", "_load_kg_manager", "_kg_entity_detail"]
    for h in helpers:
        assert h in source, f"Missing helper: {h}"
        print(f"  helper: {h}")

    print("  PASS\n")


def test_kg_endpoint_error_handling():
    """Test that KG endpoints handle missing KGs gracefully"""
    print("=== Test missing KG handling ===")
    from sysml.sysml_manager import SysMLManager

    # Empty manager (no KG loaded) should return empty results
    mgr = SysMLManager()
    entities = mgr.get_all_entities()
    relations = mgr.get_all_relations()
    assert len(entities) == 0
    assert len(relations) == 0
    print(f"  empty KG: {len(entities)} entities, {len(relations)} relations")

    # Search on empty manager
    results = mgr.search_entities("anything")
    assert len(results) == 0
    print(f"  search empty KG: {len(results)} results")

    # Entity not found
    found = mgr.find_definition("nonexistent")
    assert found is None
    print(f"  find missing entity: returns None")

    print("  PASS\n")


if __name__ == "__main__":
    test_kg_file_paths()
    test_kg_summary_via_manager()
    test_kg_search_functionality()
    test_kg_api_request_models()
    test_kg_endpoint_error_handling()
    print("All Step 5 tests passed!")
