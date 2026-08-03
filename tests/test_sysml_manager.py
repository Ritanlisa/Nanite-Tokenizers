#!/usr/bin/env python3
"""Test sysml_manager.py enhancements: AliasRegistry + CRUD + merge + search"""


from sysml.sysml_manager import (
    SysMLManager, AliasRegistry
)

def test_alias_registry():
    print("=== Test AliasRegistry ===")
    reg = AliasRegistry()

    # Normalization
    assert AliasRegistry.normalize("DAQ Module") == "daqmodule"
    assert AliasRegistry.normalize("数据采集模块") == "数据采集模块"
    assert AliasRegistry.normalize("DA Q-Module_1") == "daqmodule1"
    print("  normalize: OK")

    # Register
    reg.register("doc::3::数据采集模块", ["DAQ模块", "采集模块", "DAQ Module"])
    assert len(reg._alias_map) == 3
    print("  register: OK")

    # Lookup
    assert reg.lookup("DAQ模块") == "doc::3::数据采集模块"
    assert reg.lookup("DAQ Module") == "doc::3::数据采集模块"
    assert reg.lookup("nonexistent") is None
    print("  lookup: OK")

    # Search
    results = reg.search("DAQ")
    assert len(results) > 0
    assert results[0]["qualified_name"] == "doc::3::数据采集模块"
    print(f"  search 'DAQ': {len(results)} results, top confidence={results[0]['confidence']}")

    # Search exact
    results = reg.search("数据采集模块")
    assert len(results) > 0
    assert results[0]["confidence"] >= 0.7
    print("  search exact: OK")

    # Serialize
    d = reg.to_dict()
    reg2 = AliasRegistry.from_dict(d)
    assert reg2.lookup("DAQ模块") == "doc::3::数据采集模块"
    print("  serialize/deserialize: OK")

    # Transfer
    reg.register("doc::3::目标实体", ["target"])
    reg.transfer("doc::3::数据采集模块", "doc::3::目标实体")
    assert reg.lookup("DAQ模块") == "doc::3::目标实体"
    assert reg.lookup("采集模块") == "doc::3::目标实体"
    assert "doc::3::数据采集模块" not in reg._aliases_by_entity
    print("  transfer: OK")

    print("  PASS\n")


def test_add_entity():
    print("=== Test add_entity_with_metadata ===")
    mgr = SysMLManager()

    elem = mgr.add_entity_with_metadata(
        entity_type="PartDef",
        name="数据采集模块",
        parent_package="系统架构",
        description="负责从多个传感器节点采集数据",
        aliases=["DAQ模块", "采集模块"],
        source_sections=["第3章::3.1 数据层"],
        source_text="数据采集模块是系统的数据入口",
        properties={"采样率": "100Hz"},
    )
    assert elem is not None
    assert elem.name == "数据采集模块"
    qn = elem.qualified_name
    assert "系统架构" in qn
    print(f"  created: {qn}")

    # Check metadata
    meta = mgr.get_entity_metadata(qn)
    assert meta["description"] == "负责从多个传感器节点采集数据"
    assert "第3章::3.1 数据层" in meta["source_sections"]
    assert meta["properties"]["采样率"] == "100Hz"
    print("  metadata: OK")

    # Check aliases
    assert mgr._alias_registry.lookup("DAQ模块") == qn
    assert mgr._alias_registry.lookup("da q模块") is not None  # 归一化匹配
    print("  aliases: OK")

    # Search
    results = mgr.search_entities("DAQ")
    assert len(results) > 0
    assert results[0]["qualified_name"] == qn
    print(f"  search 'DAQ': found, confidence={results[0]['confidence']}")

    # Add second entity
    elem2 = mgr.add_entity_with_metadata(
        entity_type="AttributeDef",
        name="采样率",
        parent_package="系统架构",
        description="数据采集频率",
        aliases=["采样频率", "SampleRate"],
    )
    assert elem2 is not None
    print(f"  created second: {elem2.qualified_name}")

    # Update first entity
    mgr.update_entity_metadata(qn,
        append_description="支持MQTT协议",
        append_source_sections=["第5章::5.2 扩展性"],
        merge_aliases=["DAQ Unit"],
    )
    meta = mgr.get_entity_metadata(qn)
    assert "MQTT协议" in meta["description"]
    assert "第5章::5.2 扩展性" in meta["source_sections"]
    assert mgr._alias_registry.lookup("DAQ Unit") == qn
    print("  update: OK")

    # List entities
    entities = mgr.get_all_entities()
    assert len(entities) == 2
    print(f"  entities: {len(entities)}")

    print("  PASS\n")


def test_add_relation():
    print("=== Test add_relation ===")
    mgr = SysMLManager()

    mgr.add_entity_with_metadata("PartDef", "数据采集模块", parent_package="系统架构")
    mgr.add_entity_with_metadata("PartDef", "分析引擎", parent_package="系统架构")

    rel = mgr.add_relation(
        relation_type="connection",
        source_name="数据采集模块",
        target_name="分析引擎",
        parent_package="系统架构",
        description="数据采集模块将数据发送到分析引擎",
    )
    assert rel is not None
    assert rel.ends[0].ref == "数据采集模块"
    assert rel.ends[1].ref == "分析引擎"
    print(f"  created: {rel.qualified_name}")

    relations = mgr.get_all_relations()
    assert len(relations) == 1
    print(f"  relations: {len(relations)}")

    # Delete relation
    ok = mgr.delete_relation("数据采集模块_分析引擎_connection")
    assert ok  # 注意：关系名可能不同
    print("  delete_relation: OK")

    print("  PASS\n")


def test_merge():
    print("=== Test merge_entities ===")
    mgr = SysMLManager()

    a = mgr.add_entity_with_metadata(
        "PartDef", "数据采集模块",
        parent_package="文档A",
        description="负责采集",
        aliases=["DAQ"],
        source_sections=["文档A::第3章"],
    )
    b = mgr.add_entity_with_metadata(
        "PartDef", "DAQ模块",
        parent_package="文档B",
        description="数据采集单元",
        aliases=["数据采集模块"],
        source_sections=["文档B::第1章"],
    )

    qn_a = a.qualified_name
    qn_b = b.qualified_name

    # Suggest merges
    suggestions = mgr.suggest_merges(threshold=0.6)
    print(f"  suggestions: {len(suggestions)}")
    for s in suggestions:
        print(f"    {s['entity_a']['name']} <-> {s['entity_b']['name']}: {s['confidence']} ({s['reasons']})")

    assert len(suggestions) >= 1
    assert suggestions[0]["confidence"] >= 0.85  # alias matches name

    # Merge
    result = mgr.merge_entities(qn_b, qn_a)
    assert result == qn_a

    # Verify source is gone
    assert mgr.find_entity_by_qn(qn_b) is None

    # Verify target has merged data
    meta = mgr.get_entity_metadata(qn_a)
    assert "数据采集单元" in meta["description"]
    assert "文档A::第3章" in meta["source_sections"]
    assert "文档B::第1章" in meta["source_sections"]

    # Verify aliases transferred
    assert mgr._alias_registry.lookup("DAQ") == qn_a
    assert mgr._alias_registry.lookup("数据采集模块") == qn_a

    print(f"  merge: OK (target has {len(meta['source_sections'])} source sections)")
    print("  PASS\n")


def test_save_load_with_meta():
    print("=== Test save/load with metadata ===")
    import tempfile, os

    mgr = SysMLManager()
    mgr.add_entity_with_metadata(
        "PartDef", "测试模块",
        description="测试用",
        aliases=["TestModule"],
        source_sections=["第1章"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.sysml")
        mgr.save_to_file(path)
        print(f"  saved to: {path}")

        # Check .meta.json exists
        meta_path = path.replace(".sysml", ".meta.json")
        assert os.path.exists(meta_path)
        print(f"  meta.json exists: {os.path.exists(meta_path)}")

        # Reload
        mgr2 = SysMLManager()
        mgr2.load_from_file(path)
        entities = mgr2.get_all_entities()
        print(f"  loaded entities: {len(entities)}")
        for e in entities:
            print(f"    - {e.qualified_name} ({type(e).__name__})")
        assert len(entities) == 1
        assert entities[0].name == "测试模块"

        qn = entities[0].qualified_name
        meta = mgr2.get_entity_metadata(qn)
        assert meta["description"] == "测试用"
        assert mgr2._alias_registry.lookup("TestModule") == qn
        print(f"  reload: entity={entities[0].name}, aliases OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_alias_registry()
    test_add_entity()
    test_add_relation()
    test_merge()
    test_save_load_with_meta()
    print("All tests passed!")
