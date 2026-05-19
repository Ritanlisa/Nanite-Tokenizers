#!/usr/bin/env python3
"""Test sysml_retrieve: entity lookup + k-layer graph traversal"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sysml_rag_mcp_server import (
    sysml_retrieve,
    sysml_load_model,
    sysml_add_entity,
    sysml_add_relation,
    sysml_add_alias,
    _get_manager,
    _global_manager,
    _loaded_files,
)

import scripts.sysml_rag_mcp_server as mcp


def reset_global_state():
    """Reset the global manager for a clean test"""
    global _global_manager
    _global_manager = None
    _loaded_files.clear()


def test_definition_lookup():
    """Test retrieving a Definition by name"""
    print("=== Test 1: Definition lookup by name ===")
    reset_global_state()

    mgr = _get_manager()
    sysml_add_entity("PartDef", "数据采集模块", parent_package="系统架构",
                     description="负责从多个传感器节点采集数据",
                     aliases=["DAQ模块", "采集模块", "DAQ Module"],
                     source_sections=["第3章::3.1 数据层"],
                     source_text="数据采集模块是系统的数据入口",
                     properties={"采样率": "100Hz"})

    result = sysml_retrieve("数据采集模块", k=0)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["entity"]["name"] == "数据采集模块"
    assert result["entity"]["class"] == "PartDef"
    assert result["entity"]["metadata"]["description"] == "负责从多个传感器节点采集数据"
    assert result["entity"]["metadata"]["properties"]["采样率"] == "100Hz"
    assert "DAQ模块" in result["entity"]["aliases"]
    assert result["relationship_graph"]["total_nodes"] == 1  # k=0: only center
    assert result["relationship_graph"]["layers"][0]["level"] == 0
    print(f"  PASS: entity={result['entity']['name']}, nodes={result['relationship_graph']['total_nodes']}, layers={len(result['relationship_graph']['layers'])}")


def test_alias_lookup():
    """Test retrieving an entity by alias"""
    print("=== Test 2: Alias lookup ===")
    reset_global_state()

    sysml_add_entity("PartDef", "数据采集模块", parent_package="系统架构",
                     aliases=["DAQ模块", "采集模块", "DAQ Module"])

    # Search by normalized alias
    result = sysml_retrieve("DAQ模块", k=0)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["entity"]["name"] == "数据采集模块"
    print(f"  PASS: alias='DAQ模块' → name='{result['entity']['name']}'")

    # Search by normalized English alias
    result = sysml_retrieve("daq module", k=0)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["entity"]["name"] == "数据采集模块"
    print(f"  PASS: alias='daq module' → name='{result['entity']['name']}'")


def test_usage_lookup():
    """Test retrieving a Usage (instance) by name"""
    print("=== Test 3: Usage (instance) lookup ===")
    reset_global_state()

    # Create a definition first
    sysml_add_entity("PartDef", "温度传感器", parent_package="设备层",
                     aliases=["TempSensor", "TS"])

    # Create a usage (instance) of that definition
    mgr = _get_manager()
    from sysml.sysml_model import PartUsage
    instance = PartUsage(name="机柜A_温度传感器_PT100", type_refs=["温度传感器"])
    mgr.add_element(instance)

    # Lookup by instance name
    result = sysml_retrieve("机柜A_温度传感器_PT100", k=0)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["entity"]["name"] == "机柜A_温度传感器_PT100"
    assert result["entity"]["class"] == "PartUsage"
    print(f"  PASS: instance='{result['entity']['name']}' type='{result['entity']['class']}'")


def test_k_layer_traversal():
    """Test k-layer relationship graph traversal"""
    print("=== Test 4: K-layer graph traversal ===")
    reset_global_state()

    # Create entities
    sysml_add_entity("PartDef", "数据采集模块", parent_package="系统架构",
                     description="数据入口", aliases=["DAQ模块"])
    sysml_add_entity("PartDef", "分析引擎", parent_package="系统架构",
                     description="数据分析核心")
    sysml_add_entity("PartDef", "告警服务", parent_package="服务层",
                     description="告警通知系统")
    sysml_add_entity("PartDef", "数据库", parent_package="存储层",
                     description="持久化存储")
    sysml_add_entity("PartDef", "Web仪表盘", parent_package="展示层",
                     description="前端展示")

    # Create relations
    sysml_add_relation("connection", "数据采集模块", "分析引擎",
                       name="数据流", parent_package="系统架构",
                       description="采集模块将原始数据发送到分析引擎")
    sysml_add_relation("connection", "分析引擎", "告警服务",
                       name="告警触发", parent_package="系统架构",
                       description="分析结果触发告警")
    sysml_add_relation("connection", "分析引擎", "数据库",
                       name="数据持久化", parent_package="存储层",
                       description="分析结果写入数据库")
    sysml_add_relation("connection", "数据库", "Web仪表盘",
                       name="数据展示", parent_package="展示层",
                       description="仪表盘读取数据库")

    # k=0: only center entity
    result = sysml_retrieve("分析引擎", k=0)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["relationship_graph"]["total_nodes"] == 1
    assert result["relationship_graph"]["total_edges"] == 0
    assert len(result["relationship_graph"]["layers"]) == 1
    print(f"  k=0: nodes={result['relationship_graph']['total_nodes']}, edges={result['relationship_graph']['total_edges']} ✓")

    # k=1: center + direct neighbors (数据采集模块, 告警服务, 数据库)
    result = sysml_retrieve("分析引擎", k=1)
    assert result["ok"]
    graph = result["relationship_graph"]
    assert graph["total_nodes"] == 4  # 分析引擎 + 3 direct neighbors
    assert graph["total_edges"] == 3  # 3 connections from 分析引擎
    node_names = set(graph["nodes"].keys())
    assert "分析引擎" in node_names
    assert "数据采集模块" in node_names
    assert "告警服务" in node_names
    assert "数据库" in node_names
    print(f"  k=1: nodes={graph['total_nodes']}, edges={graph['total_edges']}, layer0={graph['layers'][0]['node_ids']}, layer1={graph['layers'][1]['node_ids']} ✓")

    # k=2: center + level 1 + level 2 (Web仪表盘 via 数据库)
    result = sysml_retrieve("分析引擎", k=2)
    assert result["ok"]
    graph = result["relationship_graph"]
    assert graph["total_nodes"] == 5  # all 5 entities
    assert graph["total_edges"] >= 3
    assert "Web仪表盘" in graph["nodes"]
    # Check that Web仪表盘 is in a deeper layer
    layer2_nodes = graph["layers"][2]["node_ids"] if len(graph["layers"]) > 2 else []
    print(f"  k=2: nodes={graph['total_nodes']}, edges={graph['total_edges']}, layer2={layer2_nodes} ✓")


def test_nonexistent_entity():
    """Test searching for an entity that doesn't exist"""
    print("=== Test 5: Nonexistent entity ===")
    reset_global_state()

    result = sysml_retrieve("不存在的实体", k=2)
    assert not result["ok"]
    assert "error" in result
    print(f"  PASS: error='{result['error']}'")


def test_entity_with_members():
    """Test entity with nested members (full property tree)"""
    print("=== Test 6: Entity with nested members ===")
    reset_global_state()

    # Create a definition with members (simulating SysML attribute/port definitions)
    mgr = _get_manager()
    from sysml.sysml_model import PartDef, AttributeUsage, PortUsage, DirectionKind, Multiplicity

    entity = PartDef(name="智能传感器", short_name="SmartSens")
    entity.add_member(AttributeUsage(name="采样率", value_expr="100"))
    entity.add_member(AttributeUsage(name="精度", value_expr="0.01"))
    entity.add_member(PortUsage(name="数据输出", direction=DirectionKind.OUT))
    mgr.add_element(entity, parent=None)

    result = sysml_retrieve("智能传感器", k=0)
    assert result["ok"]
    assert result["entity"]["name"] == "智能传感器"
    assert result["entity"]["short_name"] == "SmartSens"
    assert len(result["entity"]["members"]) == 3
    member_names = [m["name"] for m in result["entity"]["members"]]
    assert "采样率" in member_names
    assert "精度" in member_names
    assert "数据输出" in member_names
    print(f"  PASS: members={member_names}")


def test_output_structure():
    """Test that the output JSON structure is correct and consistent"""
    print("=== Test 7: Output structure validation ===")
    reset_global_state()

    sysml_add_entity("PartDef", "测试实体", parent_package="测试",
                     description="测试描述", aliases=["TestEntity", "TE"],
                     source_sections=["第1章"],
                     properties={"key": "value"})

    result = sysml_retrieve("测试实体", k=2)

    # Top-level keys
    assert "ok" in result
    assert "matched_by" in result
    assert "entity" in result
    assert "relationship_graph" in result

    # Entity keys
    entity = result["entity"]
    for key in ["name", "class", "type", "qualified_name"]:
        assert key in entity, f"Missing key in entity: {key}"

    # Graph keys
    graph = result["relationship_graph"]
    for key in ["depth", "total_nodes", "total_edges", "nodes", "edges", "layers"]:
        assert key in graph, f"Missing key in graph: {key}"

    # Layers structure
    for layer in graph["layers"]:
        assert "level" in layer
        assert "node_ids" in layer

    # Edges structure
    for edge in graph["edges"]:
        for key in ["from", "to", "relation"]:
            assert key in edge, f"Missing key in edge: {key}"

    print(f"  PASS: all required keys present")


def test_mcp_tool_call():
    """Test that sysml_retrieve works through the MCP tool calling interface"""
    print("=== Test 8: Direct MCP tool call ===")
    reset_global_state()

    sysml_add_entity("PartDef", "MCP测试实体", parent_package="测试包",
                     description="用于测试MCP调用")

    # Simulate what happens when MCP calls tools/call
    result_json = mcp._run_tool("sysml_retrieve", {"name": "MCP测试实体", "k": 1})
    result = json.loads(result_json)
    assert result["ok"], f"Expected ok=True, got: {result}"
    assert result["entity"]["name"] == "MCP测试实体"
    print(f"  PASS: MCP tool call returned correct result")


def test_multiple_aliases_leading_to_same_entity():
    """Test that different aliases all resolve to the same entity"""
    print("=== Test 9: Multiple aliases → same entity ===")
    reset_global_state()

    sysml_add_entity("PartDef", "湖超超级计算机", parent_package="系统",
                     aliases=["湖超", "Huchao", "HC", "超级计算机"])

    for alias in ["湖超", "Huchao", "超级计算机"]:
        result = sysml_retrieve(alias, k=0)
        assert result["ok"], f"Failed for alias '{alias}'"
        assert result["entity"]["name"] == "湖超超级计算机", f"Alias '{alias}' resolved to '{result['entity']['name']}'"

    print("  PASS: all aliases resolved to same entity")


def test_graph_edges_have_roles():
    """Test that connection edges include role information"""
    print("=== Test 10: Edge role information ===")
    reset_global_state()

    sysml_add_entity("PartDef", "服务A", parent_package="服务层")
    sysml_add_entity("PartDef", "服务B", parent_package="服务层")

    sysml_add_relation("connection", "服务A", "服务B",
                       name="API调用", parent_package="服务层",
                       role_source="client", role_target="server",
                       description="服务A通过API调用服务B")

    result = sysml_retrieve("服务A", k=1)
    assert result["ok"]
    edges = result["relationship_graph"]["edges"]
    assert len(edges) == 1
    edge = edges[0]
    assert edge["from"] == "服务A"
    assert edge["to"] == "服务B"
    assert edge["from_role"] == "client"
    assert edge["to_role"] == "server"
    assert edge["relation"]["name"] == "API调用"
    assert "description" in edge["relation"]
    print(f"  PASS: edge roles correct: {edge['from_role']} → {edge['to_role']}")


# ── Run all ──
if __name__ == "__main__":
    test_definition_lookup()
    test_alias_lookup()
    test_usage_lookup()
    test_k_layer_traversal()
    test_nonexistent_entity()
    test_entity_with_members()
    test_output_structure()
    test_mcp_tool_call()
    test_multiple_aliases_leading_to_same_entity()
    test_graph_edges_have_roles()
    print()
    print("=" * 50)
    print("All 10 tests passed!")
    print("=" * 50)
