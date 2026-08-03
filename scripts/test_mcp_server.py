#!/usr/bin/env python3
"""Test sysml_rag_mcp_server.py via direct function calls + MCP stdio"""
import sys, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_direct_tools():
    """Test MCP tool functions directly (no MCP transport)"""
    print("=== Test MCP Tools (direct) ===")
    from scripts.sysml_rag_mcp_server import (
        sysml_add_entity, sysml_add_relation, sysml_search_entity,
        sysml_update_entity, sysml_delete_entity, sysml_suggest_merge,
        sysml_merge_entities, sysml_normalize_name, sysml_add_alias,
        sysml_list_entities, sysml_list_relations, sysml_model_summary,
        sysml_get_connections,
    )

    # Add entities
    r = sysml_add_entity("PartDef", "传感器节点", description="采集传感器数据的节点",
                         aliases=["SensorNode", "传感器"], source_sections=["第1章::1.1"],
                         properties={"精度": "0.01"})
    assert r["ok"], f"Add entity failed: {r}"
    qn1 = r["qualified_name"]
    print(f"  add_entity: {qn1}")

    r = sysml_add_entity("AttributeDef", "采样频率", description="传感器采样频率",
                         aliases=["SampleRate", "采样率"], parent_package="传感器节点")
    assert r["ok"], f"Add entity failed: {r}"
    qn2 = r["qualified_name"]
    print(f"  add_entity (nested): {qn2}")

    # Search
    r = sysml_search_entity("传感器")
    assert r["ok"] and r["total_matches"] > 0
    print(f"  search '传感器': {r['total_matches']} matches")

    r = sysml_search_entity("SampleRate")
    assert r["ok"] and r["total_matches"] > 0
    print(f"  search 'SampleRate' (alias): {r['total_matches']} matches")

    # Update
    r = sysml_update_entity(qn1, append_description="补充：包括温度传感器",
                            merge_aliases=["温度传感器"],
                            append_source_sections=["第2章::2.1"])
    assert r["ok"], f"Update entity failed: {r}"
    print(f"  update_entity: OK")

    # Add alias
    r = sysml_add_alias(qn1, "TemperatureSensor")
    assert r["ok"], f"Add alias failed: {r}"
    print(f"  add_alias: OK")

    # Normalize
    r = sysml_normalize_name("DAQ Module-1")
    assert r["ok"]
    print(f"  normalize 'DAQ Module-1': {r['normalized']}")

    # Add relation
    r = sysml_add_entity("PartDef", "分析引擎", description="数据分析引擎")
    assert r["ok"]
    qn3 = r["qualified_name"]

    r = sysml_add_relation("connection", "传感器节点", "分析引擎",
                          name="传感器_到_引擎", description="传感器数据流向分析引擎")
    assert r["ok"], f"Add relation failed: {r}"
    print(f"  add_relation: {r.get('qualified_name')}")

    # Get connections
    r = sysml_get_connections("传感器节点")
    assert r["ok"] and r["total_connections"] > 0
    print(f"  get_connections: {r['total_connections']} connections")

    # List
    r = sysml_list_entities()
    assert r["total"] >= 3
    print(f"  list_entities: {r['total']} total")

    r = sysml_list_relations()
    assert r["total"] >= 1
    print(f"  list_relations: {r['total']} total")

    # Suggest merge
    sysml_add_entity("PartDef", "SensorNode装置", description="传感器节点(alias variant)",
                     aliases=["传感器节点"])
    r = sysml_suggest_merge(threshold=0.5)
    print(f"  suggest_merge: {r['total_suggestions']} suggestions")
    if r['total_suggestions'] > 0:
        s = r['suggestions'][0]
        print(f"    top: {s['entity_a']['name']} <-> {s['entity_b']['name']} ({s['confidence']})")

    # Merge
    if r['total_suggestions'] > 0:
        s = r['suggestions'][0]
        r = sysml_merge_entities(s['entity_b']['qualified_name'], s['entity_a']['qualified_name'])
        assert r["ok"], f"Merge failed: {r}"
        print(f"  merge_entities: OK ({r['merged_from']} -> {r['merged_into']})")

    # Model summary
    r = sysml_model_summary()
    assert r["ok"]
    print(f"  model_summary: {r['total_entities']} entities, {r['total_relations']} relations")

    # Delete
    r = sysml_delete_entity(qn3)
    assert r["ok"], f"Delete failed: {r}"
    print(f"  delete_entity: OK")

    print("  PASS\n")


def test_mcp_stdio():
    """Test MCP stdio protocol"""
    print("=== Test MCP stdio protocol ===")
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "sysml_rag_mcp_server.py"

    proc = subprocess.Popen(
        [sys.executable, str(script_path), "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )

    def send_recv(request):
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line)

    try:
        # Initialize
        resp = send_recv({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert "protocolVersion" in resp["result"]
        print(f"  initialize: {resp['result']['serverInfo']['name']} v{resp['result']['serverInfo']['version']}")

        # List tools
        resp = send_recv({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        print(f"  tools/list: {len(tools)} tools")
        assert "sysml_add_entity" in tool_names, "Missing add_entity"
        assert "sysml_search_entity" in tool_names, "Missing search_entity"
        assert "sysml_suggest_merge" in tool_names, "Missing suggest_merge"
        assert "sysml_merge_entities" in tool_names, "Missing merge_entities"
        assert "sysml_normalize_name" in tool_names, "Missing normalize_name"
        print("  All expected tools present: OK")

        # Call normalize_name
        resp = send_recv({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "sysml_normalize_name", "arguments": {"name": "DAQ Module-1"}}
        })
        content = resp["result"]["content"][0]["text"]
        result = json.loads(content)
        assert result["ok"]
        print(f"  tools/call normalize_name: {result['normalized']}")

        # Call search_entity
        resp = send_recv({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "sysml_search_entity", "arguments": {"query": "test"}}
        })
        content = resp["result"]["content"][0]["text"]
        result = json.loads(content)
        assert result["ok"]
        print(f"  tools/call search_entity: {result['total_matches']} matches")

        print("  PASS\n")

        proc.stdin.close()
        proc.wait(timeout=5)
    except Exception as e:
        proc.kill()
        raise


if __name__ == "__main__":
    test_direct_tools()
    test_mcp_stdio()
    print("All MCP tests passed!")
