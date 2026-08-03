#!/usr/bin/env python3
"""Test Step 2: MCPSession + MCPToolWrapper + refactored MCPFetchClient"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest
pytestmark = pytest.mark.integration

async def test_mcp_session():
    """Test MCPSession with SysML MCP server"""
    print("=== Test MCPSession ===")
    from mcp_client.mcp_session import MCPSession

    cmd = f"{sys.executable} scripts/sysml_rag_mcp_server.py serve"
    session = MCPSession(server_command=cmd, server_name="test-sysml")

    # Initialize
    info = await session.initialize()
    assert info["name"], "Init failed"
    print(f"  initialize: {info['name']} v{info.get('version', '?')}")

    # List tools
    tools = await session.list_tools()
    assert len(tools) >= 20
    tool_names = [t["name"] for t in tools]
    assert "sysml_add_entity" in tool_names
    assert "sysml_search_entity" in tool_names
    assert "sysml_merge_entities" in tool_names
    print(f"  list_tools: {len(tools)} tools")

    # Call tool
    result = await session.call_tool("sysml_normalize_name", {"name": "DAQ Module-1"})
    assert result and "daqmodule1" in result.lower()
    print(f"  call_tool normalize: {result[:80]}")

    # Add entity
    result = await session.call_tool("sysml_add_entity", {
        "entity_type": "PartDef",
        "name": "MCPSession测试模块",
        "description": "MCP会话测试用实体",
        "aliases": ["TestModule"],
    })
    assert '"ok": true' in result.lower() or '"ok": True' in result
    print(f"  call_tool add_entity: OK")

    # Search
    result = await session.call_tool("sysml_search_entity", {"query": "MCPSession"})
    assert '"ok": true' in result.lower()
    print(f"  call_tool search: OK")

    # Model summary
    result = await session.call_tool("sysml_model_summary", {})
    assert '"ok": true' in result.lower()
    print(f"  call_tool model_summary: OK")

    # Cleanup
    await session.call_tool("sysml_delete_entity", {"qualified_name": "MCPSession测试模块"})

    await session.close()
    print("  close: OK")
    print("  PASS\n")


async def test_tool_wrapper():
    """Test MCPToolWrapper as LangChain tools"""
    print("=== Test MCPToolWrapper ===")
    from mcp_client.mcp_session import MCPSession
    from mcp_client.tool_wrapper import build_mcp_tools

    cmd = f"{sys.executable} scripts/sysml_rag_mcp_server.py serve"
    session = MCPSession(server_command=cmd, server_name="test-sysml-wrapper")

    # Build tools
    tool_filter = ["sysml_add_entity", "sysml_search_entity", "sysml_normalize_name"]
    tools = await build_mcp_tools(session, prefix="mcp", tool_filter=tool_filter)
    assert len(tools) == 3
    print(f"  build_mcp_tools: {len(tools)} tools")

    # Check naming
    for t in tools:
        assert t.name.startswith("mcp__"), f"Bad prefix: {t.name}"
        assert t.description, f"No description for {t.name}"
        print(f"    {t.name}: {t.description[:50]}...")

    # Check args_schema
    search_tool = next(t for t in tools if t.tool_name == "sysml_search_entity")
    schema_fields = search_tool.args_schema.model_fields
    assert "query" in schema_fields
    print(f"  search_entity schema fields: {list(schema_fields.keys())}")

    # Sync invocation (in async context, use _arun first)
    result = await search_tool._arun(query="test")
    assert result
    print(f"  _arun: OK")

    # Add entity via tool
    add_tool = next(t for t in tools if t.tool_name == "sysml_add_entity")
    result = await add_tool._arun(
        entity_type="PartDef",
        name="WrapperTestModule",
        description="Tool wrapper test",
    )
    assert '"ok": true' in result.lower()
    print(f"  add entity via _arun: OK")

    # Search for it
    result = await search_tool._arun(query="WrapperTestModule")
    assert '"ok": true' in result.lower()
    print(f"  search after add: OK")

    # Cleanup
    await session.call_tool("sysml_delete_entity", {"qualified_name": "WrapperTestModule"})

    await session.close()
    print("  PASS\n")


async def test_mcp_fetch_client_structure():
    """Test refactored MCPFetchClient instantiation (no actual server needed)"""
    print("=== Test MCPFetchClient refactoring ===")
    from mcp_client.client import MCPFetchClient, get_mcp_client

    client = get_mcp_client()
    assert isinstance(client, MCPFetchClient)
    print("  get_mcp_client: OK")

    assert hasattr(client, '_fallback_mode')
    print("  structure check: OK")
    print("  PASS\n")


async def main():
    await test_mcp_session()
    await test_tool_wrapper()
    await test_mcp_fetch_client_structure()
    print("All Step 2 tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
