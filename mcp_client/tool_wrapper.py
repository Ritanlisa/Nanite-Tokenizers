"""
MCP 工具 → LangChain BaseTool 包装器
将 MCP 协议工具包装为 LangChain 可调用的工具，支持动态参数 Schema 生成。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from .mcp_session import MCPSession
import tool_approval

logger = logging.getLogger(__name__)

# MCP JSON Schema → Python 类型映射
_TYPE_MAP: Dict[str, Type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_model(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """从 MCP tool inputSchema 动态生成 Pydantic 模型"""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: Dict[str, Any] = {}
    for prop_name, prop_info in properties.items():
        mcp_type = prop_info.get("type", "string")
        python_type = _TYPE_MAP.get(mcp_type, str)
        description = prop_info.get("description", "")
        is_required = prop_name in required

        if prop_info.get("items"):
            python_type = list

        if is_required:
            fields[prop_name] = (python_type, Field(description=description))
        else:
            fields[prop_name] = (Optional[python_type], Field(default=None, description=description))

    safe_name = "".join(c if c.isalnum() else "_" for c in tool_name)
    model_name = f"MCP_{safe_name}_Args"

    if not fields:
        fields["dummy"] = (Optional[bool], Field(default=True, description="No arguments needed"))

    return create_model(model_name, **fields)  # type: ignore[call-overload]


class MCPToolWrapper(BaseTool):
    """将单个 MCP 工具包装为 LangChain BaseTool"""

    session: Any = Field(default=None, exclude=True)
    tool_name: str = Field(default="", exclude=True)
    tool_description: str = Field(default="", exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, session: MCPSession, tool_schema: Dict[str, Any], prefix: str = ""):
        tool_name = tool_schema["name"]
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})
        args_model = _build_args_model(tool_name, input_schema)

        display_name = f"{prefix}{tool_name}" if prefix else tool_name
        if prefix and not prefix.endswith("__"):
            display_name = f"{prefix}__{tool_name}"

        super().__init__(
            name=display_name,
            description=description,
            args_schema=args_model,
            session=session,
            tool_name=tool_name,
            tool_description=description,
        )

    async def _check_approval(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Check if tool execution is approved. Returns None if approved, or rejection message."""
        from tool_usage import get_current_session_id
        session_id = get_current_session_id()
        if not session_id:
            return None
        call_id = str(uuid.uuid4())
        cleaned = self._clean_kwargs(kwargs)
        approved = await tool_approval.request_tool_approval(
            session_id, call_id, self.tool_name, cleaned,
        )
        if not approved:
            return f"[Tool execution rejected by user: {self.tool_name}]"
        return None

    def _run(self, **kwargs: Any) -> str:
        """同步执行"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            rejected = asyncio.run(self._check_approval(kwargs))
            if rejected:
                return rejected
            return asyncio.run(self.session.call_tool(self.tool_name, self._clean_kwargs(kwargs)))

        rejected = asyncio.run_coroutine_threadsafe(
            self._check_approval(kwargs), loop,
        ).result(timeout=30)
        if rejected:
            return rejected

        if self.session._timeout:
            coro = asyncio.wait_for(
                self.session.call_tool(self.tool_name, self._clean_kwargs(kwargs)),
                timeout=self.session._timeout,
            )
        else:
            coro = self.session.call_tool(self.tool_name, self._clean_kwargs(kwargs))

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=(self.session._timeout or 30) + 5)

    async def _arun(self, **kwargs: Any) -> str:
        """异步执行"""
        rejected = await self._check_approval(kwargs)
        if rejected:
            return rejected
        return await self.session.call_tool(self.tool_name, self._clean_kwargs(kwargs))

    @staticmethod
    def _clean_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """清理 None 值和内部参数"""
        return {k: v for k, v in kwargs.items() if v is not None and not k.startswith("_")}


async def build_mcp_tools(
    session: MCPSession, prefix: str = "", tool_filter: Optional[List[str]] = None
) -> List[MCPToolWrapper]:
    """从 MCP 会话获取所有工具并包装为 LangChain BaseTool 列表"""
    await session.initialize()
    tool_schemas = await session.list_tools()
    tools: List[MCPToolWrapper] = []
    for schema in tool_schemas:
        name = schema.get("name", "")
        if tool_filter and name not in tool_filter:
            continue
        tools.append(MCPToolWrapper(session, schema, prefix=prefix))
    logger.info("Built %d MCP tools from [%s]", len(tools), session.server_name)
    return tools
