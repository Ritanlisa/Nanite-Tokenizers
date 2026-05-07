"""
通用 MCP stdio 会话管理器
支持连接任意 MCP 服务器并调用工具，可复用于多种 MCP 服务。
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from exceptions import MCPConnectionError, MCPFatalError, MCPTimeoutError
from monitoring import mcp_restart_count

logger = logging.getLogger(__name__)


class MCPSession:
    """通用 MCP stdio 会话，可连接任意 MCP 服务器"""

    def __init__(
        self,
        server_command: str,
        server_name: str = "",
        max_restart: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        parts = shlex.split(server_command.strip())
        if parts and parts[0] in {"python", "python3"}:
            parts[0] = sys.executable
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        self._server_params = StdioServerParameters(command=command, args=args)
        self._server_name = server_name or command
        self._max_restart = max_restart if max_restart is not None else config.settings.MCP_MAX_RESTART
        self._retry_delay = retry_delay if retry_delay is not None else config.settings.MCP_RETRY_DELAY
        self._timeout = timeout or config.settings.MCP_TIMEOUT
        self._initialized = False
        self._restart_count = 0
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()
        self._tool_cache: Optional[List[Dict[str, Any]]] = None

    async def initialize(self) -> Dict[str, Any]:
        async with self._lock:
            if self._initialized:
                return self._server_info
            self._exit_stack = AsyncExitStack()
            try:
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(self._server_params)
                )
                read, write = stdio_transport
                self._session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                init_result = await self._session.initialize()
                server_info = getattr(init_result, "serverInfo", None)
                self._server_info = {
                    "name": getattr(server_info, "name", self._server_name),
                    "version": getattr(server_info, "version", ""),
                    "protocolVersion": getattr(init_result, "protocolVersion", ""),
                }
                self._initialized = True
                self._restart_count = 0
                logger.info("MCP session [%s] connected", self._server_name)
                return self._server_info
            except Exception as exc:
                logger.error("MCP session [%s] init failed: %s", self._server_name, exc)
                if self._exit_stack is not None:
                    await self._exit_stack.aclose()
                raise MCPConnectionError(
                    f"MCP session [{self._server_name}] init failed: {exc}"
                ) from exc

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tool_cache is not None:
            return self._tool_cache
        if not self._initialized:
            await self.initialize()
        if self._session is None:
            raise MCPConnectionError(f"MCP session [{self._server_name}] not initialized")
        try:
            result = await self._session.list_tools()
            tools = getattr(result, "tools", [])
            schemas = []
            for t in tools:
                schemas.append({
                    "name": getattr(t, "name", ""),
                    "description": getattr(t, "description", ""),
                    "inputSchema": getattr(t, "inputSchema", {}),
                })
            self._tool_cache = schemas
            return schemas
        except (ConnectionError, BrokenPipeError, OSError) as exc:
            raise self._handle_connection_error(exc, "list_tools")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._initialized:
            await self.initialize()
        if self._session is None:
            raise MCPConnectionError(f"MCP session [{self._server_name}] not initialized")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self._timeout,
            )
            content = getattr(result, "content", [])
            if isinstance(content, list) and content:
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        return text
            return str(content)
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"MCP call [{self._server_name}] timed out: {name}")
        except (ConnectionError, BrokenPipeError, OSError) as exc:
            raise self._handle_connection_error(exc, name)

    def _handle_connection_error(self, exc: Exception, method: str) -> Exception:
        if self._restart_count >= self._max_restart:
            return MCPFatalError(
                f"MCP [{self._server_name}] unavailable: restart limit exceeded"
            )
        self._restart_count += 1
        mcp_restart_count.inc()
        logger.warning(
            "MCP [%s] connection dropped, restarting (%s/%s)",
            self._server_name, self._restart_count, self._max_restart,
        )
        return MCPConnectionError(f"MCP [{self._server_name}] {method}: {type(exc).__name__}")

    async def close(self) -> None:
        self._tool_cache = None
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except (RuntimeError, Exception) as exc:
                if "Event loop is closed" in str(exc) or "cancel scope" in str(exc).lower():
                    logger.debug("MCP close minor error [%s]: %s", self._server_name, exc)
                else:
                    raise
            finally:
                self._session = None
                self._exit_stack = None
                self._initialized = False
                logger.info("MCP session [%s] closed", self._server_name)
        else:
            self._initialized = False

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def is_connected(self) -> bool:
        return self._initialized and self._session is not None


# ── 为多种 MCP 服务提供工厂函数 ──

def create_sysml_mcp_session() -> MCPSession:
    """创建 SysML MCP 会话"""
    command = getattr(config.settings, "KG_MCP_SERVER_COMMAND", None)
    if not command:
        command = f"{sys.executable} scripts/sysml_rag_mcp_server.py serve"
    return MCPSession(server_command=command, server_name="sysml-rag")


def create_fetch_mcp_session() -> MCPSession:
    """创建 Fetch MCP 会话"""
    command = config.settings.MCP_FETCH_SERVER_COMMAND.strip()
    if not command or command.lower() == "disabled":
        command = "python -m mcp_server_fetch"
    return MCPSession(
        server_command=command,
        server_name="mcp-fetch",
        max_restart=config.settings.MCP_MAX_RESTART,
        retry_delay=config.settings.MCP_RETRY_DELAY,
        timeout=config.settings.MCP_TIMEOUT,
    )
