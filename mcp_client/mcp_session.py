"""
通用 MCP stdio 会话管理器
支持连接任意 MCP 服务器并调用工具，可复用于多种 MCP 服务。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import httpx
# Embedded mode - no subprocess needed

import config
from exceptions import MCPConnectionError, MCPFatalError, MCPTimeoutError
from monitoring import mcp_restart_count

logger = logging.getLogger(__name__)


class MCPSession:
    """通用 MCP stdio 会话，可连接任意 MCP 服务器"""

    def __init__(
        self,
        server_command: str = "",
        server_name: str = "",
        max_restart: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        self._server_name = server_name or "sysml-rag"
        self._max_restart = max_restart if max_restart is not None else config.settings.MCP_MAX_RESTART
        self._retry_delay = retry_delay if retry_delay is not None else config.settings.MCP_RETRY_DELAY
        self._timeout = timeout or config.settings.MCP_TIMEOUT
        self._initialized = False
        self._restart_count = 0
        self._session = None
        self._lock = asyncio.Lock()
        self._tool_cache: Optional[List[Dict[str, Any]]] = None
        
        # Import and initialize embedded server
        from scripts.sysml_rag_mcp_server import (
            TOOL_DEFINITIONS, _run_tool, _get_manager,
            sysml_model_summary,
        )
        self._tool_defs = TOOL_DEFINITIONS
        self._run_tool = _run_tool
        self._summary_fn = sysml_model_summary
        _get_manager()  # ensure manager is initialized

    async def initialize(self) -> Dict[str, Any]:
        async with self._lock:
            if self._initialized:
                return self._server_info
            try:
                self._server_info = {
                    "name": self._server_name,
                    "version": "embedded",
                    "protocolVersion": "2024-11-05",
                }
                self._initialized = True
                self._restart_count = 0
                logger.info("Embedded MCP server [%s] ready", self._server_name)
                return self._server_info
            except Exception as exc:
                raise MCPConnectionError(
                    f"Embedded MCP init failed: {exc}"
                ) from exc

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tool_cache is not None:
            return self._tool_cache
        if not self._initialized:
            await self.initialize()
        schemas = []
        for name, defn in self._tool_defs.items():
            properties = {}
            required = []
            for k, v in defn["parameters"].items():
                entry = {"type": v["type"], "description": v.get("description", "")}
                if "default" in v:
                    entry["default"] = v["default"]
                else:
                    required.append(k)
                if v.get("items"):
                    entry["items"] = v["items"]
                properties[k] = entry
            schemas.append({
                "name": name,
                "description": defn["description"],
                "inputSchema": {"type": "object", "properties": properties, "required": required},
            })
        self._tool_cache = schemas
        return schemas

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._initialized:
            await self.initialize()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_tool, name, arguments),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Embedded MCP call timed out: {name}")
        except Exception as exc:
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
        if False:
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
