from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from contextlib import AsyncExitStack
from typing import Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from exceptions import MCPConnectionError, MCPFatalError, MCPTimeoutError
from monitoring import mcp_restart_count

logger = logging.getLogger(__name__)


class MCPFetchClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._lock = asyncio.Lock()
            cls._instance._configured = False
        return cls._instance

    def __init__(self) -> None:
        if self._configured:
            return
        self._restart_count = 0
        self._max_restart = config.settings.MCP_MAX_RESTART
        self._fallback_mode = False
        self._configured = True

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            command = config.settings.MCP_FETCH_SERVER_COMMAND.strip()
            if not command or command.lower() == "disabled":
                self._enable_fallback("fetch server disabled")
                return
            self.exit_stack = AsyncExitStack()
            parts = shlex.split(command)
            if parts and parts[0] in {"python", "python3"}:
                parts[0] = sys.executable
            server_params = StdioServerParameters(command=parts[0], args=parts[1:])
            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await self.session.initialize()
                self._initialized = True
                self._restart_count = 0
                logger.info("MCP fetch server connected")
            except Exception as exc:
                logger.warning("MCP init failed (%s), falling back to direct HTTP", exc)
                await self.exit_stack.aclose()
                self._enable_fallback("mcp init failed")

    async def fetch(self, url: str, timeout: Optional[float] = None) -> str:
        if not self._initialized:
            await self.initialize()
        timeout = timeout or config.settings.MCP_TIMEOUT
        if self._fallback_mode:
            return await self._fetch_http(url, timeout)
        try:
            return await self._fetch_internal(url, timeout)
        except asyncio.TimeoutError:
            logger.warning("MCP fetch timeout: %s", url)
            raise MCPTimeoutError("Fetch timed out")
        except (ConnectionError, BrokenPipeError, OSError) as exc:
            if self._restart_count >= self._max_restart:
                raise MCPFatalError(
                    "MCP service unavailable: restart limit exceeded"
                ) from exc
            self._restart_count += 1
            mcp_restart_count.inc()
            logger.warning(
                "MCP connection dropped, restarting (%s/%s)",
                self._restart_count,
                self._max_restart,
            )
            await asyncio.sleep(config.settings.MCP_RETRY_DELAY)
            await self._reinitialize()
            return await self._fetch_internal(url, timeout)
        except Exception as exc:
            logger.exception("MCP fetch failed: %s", url)
            await self._reinitialize()
            raise MCPConnectionError(f"Fetch failed: {type(exc).__name__}") from exc

    async def _fetch_internal(self, url: str, timeout: float) -> str:
        result = await asyncio.wait_for(
            self.session.call_tool("fetch", arguments={"url": url}),
            timeout=timeout,
        )
        self._restart_count = 0
        content = result.content
        if isinstance(content, list) and content:
            return getattr(content[0], "text", "")
        return str(content)

    async def _reinitialize(self) -> None:
        try:
            await self.close()
        except Exception:
            pass
        await self.initialize()

    async def _fetch_http(self, url: str, timeout: float) -> str:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def _enable_fallback(self, reason: str) -> None:
        self._fallback_mode = True
        self._initialized = True
        logger.info("MCP fetch fallback enabled: %s", reason)

    async def close(self) -> None:
        if not hasattr(self, "exit_stack"):
            self._initialized = False
            return

        try:
            await self.exit_stack.aclose()
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                logger.debug("Skipping MCP close on closed event loop")
            else:
                raise
        finally:
            self._initialized = False
            logger.info("MCP client closed")


_mcp_client = MCPFetchClient()


def get_mcp_client() -> MCPFetchClient:
    return _mcp_client
