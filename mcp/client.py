from __future__ import annotations

import atexit
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

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
        self._configured = True

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            self.exit_stack = AsyncExitStack()
            parts = config.settings.MCP_FETCH_SERVER_COMMAND.split()
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
                logger.error("MCP init failed: %s", exc)
                await self.exit_stack.aclose()
                raise

    async def fetch(self, url: str, timeout: Optional[float] = None) -> str:
        if not self._initialized:
            await self.initialize()
        timeout = timeout or config.settings.MCP_TIMEOUT
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
            return content[0].text
        return str(content)

    async def _reinitialize(self) -> None:
        try:
            await self.close()
        except Exception:
            pass
        await self.initialize()

    async def close(self) -> None:
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()
            self._initialized = False
            logger.info("MCP client closed")


_mcp_client = MCPFetchClient()


def get_mcp_client() -> MCPFetchClient:
    return _mcp_client


def _cleanup() -> None:
    try:
        asyncio.run(_mcp_client.close())
    except RuntimeError:
        pass


atexit.register(_cleanup)
