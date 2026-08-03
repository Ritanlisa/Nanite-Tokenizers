from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

import config
from exceptions import MCPConnectionError, MCPTimeoutError
from .mcp_session import MCPSession

logger = logging.getLogger(__name__)


class MCPFetchClient:
    """MCP Fetch 客户端（基于 MCPSession），用于网页内容抓取"""
    _instance: Optional["MCPFetchClient"] = None

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
        self._fallback_mode = False
        self._session: Optional[MCPSession] = None
        self._configured = True

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            command = config.settings.MCP_FETCH_SERVER_COMMAND.strip()
            if not command or command.lower() == "disabled":
                self._enable_fallback("fetch server disabled")
                return
            try:
                self._session = MCPSession(
                    server_command=command,
                    server_name="mcp-fetch",
                    max_restart=config.settings.MCP_MAX_RESTART,
                    retry_delay=config.settings.MCP_RETRY_DELAY,
                    timeout=config.settings.MCP_TIMEOUT,
                )
                await self._session.initialize()
                self._initialized = True
                logger.info("MCP fetch server connected")
            except Exception as exc:
                logger.warning("MCP init failed (%s), falling back to direct HTTP", exc)
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
            await self._reinitialize()
            raise MCPConnectionError(f"Fetch failed: {type(exc).__name__}") from exc
        except Exception as exc:
            logger.exception("MCP fetch failed: %s", url)
            raise MCPConnectionError(f"Fetch failed: {type(exc).__name__}") from exc

    async def _fetch_internal(self, url: str, timeout: float) -> str:
        if self._session is None:
            raise MCPConnectionError("Fetch session not initialized")
        result = await asyncio.wait_for(
            self._session.call_tool("fetch", {"url": url}),
            timeout=timeout,
        )
        return result

    async def _fetch_http(self, url: str, timeout: float) -> str:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def _enable_fallback(self, reason: str) -> None:
        self._fallback_mode = True
        self._initialized = True
        logger.info("MCP fetch fallback enabled: %s", reason)

    async def _reinitialize(self) -> None:
        try:
            await self.close()
        except Exception:
            pass
        await self.initialize()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._initialized = False
        logger.info("MCP client closed")


_mcp_client = MCPFetchClient()


def get_mcp_client() -> MCPFetchClient:
    return _mcp_client
