from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
from exceptions import MCPFatalError
from mcp.client import get_mcp_client
from rag.engine import RAGEngine

logger = logging.getLogger(__name__)

rag_engine = RAGEngine()


class RAGSearchInput(BaseModel):
    query: str = Field(description="Search query")


class RAGSearchTool(BaseTool):
    name = "rag_search"
    description = "Search internal knowledge base."
    args_schema = RAGSearchInput

    async def _arun(self, query: str) -> str:
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, rag_engine.query, query),
                timeout=30,
            )
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            if sources:
                return answer
            return answer
        except asyncio.TimeoutError:
            return "Knowledge base query timed out."
        except Exception as exc:
            logger.exception("rag_search failed")
            if config.settings.ENV == "prod":
                return "RAG search failed."
            return f"RAG search failed: {str(exc)[:200]}"

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async call")


class FetchWebpageInput(BaseModel):
    url: str = Field(description="Webpage URL to fetch")


class FetchWebpageTool(BaseTool):
    name = "fetch_webpage"
    description = "Fetch the content of a webpage URL."
    args_schema = FetchWebpageInput

    @retry(
        stop=stop_after_attempt(config.settings.MCP_RETRY_TIMES),
        wait=wait_exponential(multiplier=config.settings.MCP_RETRY_DELAY),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
    )
    async def _arun(self, url: str) -> str:
        client = get_mcp_client()
        try:
            return await client.fetch(url)
        except MCPFatalError:
            return "MCP service unavailable."
        except Exception as exc:
            logger.exception("fetch_webpage failed: %s", url)
            if config.settings.ENV == "prod":
                return "Fetch failed."
            return f"Fetch failed: {type(exc).__name__}"

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async call")


tools = [RAGSearchTool(), FetchWebpageTool()]
