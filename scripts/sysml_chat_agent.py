"""
SysML Chat Agent (Search-then-answer)
—— Pre-fetches sections, enriches with live values, then asks LLM to answer.
"""

from __future__ import annotations

import asyncio, logging, json
from pathlib import Path
from typing import Optional, List

import config

# Import ChatOpenAIWithReasoning directly, bypassing agent/__init__.py
from importlib import util as _util
_chat_file = Path(__file__).resolve().parent.parent / "agent" / "chatOpenAIWithReasoning.py"
_spec = _util.spec_from_file_location("chatOpenAIWithReasoning", str(_chat_file))
_chat_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_chat_mod)
ChatOpenAIWithReasoning = _chat_mod.ChatOpenAIWithReasoning

from mcp_client.mcp_session import create_sysml_mcp_session, MCPSession
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a hardware maintenance assistant for the Huchao supercomputer.
Answer questions based on the document sections provided.
If live values (temperature, fan speed, etc.) are available, include them.
If the information is insufficient, say so clearly.

Be concise and accurate."""


class SysMLChatAgent:

    def __init__(self, kg_file: Optional[str] = None):
        if kg_file is None:
            p = Path(__file__).resolve().parent.parent / "database" / "AIOPS" / "knowledge_graph.sysml"
            if p.exists():
                kg_file = str(p)
        self._kg_file = kg_file
        self._session: Optional[MCPSession] = None
        self._llm = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        self._session = create_sysml_mcp_session()
        await self._session.initialize()

        try:
            if self._kg_file:
                r = await self._session.call_tool("sysml_load_model", {"file_path": self._kg_file})
                logger.info(f"KG: {r[:80]}")
        except Exception as e:
            logger.warning(f"KG load: {e}")

        api_key = config.settings.OPENAI_API_KEY
        self._llm = ChatOpenAIWithReasoning(
            model=config.settings.LLM_MODEL,
            temperature=0.1,
            api_key=SecretStr(api_key) if api_key else None,
            base_url=config.settings.OPENAI_API_URL,
            timeout=120,
            streaming=False,
        )

        # Warm-up
        try:
            await asyncio.wait_for(
                self._llm.ainvoke([SystemMessage(content="OK"), HumanMessage(content="ok")]),
                timeout=30,
            )
        except Exception:
            pass

        self._initialized = True
        logger.info("SysML Chat Agent ready")

    async def _call_mcp(self, tool: str, args: dict) -> str:
        if not self._session:
            return ""
        try:
            return await self._session.call_tool(tool, args)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        if not self._initialized:
            await self.initialize()

        # Step 1: Search for relevant sections
        search_result = await self._call_mcp("sysml_search_sections", {"query": message, "max_results": 5})
        try:
            search_data = json.loads(search_result)
            sections = search_data.get("sections", [])
        except Exception:
            sections = []

        # Step 2: Get enriched content for top 2 sections
        context_parts = []
        for sec in sections[:2]:
            sec_id = sec.get("id", "")
            if not sec_id:
                continue
            sec_result = await self._call_mcp("sysml_get_section", {"section_id": sec_id, "enrich": True})
            try:
                sec_data = json.loads(sec_result)
                enriched = sec_data.get("enriched_text", sec_data.get("raw_text", ""))
                if enriched:
                    context_parts.append(f"[{sec.get('title', sec_id)}]\n{enriched}")
            except Exception:
                pass

        # Step 3: Resolve HVs mentioned in the user query
        hv_map = {
            "温度": "temperature", "风扇": "fan_speed", "转速": "fan_speed",
            "湿度": "humidity", "电压": "voltage_level", "气压": "pressure",
            "功率": "power_load", "电池": "battery_capacity",
        }
        hv_values = []
        for keyword, hv_id in hv_map.items():
            if keyword in message:
                hv_result = await self._call_mcp("sysml_resolve_hv", {"hv_id": hv_id})
                try:
                    hv_data = json.loads(hv_result)
                    if hv_data.get("ok"):
                        hv_values.append(f"{hv_id}: {hv_data.get('value', '?')}")
                except Exception:
                    pass

        # Step 4: Build context and ask LLM
        context = "\n\n".join(context_parts) if context_parts else "No relevant sections found."
        hv_info = "\n".join(hv_values) if hv_values else ""
        hv_block = f"\n\nLive values:\n{hv_info}" if hv_info else ""

        # Build messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            for h in history[-8:]:
                r, c = h.get("role", ""), h.get("content", "")
                if r == "user":
                    messages.append(HumanMessage(content=c))
                elif r == "assistant":
                    messages.append(HumanMessage(content=c))

        prompt = f"""Document context:
{context}{hv_block}

Question: {message}"""
        messages.append(HumanMessage(content=prompt))

        # Step 5: LLM answers
        try:
            response = await asyncio.wait_for(self._llm.ainvoke(messages), timeout=180)
            return str(response.content)
        except asyncio.TimeoutError:
            return "请求超时。"
        except Exception as e:
            logger.exception("LLM error")
            return f"处理失败: {e}"

    async def close(self):
        if self._session:
            await self._session.close()
            self._initialized = False


_sysml_chat_agent: Optional[SysMLChatAgent] = None


def get_sysml_chat_agent() -> SysMLChatAgent:
    global _sysml_chat_agent
    if _sysml_chat_agent is None:
        _sysml_chat_agent = SysMLChatAgent()
    return _sysml_chat_agent


async def chat_with_sysml(message: str, history: Optional[List[dict]] = None) -> str:
    return await get_sysml_chat_agent().chat(message, history)
