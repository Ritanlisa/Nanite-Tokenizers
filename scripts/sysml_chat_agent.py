"""
SysML Chat Agent
—— Uses sysml_retrieve to pull entity + k-layer graph, then asks LLM to answer.
"""
from __future__ import annotations

import asyncio, logging, json, re
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

SYSTEM_PROMPT = """You are a QA agent for the Huchao supercomputer SysML knowledge graph.
Answer questions based on entity information retrieved from the knowledge graph.
If live values (temperature, fan speed, etc.) are available, include them.
If the information is insufficient, say so clearly.

Be concise and accurate. Use Chinese unless the user asks in English."""


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

    def _extract_entity_names(self, message: str) -> list[str]:
        """Extract candidate entity names from user message."""
        candidates: list[str] = []
        msg = message.strip()

        # Quoted names
        quoted = re.findall(r'["\'""\u300c\u300d]([^"\'""\u300c\u300d]+)["\'""\u300c\u300d]', msg)
        candidates.extend(quoted)

        # Chinese compound names matching common patterns
        patterns = [
            r'([\u4e00-\u9fff]{2,8}(?:模块|系统|节点|引擎|服务|仪表盘|数据库|传感器|计算机|设备|组件|接口|控制器|处理器|存储器|网络|总线|通道))',
        ]
        for pat in patterns:
            found = re.findall(pat, msg)
            candidates.extend(found)

        # General Chinese words (2-6 chars) as fallback
        if not candidates:
            noise = {'什么', '哪个', '怎么', '为什么', '如何', '请问', '帮我', '我想', '可以', '是否',
                     '有没有', '在哪里', '是什么', '怎么样', '好不好', '多大', '多少', '哪些'}
            words = re.findall(r'[\u4e00-\u9fff]{2,6}', msg)
            candidates.extend([w for w in words if w not in noise][:5])

        seen = set()
        result = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result[:5]

    def _build_retrieval_context(self, retrieve_result: dict) -> str:
        """Build readable context from sysml_retrieve result."""
        if not retrieve_result.get("ok"):
            return ""

        entity = retrieve_result.get("entity", {})
        graph = retrieve_result.get("relationship_graph", {})

        parts = []
        parts.append(f"Entity: {entity.get('name', '?')} ({entity.get('type', entity.get('class', '?'))})")

        meta = entity.get("metadata", {})
        if meta.get("description"):
            parts.append(f"Description: {meta['description']}")
        if meta.get("properties"):
            parts.append(f"Properties: {json.dumps(meta['properties'], ensure_ascii=False)}")
        if entity.get("aliases"):
            parts.append(f"Aliases: {', '.join(entity['aliases'])}")

        members = entity.get("members", [])
        if members:
            member_lines = [f"  - {m.get('type','?')} {m.get('name','')}" + (f" = {m['value']}" if m.get('value') else "")
                          for m in members[:20]]
            parts.append(f"Members:\n" + "\n".join(member_lines))

        total_nodes = graph.get("total_nodes", 0)
        total_edges = graph.get("total_edges", 0)
        if total_nodes > 1 or total_edges > 0:
            parts.append(f"\nRelationship Graph (depth={graph.get('depth',0)}): {total_nodes} nodes, {total_edges} edges")
            nodes = graph.get("nodes", {})
            parts.append(f"Nodes: {', '.join(f'{n}({v.get('type',v.get('class','?'))})' for n,v in nodes.items())}")
            for e in graph.get("edges", []):
                parts.append(f"  {e['from']} --[{e.get('relation',{}).get('name','?')}]--> {e['to']}")

        return "\n".join(parts)

    async def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        if not self._initialized:
            await self.initialize()

        # Step 1: Extract candidate entity names
        candidates = self._extract_entity_names(message)
        logger.info(f"Candidates: {candidates}")

        # Step 2: Call sysml_retrieve for each candidate
        context_parts = []
        retrieved = set()
        for name in candidates:
            for k_val in [1, 2, 0]:
                result_str = await self._call_mcp("sysml_retrieve", {"name": name, "k": k_val})
                try:
                    result = json.loads(result_str)
                    if result.get("ok"):
                        ename = result.get("entity", {}).get("name", "")
                        if ename in retrieved:
                            continue
                        retrieved.add(ename)
                        ctx = self._build_retrieval_context(result)
                        if ctx:
                            context_parts.append(ctx)
                        break
                except Exception:
                    pass

        # Step 3: Resolve HVs if user asks about live values
        hv_map = {
            "温度": "temperature", "风扇": "fan_speed", "转速": "fan_speed",
            "湿度": "humidity", "电压": "voltage_level", "气压": "pressure",
            "功率": "power_load", "电池": "battery_capacity",
        }
        hv_parts = []
        for keyword, hv_id in hv_map.items():
            if keyword in message:
                hv_result = await self._call_mcp("sysml_resolve_hv", {"hv_id": hv_id})
                try:
                    hv_data = json.loads(hv_result)
                    if hv_data.get("ok"):
                        hv_parts.append(f"{hv_id}: {hv_data.get('value', '?')}")
                except Exception:
                    pass

        # Step 4: Build messages and ask LLM
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant entities found."
        hv_block = f"\n\nLive values:\n" + "\n".join(hv_parts) if hv_parts else ""

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            for h in history[-6:]:
                r, c = h.get("role", ""), h.get("content", "")
                if r == "user":
                    messages.append(HumanMessage(content=c))
                elif r == "assistant":
                    messages.append(HumanMessage(content=c))

        prompt = f"""Knowledge Graph context:
{context}{hv_block}

User Question: {message}"""
        messages.append(HumanMessage(content=prompt))

        try:
            response = await asyncio.wait_for(self._llm.ainvoke(messages), timeout=180)
            return str(response.content)
        except asyncio.TimeoutError:
            return "请求处理超时。"
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
