"""
KG Build Agent — 基于 MCP 协议的知识图谱构建协调器

使用两个独立的 LangChain Agent：
1. Entity Extraction Agent — 从文档 Section 提取实体（PartDef/AttributeDef/RequirementDef 等）
2. Relation Extraction Agent — 从文档 Section 提取实体间关系（Connection/Interface/Allocation）

关键设计：
- Per-Section 独立会话，避免上下文无限增长
- search-before-create 去重策略
- MCP Server 持持久状态，Agent 通过 MCP 工具查询/操作
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import SecretStr

import config
from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
from mcp_client.mcp_session import MCPSession
from mcp_client.tool_wrapper import MCPToolWrapper, build_mcp_tools

logger = logging.getLogger(__name__)

# ── System Prompts ────────────────────────────────────────────

ENTITY_EXTRACTION_SYSTEM_PROMPT = """你是SysML v2实体提取专家。你的任务是从技术文档章节中识别系统架构实体。

## 工作规则：
1. 仔细阅读章节内容，理解其描述的系统和架构元素
2. 识别任何实质性的系统组成元素：组件、模块、设备、子系统、属性参数、接口、需求约束、数据实体
3. **关键**：对每个候选实体，先用 mcp__sysml_search_entity 搜索是否已存在——使用不同的表述尝试搜索（例如 "DAQ" 可能已注册为 "数据采集模块" 的别名）
4. 搜索到高置信度匹配（confidence >= 0.7）时，用 mcp__sysml_update_entity 补充信息（追加别名、描述、来源章节），而非创建新实体
5. 搜索无匹配或低置信度时，用 mcp__sysml_add_entity 创建新实体，并注册你识别到的所有别名
6. 每个实体必须记录原文出处（source_sections）
7. 章节中无系统架构内容时，直接结束，不要臆造实体

## 实体类型参考：
- PartDef/PartUsage: 系统组件、模块、设备、子系统
- AttributeDef/AttributeUsage: 属性、参数、特性、指标
- PortDef/PortUsage: 接口、端口、连接点
- ItemDef/ItemUsage: 数据结构、信息流、消息
- RequirementDef/RequirementUsage: 需求、约束、规范要求
- CommandDef: Shell命令、CLI运维操作（如 yhst, smu_tranfer_cmd, ncid, lspci）

## P2: 新增必提取信息类型：
8. **运维命令**：识别章节中的 Shell 命令/CLI 操作（如 smu_tranfer_cmd, yhst），创建为 CommandDef。记录命令文本、执行目标设备、完整调用示例
9. **主机名/标识**：识别节点的主机名（mn0, smu01）并用 mcp__sysml_set_hostname 为对应实体设置 hostname
10. **机柜映射**：识别机柜编号到物理位置的映射（R1P3→a/b/c/d），用 mcp__sysml_add_cabinet_instance 创建
11. **精确数量**：提取 "216个计算结点"、"2个管理结点" 等精确计数，用 mcp__sysml_add_quantity
12. **章节出处**：为每个创建的实体用 mcp__sysml_add_chapter_ref 记录章节来源
13. **IP配置**：提取 IP 地址、子网、网关等网络配置，用 mcp__sysml_add_ip_config
14. **中文显示名**：为 CamelCase 实体名设置中文显示名（compute_module → 计算模块），用 mcp__sysml_set_display_name

## 可用工具：
- mcp__sysml_search_entity: 多策略搜索（别名/子串）——用来查重
- mcp__sysml_add_entity: 创建新实体（含别名、描述、来源）
- mcp__sysml_update_entity: 补充/合并已有实体的信息
- mcp__sysml_add_alias: 为实体追加别名
- mcp__sysml_normalize_name: 标准化名称用于比较
- mcp__sysml_add_command: 创建运维命令实体（CommandDef）
- mcp__sysml_set_hostname: 为实体设置主机名标识
- mcp__sysml_add_cabinet_instance: 创建机柜实例（含子机柜）
- mcp__sysml_add_chapter_ref: 添加章节引用
- mcp__sysml_add_quantity: 设置精确数量
- mcp__sysml_add_ip_config: 添加IP网络配置
- mcp__sysml_set_display_name: 设置中文显示名"""

RELATION_EXTRACTION_SYSTEM_PROMPT = """你是SysML v2关系提取专家。你的任务是从技术文档章节中识别实体间关系。

## 工作规则：
1. 仔细阅读章节内容，识别实体之间的关联描述
2. 关系类型包括：物理连接、数据流、接口实现、功能分配、继承/组合、需求满足
3. 对每个候选关系，**必须**使用 mcp__sysml_search_entity 验证两个端点实体是否存在
4. 端点实体必须精确匹配——不确定时尝试不同表述搜索
5. 确认端点存在后再用 mcp__sysml_add_relation 创建关系
6. 关系描述要记录原文出处
7. 无明确关系描述时直接结束

## 关系类型：
- connection: 物理连接或数据流关系
- interface: 接口实现关系
- allocation: 功能/资源分配关系

## 可用工具：
- mcp__sysml_search_entity: 搜索实体——用来验证端点是否存在
- mcp__sysml_add_relation: 创建关系
- mcp__sysml_get_connections: 查看实体已有的关联"""


# ── Entity extraction tools (Phase 1) ──
ENTITY_TOOL_NAMES = [
    "sysml_search_entity",
    "sysml_add_entity",
    "sysml_update_entity",
    "sysml_add_alias",
    "sysml_normalize_name",
    "sysml_list_entities",
    "sysml_model_summary",
    "sysml_add_command",
    "sysml_set_hostname",
    "sysml_add_cabinet_instance",
    "sysml_add_chapter_ref",
    "sysml_add_quantity",
    "sysml_add_ip_config",
    "sysml_set_display_name",
]

# ── Relation extraction tools (Phase 2) ──
RELATION_TOOL_NAMES = [
    "sysml_add_relation",
    "sysml_search_entity",
    "sysml_get_connections",
    "sysml_list_relations",
    "sysml_list_entities",
    "sysml_model_summary",
]

# ── Merge/dedup tools ──
MERGE_TOOL_NAMES = [
    "sysml_suggest_merge",
    "sysml_merge_entities",
    "sysml_list_entities",
    "sysml_search_entity",
]


class SectionInfo:
    """文档 Section 信息"""
    def __init__(self, section_id: str, title: str, text: str,
                 parent_title: str = "", page: int = 0):
        self.section_id = section_id
        self.title = title
        self.text = text
        self.parent_title = parent_title
        self.page = page

    @property
    def path(self) -> str:
        if self.parent_title:
            return f"{self.parent_title} > {self.title}"
        return self.title


class KGBuildAgent:
    """知识图谱构建 Agent 协调器"""

    def __init__(
        self,
        db_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        persist_dir: Optional[str] = None,
    ):
        self.db_name = db_name
        self.model = model or self._get_config("KG_EXTRACTION_MODEL", "qwen3-vl:32b")
        self.temperature = temperature if temperature is not None else self._get_config("KG_EXTRACTION_TEMPERATURE", 0.1)
        self.max_iterations = self._get_config("KG_EXTRACTION_MAX_ITERATIONS", 10)
        self.timeout = self._get_config("KG_EXTRACTION_TIMEOUT", 120)
        base_dir = persist_dir or self._get_config("PERSIST_DIR", "./database")
        self.persist_dir = Path(base_dir) / db_name

        self._mcp_session: Optional[MCPSession] = None
        self._llm: Optional[ChatOpenAIWithReasoning] = None
        self._entity_tools: Optional[List[BaseTool]] = None
        self._relation_tools: Optional[List[BaseTool]] = None
        self._merge_tools: Optional[List[BaseTool]] = None

    @staticmethod
    def _get_config(key: str, default: Any = None) -> Any:
        return getattr(config.settings, key, default)

    @property
    def llm(self) -> ChatOpenAIWithReasoning:
        if self._llm is None:
            self._llm = ChatOpenAIWithReasoning(
                model=self.model,
                temperature=self.temperature,
                api_key=SecretStr(config.settings.OPENAI_API_KEY) if config.settings.OPENAI_API_KEY else None,
                base_url=config.settings.OPENAI_API_URL,
                timeout=self.timeout,
                streaming=True,
            )
        return self._llm

    async def initialize(self) -> None:
        """初始化 MCP 会话并构建工具包装"""
        if self._mcp_session is not None:
            return
        from mcp_client.mcp_session import create_sysml_mcp_session
        self._mcp_session = create_sysml_mcp_session()
        await self._mcp_session.initialize()

        # 加载该数据库的已有 KG（如果存在）
        kg_file = self.persist_dir / "knowledge_graph.sysml"
        if kg_file.exists():
            logger.info("Loading existing KG: %s", kg_file)
            await self._mcp_session.call_tool("sysml_load_model", {"file_path": str(kg_file)})

        self._entity_tools = await build_mcp_tools(
            self._mcp_session, prefix="mcp", tool_filter=ENTITY_TOOL_NAMES
        )
        self._relation_tools = await build_mcp_tools(
            self._mcp_session, prefix="mcp", tool_filter=RELATION_TOOL_NAMES
        )
        self._merge_tools = await build_mcp_tools(
            self._mcp_session, prefix="mcp", tool_filter=MERGE_TOOL_NAMES
        )
        logger.info("KGBuildAgent initialized: %d entity tools, %d relation tools, %d merge tools",
                     len(self._entity_tools), len(self._relation_tools), len(self._merge_tools))

    async def close(self) -> None:
        if self._mcp_session:
            await self._mcp_session.close()
            self._mcp_session = None

    # ── Document-level build ───────────────────────────────────

    async def build_kg_for_sections(
        self,
        doc_name: str,
        sections: List[SectionInfo],
    ) -> Dict[str, Any]:
        """为一份文档的多个 Section 构建知识图谱

        Args:
            doc_name: 文档名称
            sections: Section 信息列表

        Returns:
            构建结果摘要
        """
        await self.initialize()
        stats = {"entity_count_before": 0, "entity_count_after": 0,
                 "relation_count_before": 0, "relation_count_after": 0}

        # 获取初始统计
        summary = await self._summary()
        stats["entity_count_before"] = summary.get("total_entities", 0)
        stats["relation_count_before"] = summary.get("total_relations", 0)

        # 创建文档级 Package
        safe_doc_name = doc_name.replace(" ", "_").replace(".", "_")
        await self._mcp_session.call_tool("sysml_add_entity", {
            "entity_type": "Package",
            "name": safe_doc_name,
        })

        logger.info("Phase 1: Entity extraction for %d sections of '%s'", len(sections), doc_name)
        for i, section in enumerate(sections):
            logger.info("  Section %d/%d: %s", i + 1, len(sections), section.path)
            try:
                await self._extract_entities_from_section(section, safe_doc_name)
            except Exception as exc:
                logger.warning("  Entity extraction failed for '%s': %s", section.path, exc)

        # 跨 Section 去重合并
        logger.info("Cross-section entity deduplication")
        try:
            await self._deduplicate_entities()
        except Exception as exc:
            logger.warning("  Dedup failed: %s", exc)

        logger.info("Phase 2: Relation extraction for %d sections of '%s'", len(sections), doc_name)
        for i, section in enumerate(sections):
            logger.info("  Section %d/%d: %s", i + 1, len(sections), section.path)
            try:
                await self._extract_relations_from_section(section, safe_doc_name)
            except Exception as exc:
                logger.warning("  Relation extraction failed for '%s': %s", section.path, exc)

        # 保存 KG 文件
        await self._save_knowledge_graph()

        summary = await self._summary()
        stats["entity_count_after"] = summary.get("total_entities", 0)
        stats["relation_count_after"] = summary.get("total_relations", 0)
        return stats

    # ── Phase 1: Entity extraction ─────────────────────────────

    async def _extract_entities_from_section(self, section: SectionInfo, doc_package: str) -> None:
        """对单个 Section 进行实体提取（新建 Agent 会话）"""
        tools = self._entity_tools or []
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            debug=config.settings.AGENT_VERBOSE,
            name="entity_extractor",
        )

        prompt = f"""文档: {doc_package}
章节: {section.path}
页码: {section.page}

章节内容:
---BEGIN---
{section.text}
---END---

请提取本章节中描述的系统架构实体。对每个发现的实体，先搜索是否存在，再决定新建或更新。"""

        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=self.timeout,
            )
            logger.debug("Entity extraction result for '%s': %s",
                         section.path, str(result.get("messages", []))[:200])
        except asyncio.TimeoutError:
            logger.warning("Entity extraction timeout for '%s'", section.path)

    # ── Cross-section dedup ────────────────────────────────────

    async def _deduplicate_entities(self) -> None:
        """跨 Section 实体去重：获取合并建议 → Agent 审核 → 执行合并"""
        tools = self._merge_tools or []
        if not tools:
            return

        # Step 1: 获取合并建议
        suggest_result = await self._mcp_session.call_tool(
            "sysml_suggest_merge", {"threshold": 0.5}
        )
        suggestions = json.loads(suggest_result)
        if not suggestions.get("ok") or not suggestions.get("suggestions"):
            logger.info("No merge suggestions found")
            return

        suggestion_count = len(suggestions["suggestions"])
        logger.info("Found %d merge suggestions", suggestion_count)

        # Step 2: 让 Agent 审核并执行合并
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="""你是SysML v2实体合并审核专家。

## 工作规则：
1. 先用 mcp__sysml_get_entity 查看要合并的两个实体的详细信息
2. 判断它们是否是同一事物的不同名称/表述
3. 如果是，用 mcp__sysml_merge_entities 执行合并（将其中一个合并到另一个）
4. 如果不确定，保留不合并
5. 逐个处理合并建议""",
            debug=config.settings.AGENT_VERBOSE,
            name="entity_merger",
        )

        prompt = f"""以下实体对可能需要合并（可能描述同一事物），请逐一审核并处理：

{json.dumps(suggestions['suggestions'], ensure_ascii=False, indent=2)}

请逐个审核，对确实重复的实体对执行合并操作。"""

        try:
            await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=self.timeout * 2,
            )
        except asyncio.TimeoutError:
            logger.warning("Dedup Agent timeout")

    # ── Phase 2: Relation extraction ───────────────────────────

    async def _extract_relations_from_section(self, section: SectionInfo, doc_package: str) -> None:
        """对单个 Section 进行关系提取（新建 Agent 会话）"""
        tools = self._relation_tools or []
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=RELATION_EXTRACTION_SYSTEM_PROMPT,
            debug=config.settings.AGENT_VERBOSE,
            name="relation_extractor",
        )

        prompt = f"""文档: {doc_package}
章节: {section.path}

章节内容:
---BEGIN---
{section.text}
---END---

请提取本章节中描述的实体间关系。对每个关系，先用搜索工具验证端点实体是否存在，再创建关系。"""

        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=self.timeout,
            )
            logger.debug("Relation extraction result for '%s': %s",
                         section.path, str(result.get("messages", []))[:200])
        except asyncio.TimeoutError:
            logger.warning("Relation extraction timeout for '%s'", section.path)

    # ── Persistence ────────────────────────────────────────────

    async def _save_knowledge_graph(self) -> None:
        """保存知识图谱到 .sysml 文件"""
        if self._mcp_session is None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        kg_file = self.persist_dir / "knowledge_graph.sysml"
        await self._mcp_session.call_tool("sysml_save_model", {"file_path": str(kg_file)})
        logger.info("KG saved to %s", kg_file)

    async def _summary(self) -> Dict[str, Any]:
        if self._mcp_session is None:
            return {}
        result = await self._mcp_session.call_tool("sysml_model_summary", {})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {}
