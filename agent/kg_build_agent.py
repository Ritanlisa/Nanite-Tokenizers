"""
KG Build Agent — 基于 MCP 协议的知识图谱构建协调器

关键设计：
- 统一 Agent：单一 Agent 同时负责实体和关系提取（更符合阅读直觉）
- 文档树自主导航：Agent 通过 get_document_tree / read_section / mark_section_done 自主选择提取路径
- 打勾机制：提取完成的页面打勾隐藏，全部打勾即完成
- search-before-create 去重策略
- MCP Server 持久化状态，Agent 通过 MCP 工具查询/操作
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from pydantic import SecretStr

import config
from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
from mcp_client.mcp_session import MCPSession
from mcp_client.tool_wrapper import build_mcp_tools

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────

UNIFIED_EXTRACTION_SYSTEM_PROMPT = """你是技术文档知识提取专家。请自主导航文档树，提取系统架构知识到 SysML v2 模型。

## 工作流程
1. 先用 get_document_tree 了解文档整体层次结构
2. 选择未处理的章节/页面，用 read_section 读取完整内容
3. 从内容中同时识别实体和关系
4. 提取完成后用 mark_section_done 打勾
5. 用 get_progress 确认进度，继续处理剩余内容
6. **从整体到局部**：先掌握章节目录脉络，再深入具体页面
7. 无实质架构内容的页面直接打勾跳过

## 实体提取
- 识别任何实质性的系统组成元素
- 类型参考（非强制清单）：部件/模块/设备、属性/参数、接口/端口、数据结构、需求/约束
- 对每个候选实体，先用 mcp__sysml_search_entity 搜索是否已存在
- 搜索到高置信度匹配（confidence >= 0.7）时更新实体，而非新建
- 搜索无匹配时创建新实体，并注册你识别到的所有别名
- 记录每个实体的原文出处

## 关系提取
- 在实体提取的同时识别关系
- 关系类型：物理连接（connection）、数据流/信号（connection）、接口实现（interface）、功能分配（allocation）
- 创建关系前必须验证端点实体存在

## 可用工具

### 文档导航
- get_document_tree: 查看文档层次结构（含打勾状态）
- read_section: 读取指定节点/页面的完整内容
- mark_section_done: 打勾完成，支持批量
- get_progress: 查看提取进度百分比和剩余页面

### 知识图谱操作
- mcp__sysml_search_entity: 多策略搜索实体（查重用）
- mcp__sysml_add_entity: 创建新实体
- mcp__sysml_update_entity: 补充/合并已有实体
- mcp__sysml_add_alias: 为实体追加别名
- mcp__sysml_normalize_name: 标准化名称用于比较
- mcp__sysml_list_entities: 列出所有实体
- mcp__sysml_add_relation: 创建关系
- mcp__sysml_get_connections: 查看实体已有的关联
- mcp__sysml_list_relations: 列出所有关系
- mcp__sysml_model_summary: 全局统计信息"""

# ── Unified Tool Names ───────────────────────────────────────

UNIFIED_TOOL_NAMES = [
    "sysml_search_entity",
    "sysml_add_entity",
    "sysml_update_entity",
    "sysml_add_alias",
    "sysml_normalize_name",
    "sysml_list_entities",
    "sysml_add_relation",
    "sysml_get_connections",
    "sysml_list_relations",
    "sysml_model_summary",
]

# ═══════════════════════════════════════════════════════════════
# 以下为"冻结"动态数据工具 —— 暂不纳入提取流程，保留代码以备后用
# ═══════════════════════════════════════════════════════════════
# "sysml_add_command",          # 运维命令创建 (CommandDef)
# "sysml_set_hostname",         # 主机名标识
# "sysml_add_cabinet_instance", # 机柜实例映射
# "sysml_add_chapter_ref",      # 章节出处引用
# "sysml_add_quantity",         # 精确数量提取
# "sysml_add_ip_config",        # IP 网络配置
# "sysml_set_display_name",     # 中文显示名
# ═══════════════════════════════════════════════════════════════

MERGE_TOOL_NAMES = [
    "sysml_suggest_merge",
    "sysml_merge_entities",
    "sysml_list_entities",
    "sysml_search_entity",
]


# ── Document Tree State ──────────────────────────────────────

@dataclass
class DocTreeNode:
    """文档树中的一个节点（章节或页面）"""
    node_id: str
    title: str
    page_start: int
    page_end: int
    text: str
    level: int
    parent_id: str
    children: List[str] = field(default_factory=list)  # child node_ids
    processed: bool = False


class DocumentTreeState:
    """
    文档树状态管理器。
    从 RAG_DB_Document 构建层次树，管理打勾状态，提供导航信息。
    """

    def __init__(self, rag_doc):
        self._doc_name: str = getattr(rag_doc, "doc_name", "") or ""
        self._doc_title: str = getattr(rag_doc, "title", "") or ""
        self.nodes: Dict[str, DocTreeNode] = {}
        self._all_page_ids: List[str] = []

        if not getattr(rag_doc, "_skip_build", False):
            self._build_tree(rag_doc)
        else:
            logger.debug("DocumentTreeState: skipping tree build (manual population)")

    def _build_tree(self, rag_doc) -> None:
        """从 RAG_DB_Document 构建层次树（两步法：先建章节层次，再挂载合并后的页面）"""
        from rag.document_interface import PageType

        _counter = [0]

        def _next_id() -> str:
            _counter[0] += 1
            return f"n{_counter[0]}"

        # ── Step 1: 收集并合并所有内容页（按物理页分组，合并 SemiPage 片段）──
        page_groups: Dict[int, Dict[str, Any]] = {}
        # {page_num: {"title": ..., "texts": [...]}}

        mono_pages = getattr(rag_doc, "get_mono_pages", None)
        if mono_pages:
            try:
                mono_pages = mono_pages()
            except Exception:
                mono_pages = []
        else:
            mono_pages = []

        for mp in (mono_pages or []):
            page_num = int(mp.metadata.get("page", 0)
                           or mp.metadata.get("physical_page", 0) or 0)
            if page_num <= 0:
                continue
            cat = getattr(mp, "category", "")
            if cat in (PageType.COVER, PageType.CATALOGUE):
                continue

            text = mp.markdown_text or ""
            title = (getattr(mp, "title", "")
                     or mp.metadata.get("section_title", "")
                     or f"第{page_num}页")

            if page_num not in page_groups:
                page_groups[page_num] = {"title": title, "texts": []}
            if text.strip():
                page_groups[page_num]["texts"].append(text)
            if title and f"第{page_num}页" not in title:
                page_groups[page_num]["title"] = title

        # 创建合并后的页面节点映射
        page_nodes: Dict[int, str] = {}  # page_num → node_id
        for page_num in sorted(page_groups):
            info = page_groups[page_num]
            merged_text = "\n\n".join(info["texts"])
            if not merged_text.strip():
                continue
            node_id = _next_id()
            node = DocTreeNode(
                node_id=node_id, title=str(info["title"]),
                page_start=page_num, page_end=page_num,
                text=merged_text,
                level=0, parent_id="", children=[],
            )
            self.nodes[node_id] = node
            self._all_page_ids.append(node_id)
            page_nodes[page_num] = node_id

        if not self._all_page_ids:
            logger.warning("No content pages found for '%s'", self._doc_name)
            return

        # ── Step 2: 构建章节层次树，并将页面挂到对应章节下 ──
        def _traverse_chapters(page, parent_id: str, level: int) -> Tuple[int, int]:
            """遍历章节树返回 (min_page, max_page)，同时将页面挂载到最近的章节"""
            from rag.document_interface import Chapter as ChCls
            from rag.document_interface import MonoPage as MPCls

            if isinstance(page, ChCls):
                node_id = _next_id()
                chapter_title = page.title or f"章节 {node_id}"
                min_p, max_p = 99999, 0

                child_range = (0, 0)
                for child in (page.SubContent or []):
                    child_range = _traverse_chapters(child, node_id, level + 1)
                    min_p = min(min_p, child_range[0]) if child_range[0] > 0 else min_p
                    max_p = max(max_p, child_range[1]) if child_range[1] > 0 else max_p

                # 将章节范围内的页面节点挂载到此章节
                children_ids = []
                for pn in sorted(page_nodes):
                    if min_p <= pn <= max_p and pn > 0:
                        pn_node = self.nodes.get(page_nodes[pn])
                        if pn_node:
                            pn_node.parent_id = node_id
                            pn_node.level = level + 1
                            children_ids.append(page_nodes[pn])

                if min_p == 99999:
                    min_p = 0
                node = DocTreeNode(
                    node_id=node_id, title=chapter_title,
                    page_start=min_p, page_end=max_p,
                    text="",
                    level=level, parent_id=parent_id,
                    children=children_ids,
                )
                self.nodes[node_id] = node
                return (min_p, max_p)

            elif isinstance(page, MPCls):
                page_num = int(page.metadata.get("page", 0)
                               or page.metadata.get("physical_page", 0) or 0)
                return (page_num, page_num)

            return (0, 0)

        # 从根文档的 SubContent 开始遍历
        sub_content = getattr(rag_doc, "SubContent", None) or []
        if sub_content:
            for child in sub_content:
                _traverse_chapters(child, "", 1)
        else:
            # 没有 SubContent 的纯页面列表：直接标记为顶层页面
            for nid in self._all_page_ids:
                node = self.nodes[nid]
                node.level = 1

        logger.info("DocumentTreeState built: %d nodes, %d leaf pages for '%s'",
                     len(self.nodes), len(self._all_page_ids), self._doc_name)

    @property
    def processed_count(self) -> int:
        return sum(1 for n in self.nodes.values()
                   if n.processed and n.page_start > 0)

    @property
    def total_pages(self) -> int:
        return len(self._all_page_ids)

    def unprocessed_page_ids(self) -> List[str]:
        return [nid for nid in self._all_page_ids
                if nid in self.nodes and not self.nodes[nid].processed]

    def mark_processed(self, node_ids: List[str]) -> int:
        count = 0
        for nid in node_ids:
            if nid in self.nodes:
                self.nodes[nid].processed = True
                count += 1
        return count

    def is_complete(self) -> bool:
        return all(self.nodes[nid].processed for nid in self._all_page_ids
                   if nid in self.nodes)

    def get_tree_structure(self) -> Dict[str, Any]:
        """返回给 LLM 的层次化树视图"""

        def _node_view(node_id: str) -> Dict[str, Any]:
            node = self.nodes[node_id]
            view = {
                "node_id": node.node_id,
                "title": node.title,
                "level": node.level,
            }
            if node.page_start > 0:
                view["page"] = node.page_start if node.page_start == node.page_end \
                    else f"{node.page_start}-{node.page_end}"
                view["processed"] = node.processed
                view["text_length"] = len(node.text)
            if node.children:
                view["children"] = [_node_view(c) for c in node.children]
            return view

        root_children = [nid for nid in self.nodes
                         if self.nodes[nid].parent_id == ""]
        structure = [_node_view(c) for c in root_children]

        return {
            "document": self._doc_title or self._doc_name,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_count,
            "remaining_pages": self.total_pages - self.processed_count,
            "structure": structure,
        }

    def get_progress(self) -> Dict[str, Any]:
        remaining = self.unprocessed_page_ids()
        return {
            "total": self.total_pages,
            "done": self.processed_count,
            "remaining": len(remaining),
            "percent": round(self.processed_count / max(self.total_pages, 1) * 100, 1),
            "remaining_pages": [
                {"node_id": nid, "title": self.nodes[nid].title,
                 "page": self.nodes[nid].page_start}
                for nid in remaining[:20]
            ],
            "complete": self.is_complete(),
        }


# ── Navigation Tools (本地 LangChain 工具) ────────────────────

def create_navigation_tools(tree_state: DocumentTreeState) -> List[BaseTool]:
    """创建文档导航工具集"""

    @tool
    def get_document_tree() -> str:
        """
        查看文档层次结构。
        返回章节目录树、每页的打勾状态、剩余未处理页面数。
        首次使用时调用以了解文档全局结构。
        """
        return json.dumps(tree_state.get_tree_structure(), ensure_ascii=False, indent=2)

    @tool
    def read_section(node_id: str) -> str:
        """
        读取指定节点的完整文本内容。

        Args:
            node_id: 节点ID（来自 get_document_tree 返回的 node_id）
        """
        node = tree_state.nodes.get(node_id)
        if node is None:
            return json.dumps({"error": f"Node not found: {node_id}"}, ensure_ascii=False)
        return json.dumps({
            "node_id": node_id,
            "title": node.title,
            "page": node.page_start,
            "text": node.text,
        }, ensure_ascii=False)

    @tool
    def mark_section_done(node_ids: List[str]) -> str:
        """
        标记一个或多个页面/小节为提取完成。已标记的页面将在文档树中显示为已处理。
        支持一次标记多个节点。

        Args:
            node_ids: 要标记的节点ID列表
        """
        count = tree_state.mark_processed(node_ids)
        progress = tree_state.get_progress()
        return json.dumps({
            "marked": count,
            "progress": progress,
        }, ensure_ascii=False, indent=2)

    @tool
    def get_progress() -> str:
        """
        查看整体提取进度：已完成/总计页面数、百分比、剩余页面列表。
        当剩余页面为0时提取完成。
        """
        return json.dumps(tree_state.get_progress(), ensure_ascii=False, indent=2)

    return [get_document_tree, read_section, mark_section_done, get_progress]


# ── SectionInfo (保留向后兼容) ───────────────────────────────

class SectionInfo:
    """文档 Section 信息（保留向后兼容）"""
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


# ── KGBuildAgent ─────────────────────────────────────────────

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
        self.temperature = temperature if temperature is not None \
            else self._get_config("KG_EXTRACTION_TEMPERATURE", 0.1)
        self.max_iterations = self._get_config("KG_EXTRACTION_MAX_ITERATIONS", 60)
        self.timeout = self._get_config("KG_EXTRACTION_TIMEOUT", 120)
        base_dir = persist_dir or self._get_config("PERSIST_DIR", "./database")
        self.persist_dir = Path(base_dir) / db_name

        self._mcp_session: Optional[MCPSession] = None
        self._llm: Optional[ChatOpenAIWithReasoning] = None
        self._unified_tools: Optional[List[BaseTool]] = None
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
                api_key=SecretStr(config.settings.OPENAI_API_KEY)
                if config.settings.OPENAI_API_KEY else None,
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
            await self._mcp_session.call_tool(
                "sysml_load_model", {"file_path": str(kg_file)})

        self._unified_tools = await build_mcp_tools(
            self._mcp_session, prefix="mcp", tool_filter=UNIFIED_TOOL_NAMES,
        )
        self._merge_tools = await build_mcp_tools(
            self._mcp_session, prefix="mcp", tool_filter=MERGE_TOOL_NAMES,
        )
        logger.info("KGBuildAgent initialized: %d unified tools, %d merge tools",
                     len(self._unified_tools), len(self._merge_tools))

    async def close(self) -> None:
        if self._mcp_session:
            await self._mcp_session.close()
            self._mcp_session = None

    # ── 新接口：从文档树构建 KG ────────────────────────────────

    async def build_kg_from_document(self, rag_doc) -> Dict[str, Any]:
        """
        从 RAG_DB_Document 文档树构建知识图谱。

        Args:
            rag_doc: RAG_DB_Document 实例

        Returns:
            构建结果摘要
        """
        await self.initialize()

        doc_name = getattr(rag_doc, "doc_name", "")
        stats = {
            "entity_count_before": 0, "entity_count_after": 0,
            "relation_count_before": 0, "relation_count_after": 0,
        }

        # 获取初始统计
        summary = await self._summary()
        stats["entity_count_before"] = summary.get("total_entities", 0)
        stats["relation_count_before"] = summary.get("total_relations", 0)

        # 创建文档级 Package
        safe_doc_name = doc_name.replace(" ", "_").replace(".", "_")
        if safe_doc_name:
            await self._mcp_session.call_tool("sysml_add_entity", {
                "entity_type": "Package",
                "name": safe_doc_name,
            })

        # 构建文档树状态
        tree_state = DocumentTreeState(rag_doc)
        logger.info("Document tree: %d leaf pages", tree_state.total_pages)

        if tree_state.total_pages == 0:
            logger.warning("No content pages found in document")
            return stats

        # 创建导航工具
        nav_tools = create_navigation_tools(tree_state)

        # 合并所有工具：导航工具 + MCP 统一工具
        all_tools = list(nav_tools) + (self._unified_tools or [])

        # 创建统一 Agent
        agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=UNIFIED_EXTRACTION_SYSTEM_PROMPT,
            debug=config.settings.AGENT_VERBOSE,
            name="kg_extractor",
        )

        prompt = f"""文档: {doc_name}

请开始提取：
1. 先用 get_document_tree 查看文档结构
2. 从最顶层章节开始，逐章逐页向下深入
3. 每读完一页，同时提取实体和关系，然后 mark_section_done 打勾
4. 用 get_progress 确认进度，直到所有页面完成

注意：从整体到局部，先理解文档脉络再深入细节。"""

        logger.info("Starting unified extraction for '%s'", doc_name)
        try:
            # Agent 自主循环提取，使用较大的超时
            await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": self.max_iterations},
                ),
                timeout=self.timeout * 30,  # 全局超时：30个 timeouts
            )
        except asyncio.TimeoutError:
            logger.warning("Global extraction timeout for '%s'", doc_name)

        # 跨章节去重合并
        logger.info("Cross-section entity deduplication after extraction")
        try:
            await self._deduplicate_entities()
        except Exception as exc:
            logger.warning("Dedup failed: %s", exc)

        # 保存 KG 文件
        await self._save_knowledge_graph()

        summary = await self._summary()
        stats["entity_count_after"] = summary.get("total_entities", 0)
        stats["relation_count_after"] = summary.get("total_relations", 0)

        logger.info("KG build complete: entities %d→%d, relations %d→%d",
                     stats["entity_count_before"], stats["entity_count_after"],
                     stats["relation_count_before"], stats["relation_count_after"])
        return stats

    # ── 旧接口：向后兼容 ───────────────────────────────────────

    async def build_kg_for_sections(
        self,
        doc_name: str,
        sections: List[SectionInfo],
    ) -> Dict[str, Any]:
        """
        为一份文档的多个 Section 构建知识图谱（向后兼容接口）。

        注意：此接口已废弃，推荐使用 build_kg_from_document。
        它将 sections 包装为简化的文档树。
        """
        await self.initialize()
        stats = {
            "entity_count_before": 0, "entity_count_after": 0,
            "relation_count_before": 0, "relation_count_after": 0,
        }

        summary = await self._summary()
        stats["entity_count_before"] = summary.get("total_entities", 0)
        stats["relation_count_before"] = summary.get("total_relations", 0)

        # 创建文档级 Package
        safe_doc_name = doc_name.replace(" ", "_").replace(".", "_")
        if safe_doc_name and self._mcp_session:
            await self._mcp_session.call_tool("sysml_add_entity", {
                "entity_type": "Package",
                "name": safe_doc_name,
            })

        # 构建简易文档树（从 flat sections）
        tree_state = _build_simple_tree(doc_name, sections)
        nav_tools = create_navigation_tools(tree_state)
        all_tools = list(nav_tools) + (self._unified_tools or [])

        agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=UNIFIED_EXTRACTION_SYSTEM_PROMPT,
            debug=config.settings.AGENT_VERBOSE,
            name="kg_extractor",
        )

        prompt = f"""文档: {doc_name}

请开始提取：
1. 先用 get_document_tree 查看结构
2. 逐页读取、提取、打勾
3. 直到所有页面完成"""

        try:
            await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": self.max_iterations},
                ),
                timeout=self.timeout * max(10, len(sections)),
            )
        except asyncio.TimeoutError:
            logger.warning("Extraction timeout for '%s'", doc_name)

        await self._deduplicate_entities()
        await self._save_knowledge_graph()

        summary = await self._summary()
        stats["entity_count_after"] = summary.get("total_entities", 0)
        stats["relation_count_after"] = summary.get("total_relations", 0)
        return stats

    # ── Cross-section dedup ────────────────────────────────────

    async def _deduplicate_entities(self) -> None:
        """跨章节实体去重：获取合并建议 → Agent 审核 → 执行合并"""
        if not self._merge_tools or self._mcp_session is None:
            return

        suggest_result = await self._mcp_session.call_tool(
            "sysml_suggest_merge", {"threshold": 0.5}
        )
        suggestions = json.loads(suggest_result)
        if not suggestions.get("ok") or not suggestions.get("suggestions"):
            logger.info("No merge suggestions found")
            return

        suggestion_count = len(suggestions["suggestions"])
        logger.info("Found %d merge suggestions", suggestion_count)

        agent = create_agent(
            model=self.llm,
            tools=self._merge_tools,
            system_prompt="""你是SysML v2实体合并审核专家。

## 工作规则：
1. 先用 mcp__sysml_get_entity 查看要合并的两个实体的详细信息
2. 判断它们是否是同一事物的不同名称/表述
3. 如果是，用 mcp__sysml_merge_entities 执行合并
4. 如果不确定，保留不合并
5. 逐个处理合并建议""",
            debug=config.settings.AGENT_VERBOSE,
            name="entity_merger",
        )

        prompt = f"""以下实体对可能需要合并，请逐一审核并处理：

{json.dumps(suggestions['suggestions'], ensure_ascii=False, indent=2)}

请逐个审核，对确实重复的实体对执行合并操作。"""

        try:
            await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=self.timeout * 2,
            )
        except asyncio.TimeoutError:
            logger.warning("Dedup Agent timeout")

    # ── Persistence ────────────────────────────────────────────

    async def _save_knowledge_graph(self) -> None:
        """保存知识图谱到 .sysml 文件"""
        if self._mcp_session is None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        kg_file = self.persist_dir / "knowledge_graph.sysml"
        await self._mcp_session.call_tool(
            "sysml_save_model", {"file_path": str(kg_file)})
        logger.info("KG saved to %s", kg_file)

    async def _summary(self) -> Dict[str, Any]:
        if self._mcp_session is None:
            return {}
        result = await self._mcp_session.call_tool("sysml_model_summary", {})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {}


def _build_simple_tree(doc_name: str, sections: List[SectionInfo]) -> DocumentTreeState:
    """从 flat SectionInfo 列表构建简易文档树（用于向后兼容）"""

    class _FakeDoc:
        _skip_build = True
        doc_name = doc_name
        title = doc_name

    tree = DocumentTreeState(_FakeDoc())

    _counter = [0]

    def _next_id() -> str:
        _counter[0] += 1
        return f"n{_counter[0]}"

    for i, sec in enumerate(sections):
        node_id = _next_id()
        tree.nodes[node_id] = DocTreeNode(
            node_id=node_id,
            title=sec.path or f"Section {i + 1}",
            page_start=sec.page,
            page_end=sec.page,
            text=sec.text,
            level=1,
            parent_id="",
            children=[],
        )
        tree._all_page_ids.append(node_id)

    return tree
