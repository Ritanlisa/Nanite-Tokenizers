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
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from pydantic import SecretStr

import config
from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
from mcp_client.mcp_session import MCPSession
from mcp_client.tool_wrapper import build_mcp_tools

logger = logging.getLogger(__name__)

# Suppress non-fatal MCP client stdio parse errors
logging.getLogger("mcp.client.stdio").setLevel(logging.WARNING)

# ── System Prompt ────────────────────────────────────────────

UNIFIED_EXTRACTION_SYSTEM_PROMPT = """你是技术文档知识提取专家。请自主导航文档树，提取系统架构知识到 SysML v2 模型。

## ⚠️ 关键规则 — 不打勾空页
每读完一页，**必须先用 mcp__sysml_search_entity + mcp__sysml_add_entity (或 mcp__sysml_add_relation) 提取至少一个实体或关系**，然后才能 mark_section_done 打勾。
- 有架构内容 → 搜索去重 → 创建/更新实体或关系 → 打勾
- 确实没有任何系统架构内容 → 直接打勾（极少发生）
- **禁止**读完页面后思考一下就打勾而不执行任何 MCP 知识图谱操作

## 工作流程（每页的标准操作序列）
1. read_section(node_id) → 获取页面完整文本
2. 从文本中识别实体：组件名、模块名、参数、接口、命令、设备...
3. 对每个发现的实体：
   a. mcp__sysml_search_entity(name) → 查重
   b. 无匹配 → mcp__sysml_add_entity(...) → 创建
   c. 有匹配 → mcp__sysml_update_entity(...) → 补充
4. 从文本中识别关系：连接、数据流、依赖...
5. 对每个发现的关系：mcp__sysml_add_relation(...) → 创建
6. mark_section_done([node_id]) → 打勾
7. get_progress() → 确认进度，继续下一页

## 实体提取
- 类型参考（非强制清单）：部件/模块/设备、属性/参数、接口/端口、数据结构、需求/约束、命令
- 搜索置信度 >= 0.7 时更新，< 0.7 时新建
- 记录原文出处和所有别名

## 关系提取
- 类型：connection（物理连接/数据流）、interface（接口实现）、allocation（功能分配）
- 创建前验证端点存在

## 文档导航
1. 先用 get_document_tree 了解整体结构
2. **从整体到局部**：先掌握章节目录脉络，再深入具体页面
3. 全部打勾即完成

## 可用工具
### 文档导航
- get_document_tree / read_section / mark_section_done / get_progress

### 知识图谱操作
- mcp__sysml_search_entity / mcp__sysml_add_entity / mcp__sysml_update_entity
- mcp__sysml_add_alias / mcp__sysml_normalize_name / mcp__sysml_list_entities
- mcp__sysml_add_relation / mcp__sysml_get_connections / mcp__sysml_list_relations
- mcp__sysml_model_summary"""


# ── Fast Extraction Prompt (qwen3:8b — JSON output, no tool calls) ──

EXTRACTION_CANDIDATES_PROMPT = """你是一个技术文档实体提取器。阅读给定的文档页面文本，输出其中描述的系统架构实体和关系。

## 输出格式
输出一个JSON数组，每个元素是一个实体或关系：

实体格式: {"type":"PartDef","name":"实体名","description":"简短描述","aliases":["别名"]}
关系格式: {"relation":true,"type":"Connection","source":"源实体","target":"目标实体","description":"关系描述"}

## 实体类型
- PartDef: 系统组件、模块、设备、子系统、机柜、服务器
- AttributeDef: 属性、参数、特性、指标（如带宽400Gbps、处理器数量1024）
- PortDef: 接口、端口、连接点
- ItemDef: 数据结构、信息流、消息
- RequirementDef: 需求、约束、规范要求
- CommandDef: Shell命令、CLI操作、工具命令（如 yhst, smu_tranfer_cmd, ncid, lspci）

## 关系类型
- Connection: 物理连接或数据流关系
- Interface: 接口实现关系
- Allocation: 功能/资源分配关系

## 示例
输入文本: "系统提供1个FT计算柜（1024个处理器）和10个MT加速柜（共10240个加速器）"
输出:
```json
[
  {"type":"PartDef","name":"FT计算柜","description":"1024个处理器","aliases":["FT柜","计算柜"]},
  {"type":"PartDef","name":"MT加速柜","description":"10个加速柜共10240个加速器","aliases":["MT柜","加速柜"]},
  {"type":"AttributeDef","name":"处理器数量","description":"FT计算柜包含1024个处理器","aliases":["CPU数量"]}
]
```

输入文本: "通过 smu_tranfer_cmd 转发命令，yhst 查看加电信息"
输出:
```json
[
  {"type":"CommandDef","name":"smu_tranfer_cmd","description":"通过SMU向指定CMU转发命令","aliases":["SMU命令","smu转发"]},
  {"type":"CommandDef","name":"yhst","description":"查看所有结点加电信息","aliases":["加电查询","yhst命令"]},
  {"relation":true,"type":"Connection","source":"smu_tranfer_cmd","target":"yhst","description":"通过smu_tranfer_cmd转发yhst命令"}
]
```

输入文本无任何系统架构内容时输出: []"""


# ── Enrichment JSON Prompt (qwen3:8b — JSON enrichment instructions) ──

ENRICHMENT_JSON_PROMPT = """你是SysML v2知识图谱专家。审查已有KG实体列表，以JSON格式输出需要添加的关系和别名。

## 输出格式
输出一个JSON数组，每个元素是一个富化操作：

关系操作: {"action":"add_relation","type":"allocation|connection|interface","source":"实体A","target":"实体B","description":"关系描述"}
别名操作: {"action":"add_alias","entity":"实体名","alias":"别名"}
更新操作: {"action":"update_entity","entity":"实体名","append_description":"补充描述"}

## 硬性规则
- **每个实体至少创建1条关系**（add_relation），禁止孤立实体
- 如果实在无法关联，连接到文档 Package
- 优先在同章节实体之间创建关系
- 补充常用中英文别名
- 属性/端口类实体应关联到所属的 PartDef
- 宁可冗余不要遗漏

## 示例
实体: [{"name":"FT计算柜","type":"PartDef","description":"1024个处理器"},{"name":"处理器数量","type":"AttributeDef","description":"1024"}]
输出:
```json
[{"action":"add_relation","type":"allocation","source":"处理器数量","target":"FT计算柜","description":"处理器数量是FT计算柜的属性"}]
```"""

# ── Unified Tool Names ───────────────────────────────────────

UNIFIED_TOOL_NAMES = [
    "sysml_search_entity",
    "sysml_add_entity",
    "sysml_update_entity",
    "sysml_add_alias",
    "sysml_normalize_name",
    "sysml_list_entities",
    "sysml_get_entity",
    "sysml_add_relation",
    "sysml_get_connections",
    "sysml_list_relations",
    "sysml_model_summary",
    "sysml_connected_components",
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
        light_model: Optional[str] = None,
        temperature: Optional[float] = None,
        persist_dir: Optional[str] = None,
    ):
        self.db_name = db_name
        self.model = model or self._get_config("KG_EXTRACTION_MODEL", "qwen3:8b")
        self.light_model = light_model or self._get_config("KG_LIGHT_MODEL", "qwen3:8b")
        self.temperature = temperature if temperature is not None \
            else self._get_config("KG_EXTRACTION_TEMPERATURE", 0.1)
        self.max_iterations = self._get_config("KG_EXTRACTION_MAX_ITERATIONS", 60)
        self.timeout = self._get_config("KG_EXTRACTION_TIMEOUT", 120)
        base_dir = persist_dir or self._get_config("PERSIST_DIR", "./database")
        self.persist_dir = Path(base_dir) / db_name

        self._mcp_session: Optional[MCPSession] = None
        self._llm: Optional[ChatOpenAIWithReasoning] = None
        self._light_llm: Optional[ChatOpenAIWithReasoning] = None
        self._unified_tools: Optional[List[BaseTool]] = None
        self._merge_tools: Optional[List[BaseTool]] = None

        self._save_lock = asyncio.Lock()
        self._save_desired = False
        self._save_task: Optional[asyncio.Task] = None

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

    @property
    def light_llm(self) -> ChatOpenAIWithReasoning:
        """快速小模型 — 用于纯文本实体候选提取"""
        if self._light_llm is None:
            self._light_llm = ChatOpenAIWithReasoning(
                model=self.light_model,
                temperature=0.0,
                api_key=SecretStr(config.settings.OPENAI_API_KEY)
                if config.settings.OPENAI_API_KEY else None,
                base_url=config.settings.OPENAI_API_URL,
                timeout=600,
                streaming=False,
            )
        return self._light_llm

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

    # ── 新接口：从文档树构建 KG（两模型架构）────────────────

    async def build_kg_from_document(self, rag_doc) -> Dict[str, Any]:
        """
        从 RAG_DB_Document 构建知识图谱。
        支持断点续跑：崩溃/Ctrl+C 后可恢复，不重复已完成阶段。

        阶段:
        1. 并行 qwen3:8b 逐节扫描提取实体/关系候选（Semaphore 控制并发）
        2. 跨章节去重合并（无 LLM）
        3. 并行 qwen3:8b 逐节 JSON 富化关系/别名
        4. 并行图聚合消除孤立子图
        """
        await self.initialize()

        doc_name = getattr(rag_doc, "doc_name", "")
        build = self._load_build_state(doc_name)
        phase = build.get("phase", "")

        if phase == "done":
            logger.info("Document %s already fully built, reusing stats", doc_name)
            return build.get("stats", {})

        stats: Dict[str, Any] = {
            "entity_count_before": 0, "entity_count_after": 0,
            "relation_count_before": 0, "relation_count_after": 0,
        }
        stats.update(build.get("stats", {}))
        errors: List[Dict] = []

        try:
            summary = await self._summary()
            stats["entity_count_before"] = summary.get("total_entities", 0)
            stats["relation_count_before"] = summary.get("total_relations", 0)

            safe_doc_name = doc_name.replace(" ", "_").replace(".", "_")
            if safe_doc_name:
                await self._mcp_session.call_tool("sysml_add_entity", {
                    "entity_type": "Package", "name": safe_doc_name,
                })

            tree_state = DocumentTreeState(rag_doc)
            logger.info("Document tree: %d leaf pages", tree_state.total_pages)

            if tree_state.total_pages == 0:
                logger.warning("No content pages found in document")
                return stats

            # ── Phase 1: 按小节分组提取 ──
            run_phase1 = phase in ("", "phase1")
            phase1_processed = set(build.get("phase1_processed_sections", []))
            if run_phase1:
                stats, errors_ph1 = await self._run_phase1(
                    rag_doc, doc_name, tree_state, stats, phase1_processed
                )
                errors.extend(errors_ph1)
                self._save_build_state(doc_name, "phase2", stats=stats, errors=errors)
            else:
                logger.info("Phase 1 already done (resuming from phase: %s)", phase)
                # Re-mark processed sections so tree_state reflects reality
                for nid in list(tree_state._all_page_ids):
                    if nid in tree_state.nodes:
                        parent_id = tree_state.nodes[nid].parent_id or "__root__"
                        if parent_id in phase1_processed:
                            tree_state.mark_processed(nid)

            # ── Phase 2: 跨章节去重 ──
            run_phase2 = phase in ("", "phase1", "phase2")
            if run_phase2:
                logger.debug("=== Phase 2 START: cross-section dedup ===")
                t2_start = time.time()
                await self._deduplicate_entities()
                await self._trigger_save()
                stats["phase2_time_s"] = round(time.time() - t2_start, 1)
                self._save_build_state(doc_name, "phase3", stats=stats, errors=errors)
                logger.debug("=== Phase 2 DONE (%.1fs) ===", stats["phase2_time_s"])

            # ── Phase 3: 并行 JSON 富化 ──
            run_phase3 = phase in ("", "phase1", "phase2", "phase3")
            if run_phase3:
                pre_enrich_summary = await self._summary()
                stats["pre_enrich_entities"] = pre_enrich_summary.get("total_entities", 0)
                stats["pre_enrich_relations"] = pre_enrich_summary.get("total_relations", 0)
                pre_orphan_count = await self._count_orphan_entities()
                stats["pre_enrich_orphans"] = pre_orphan_count
                logger.info("Pre-enrichment: %d entities, %d relations, %d orphans",
                             stats["pre_enrich_entities"], stats["pre_enrich_relations"],
                             pre_orphan_count)

                processed_sections = set(build.get("processed_sections", []))
                stats, errors_ph3 = await self._run_phase3(doc_name, processed_sections, stats)
                errors.extend(errors_ph3)
                self._save_build_state(doc_name, "phase4", stats=stats, errors=errors)
            else:
                logger.info("Phase 3 already done (resuming from phase: %s)", phase)

            # ── Phase 4: 图聚合 ──
            run_phase4 = phase in ("", "phase1", "phase2", "phase3", "phase4")
            if run_phase4:
                stats, errors_ph4 = await self._run_phase4(doc_name, stats)
                errors.extend(errors_ph4)

                comps_after = await self._mcp_session.call_tool("sysml_connected_components", {})
                try:
                    cc_data = json.loads(comps_after)
                    stats["final_components"] = cc_data.get("total_components", 0)
                    stats["final_component_sizes"] = [c.get("size", 0) for c in cc_data.get("components", [])[:5]]
                except Exception:
                    stats["final_components"] = -1
                logger.info("Post-aggregation: %d components", stats["final_components"])

            # ── Final Save ──
            await self._trigger_save()

            summary = await self._summary()
            stats["entity_count_after"] = summary.get("total_entities", 0)
            stats["relation_count_after"] = summary.get("total_relations", 0)
            stats["total_time_s"] = round(time.time(), 1)

            self._save_build_state(doc_name, "done", stats=stats, errors=errors if errors else None)
            logger.info("KG build complete: entities %d→%d, relations %d→%d",
                         stats["entity_count_before"], stats["entity_count_after"],
                         stats["relation_count_before"], stats["relation_count_after"])
            return stats

        except KeyboardInterrupt:
            logger.warning("Build interrupted by user (Ctrl+C), saving current state...")
            await self._trigger_save()
            self._save_build_state(doc_name, phase or "phase1", stats=stats, errors=errors)
            raise
        except Exception as e:
            logger.error("Build error for %s: %s\n%s", doc_name, e, traceback.format_exc())
            await self._trigger_save()
            errors.append({"phase": phase or "phase1", "error": str(e), "time": time.strftime("%H:%M:%S")})
            self._save_build_state(doc_name, phase or "phase1", stats=stats, errors=errors)
            raise

    async def _run_phase1(
        self, rag_doc, doc_name: str, tree_state, stats: Dict[str, Any],
        phase1_processed: set = None,
    ) -> tuple:
        """Phase 1: 并行 light_llm 逐节提取 + 顺序 MCP 创建实体。每节结束后保存。支持断点续跑。"""
        if phase1_processed is None:
            phase1_processed = set()
        errors: List[Dict] = []
        try:
            from tqdm import tqdm
            pbar = tqdm(total=tree_state.total_pages, desc="Phase 1: Section extraction",
                        unit="pg", ncols=120, file=sys.stderr,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        except ImportError:
            pbar = None

        page_count = 0
        entity_count = 0
        relation_count = 0
        t0 = time.time()

        section_pages: Dict[str, List[Any]] = {}
        section_order: list = []
        for nid in tree_state._all_page_ids:
            if nid not in tree_state.nodes:
                continue
            node = tree_state.nodes[nid]
            if not node.text.strip():
                tree_state.mark_processed(nid)
                continue
            parent_id = node.parent_id or "__root__"
            if parent_id not in section_pages:
                section_pages[parent_id] = []
                section_order.append(parent_id)
            section_pages[parent_id].append(node)

        batch_concurrency = self._get_config("BATCH_CONCURRENCY", 5)
        logger.info("Phase 1: section-based extraction, %d sections, concurrency=%d",
                     len(section_order), batch_concurrency)
        sem = asyncio.Semaphore(batch_concurrency)

        async def _extract_section(nodes: list) -> Any:
            async with sem:
                page_range = f"p{nodes[0].page_start}-{nodes[-1].page_start}"
                section_title = nodes[0].title[:40]
                text_parts = [
                    f"## 页面{nd.page_start}: {nd.title}\n{nd.text}"
                    for nd in nodes
                ]
                merged_text = "\n\n".join(text_parts)
                t_start = time.time()
                candidates, relations = await self._extract_page_candidates(
                    merged_text, section_title, nodes[0].page_start
                )
                elapsed = time.time() - t_start
                all_pages = [f"p{nd.page_start}" for nd in nodes]
                for c in candidates:
                    c.setdefault("source_pages", []).extend(all_pages)
                return (nodes, candidates, relations, elapsed)

        tasks = [_extract_section(section_pages[pid]) for pid in section_order]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        llm_time = time.time() - t0
        logger.info("Phase 1a (%d section LLM calls) done in %.0fs", len(results), llm_time)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Section extraction error: %s", result)
                errors.append({"section": "unknown", "phase": "phase1a", "error": str(result)})
                continue
            nodes, candidates, relations, elapsed = result
            n_pages = len(nodes)
            page_count += n_pages

            section_title = nodes[0].title[:40]
            section_key = tree_state.nodes[nodes[0].node_id].parent_id or "__root__"
            all_sections = [f"p{nd.page_start} {nd.title[:20]}" for nd in nodes]

            if section_key in phase1_processed:
                logger.info("Phase 1 skip [%s]: already processed", section_title)
                for nd in nodes:
                    tree_state.mark_processed(nd.node_id)
                if pbar:
                    pbar.update(n_pages)
                continue

            try:
                created = await self._process_candidates(
                    candidates, relations, doc_name,
                    nodes[0].page_start, section_title,
                    source_pages=all_sections,
                )
                entity_count += created["entities"]
                relation_count += created["relations"]
            except Exception as e:
                logger.error("Process candidates error for section %s: %s", section_title, e)
                errors.append({"section": section_title, "phase": "phase1b", "error": str(e)})
                created = {"entities": 0, "relations": 0}

            for nd in nodes:
                tree_state.mark_processed(nd.node_id)

            phase1_processed.add(section_key)

            # 每节结束后异步保存 + 更新断点续跑状态
            await self._trigger_save()
            try:
                self._save_build_state(
                    doc_name, "phase1",
                    phase1_processed_sections=list(phase1_processed),
                    stats=stats, errors=errors,
                )
            except Exception:
                pass

            if pbar:
                pbar.update(n_pages)
                pbar.set_postfix_str(
                    f"x{n_pages} {nodes[0].title[:15]} | +{created['entities']}e {created['relations']}r {elapsed:.0f}s"
                )
            else:
                logger.info(
                    "  Section [%d pages, p%d-p%d]: %d entities, %d relations in %.1fs",
                    n_pages, nodes[0].page_start, nodes[-1].page_start,
                    created["entities"], created["relations"], elapsed,
                )

        t1 = time.time()
        logger.info("Phase 1 complete: %d pages → %d entities, %d relations in %.0fs",
                     page_count, entity_count, relation_count, t1 - t0)
        stats["phase1_pages"] = page_count
        stats["phase1_entities"] = entity_count
        stats["phase1_relations"] = relation_count
        stats["phase1_time_s"] = round(t1 - t0, 1)
        await self._trigger_save()
        return stats, errors

    async def _run_phase3(
        self, doc_name: str, processed_sections: set, stats: Dict[str, Any]
    ) -> tuple:
        """Phase 3: 并行 JSON 模式富化。支持断点续跑（跳过已处理小节）。"""
        errors: List[Dict] = []
        t3_start = time.time()
        try:
            await self._enrich_entities(doc_name, processed_sections, errors)
        except Exception as e:
            logger.error("Phase 3 enrichment error: %s", e)
            errors.append({"phase": "phase3", "error": str(e)})
        stats["phase3_time_s"] = round(time.time() - t3_start, 1)

        post_enrich_summary = await self._summary()
        stats["post_enrich_entities"] = post_enrich_summary.get("total_entities", 0)
        stats["post_enrich_relations"] = post_enrich_summary.get("total_relations", 0)
        post_orphan_count = await self._count_orphan_entities()
        stats["post_enrich_orphans"] = post_orphan_count
        logger.info("Post-enrichment: %d entities, %d relations, %d orphans",
                     stats["post_enrich_entities"], stats["post_enrich_relations"],
                     post_orphan_count)
        await self._trigger_save()
        return stats, errors

    async def _run_phase4(
        self, doc_name: str, stats: Dict[str, Any]
    ) -> tuple:
        """Phase 4: 图聚合。每轮后保存。"""
        errors: List[Dict] = []
        t4_start = time.time()
        try:
            await self._aggregate_graph(doc_name)
        except Exception as e:
            logger.error("Phase 4 aggregation error: %s", e)
            errors.append({"phase": "phase4", "error": str(e)})
        stats["phase4_time_s"] = round(time.time() - t4_start, 1)
        await self._trigger_save()
        return stats, errors

    # ── Phase 1 helpers ──────────────────────────────────────

    async def _extract_page_candidates(
        self, text: str, title: str, page: int
    ) -> tuple:
        """用轻量模型从页面文本提取实体/关系候选列表 (JSON).
        支持滑动窗口处理长文本: >4000 字符时分块提取后合并去重。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        max_chars = 4000
        if len(text) <= max_chars:
            chunks = [text]
        else:
            # 滑动窗口: window=3000, stride=2500 (重叠500)
            window = min(3000, max_chars)
            stride = max(1, window - 500)
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + window, len(text))
                chunks.append(text[start:end])
                if end >= len(text):
                    break
                start += stride
            logger.debug("  Page %d: %d chars → %d chunks", page, len(text), len(chunks))

        all_candidates = []
        all_relations = []
        for ci, chunk in enumerate(chunks):
            suffix = f" (chunk {ci+1}/{len(chunks)})" if len(chunks) > 1 else ""
            prompt = f"""页面标题: {title} (第{page}页){suffix}

文本内容:
---BEGIN---
{chunk}
---END---

请输出JSON数组。无系统架构内容则输出 []"""
            try:
                t_call = time.time()
                response = await asyncio.wait_for(
                    self.light_llm.ainvoke([
                        SystemMessage(content=EXTRACTION_CANDIDATES_PROMPT),
                        HumanMessage(content=prompt),
                    ]),
                    timeout=180,
                )
                raw = str(response.content) if hasattr(response, "content") else str(response)
                logger.debug("  Phase1a LLM p%d c%d (%.1fs): prompt=%dch resp=%dch",
                             page, ci+1, time.time()-t_call, len(prompt), len(raw))
                logger.debug("  Phase1a RESP p%d c%d: %s", page, ci+1, raw[:500])
            except asyncio.TimeoutError:
                logger.warning("  Light LLM timeout for page %d chunk %d, retrying...", page, ci+1)
                try:
                    response = await asyncio.wait_for(
                        self.light_llm.ainvoke([
                            SystemMessage(content=EXTRACTION_CANDIDATES_PROMPT),
                            HumanMessage(content=prompt),
                        ]),
                        timeout=180,
                    )
                    raw = str(response.content) if hasattr(response, "content") else str(response)
                except asyncio.TimeoutError:
                    logger.warning("  Light LLM retry also timeout for page %d chunk %d, skipping", page, ci+1)
                    continue
            except Exception as e:
                logger.warning("  Light LLM error for page %d chunk %d: %s", page, ci+1, e)
                continue

            candidates, relations = self._parse_candidates(raw)
            all_candidates.extend(candidates)
            all_relations.extend(relations)

        # 合并去重: 同名实体保留最详细的一个
        if len(chunks) > 1:
            seen = {}
            merged_candidates = []
            for c in all_candidates:
                name = c.get("name", "")
                if name in seen:
                    existing = seen[name]
                    if len(c.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = c["description"]
                    existing_aliases = set(existing.get("aliases", []))
                    for a in c.get("aliases", []):
                        if a not in existing_aliases:
                            existing["aliases"].append(a)
                else:
                    seen[name] = dict(c)
                    merged_candidates.append(c)
            all_candidates = merged_candidates

            seen_rel = set()
            merged_relations = []
            for r in all_relations:
                key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
                if key not in seen_rel:
                    seen_rel.add(key)
                    merged_relations.append(r)
            all_relations = merged_relations

        if all_candidates or all_relations:
            logger.debug("  Page %d: %d entities, %d relations extracted (from %d chunks)",
                         page, len(all_candidates), len(all_relations), len(chunks))
        return all_candidates, all_relations

    @staticmethod
    def _parse_candidates(raw: str) -> tuple:
        """从 LLM 输出解析实体候选和关系候选."""
        import re
        candidates = []
        relations = []

        # 提取 JSON 块
        json_str = raw
        m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r'(\[.*\])', raw, re.DOTALL)
            if m:
                json_str = m.group(1)

        try:
            items = json.loads(json_str)
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("relation"):
                        relations.append(item)
                    elif item.get("name"):
                        candidates.append(item)
        except json.JSONDecodeError:
            # 逐行解析
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                m2 = re.search(r'\{.*\}', line)
                if m2:
                    try:
                        item = json.loads(m2.group())
                        if isinstance(item, dict):
                            if item.get("relation"):
                                relations.append(item)
                            elif item.get("name"):
                                candidates.append(item)
                    except json.JSONDecodeError:
                        pass

        return candidates, relations

    @staticmethod
    def _parse_enrichment_json(raw: str) -> list:
        """从 LLM 输出解析富化操作 JSON 数组."""
        import re
        json_str = raw
        m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r'(\[.*\])', raw, re.DOTALL)
            if m:
                json_str = m.group(1)
        try:
            items = json.loads(json_str)
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict) and i.get("action")]
        except json.JSONDecodeError:
            pass
        return []

    async def _process_candidates(
        self, candidates: list, relations: list,
        doc_name: str, page: int, section_title: str,
        source_pages: Optional[list] = None,
    ) -> dict:
        """系统直接调用 MCP 工具: 搜索去重 + 创建/更新实体和关系."""
        created_entities = 0
        created_relations = 0

        # 记录已有实体名 → QN 映射（避免重复创建）
        entity_qn_map: dict = {}
        pages_list = source_pages or [section_title or f"p{page}"]

        for c in candidates:
            name = str(c.get("name", "")).strip()
            etype = str(c.get("type", "PartDef")).strip()
            desc = str(c.get("description", "")).strip()
            aliases = c.get("aliases", []) or []

            if not name:
                continue

            # 搜索去重
            search_result = await self._mcp_session.call_tool(
                "sysml_search_entity",
                {"query": name, "regex_pattern": "", "threshold": 0.7},
            )
            try:
                search_data = json.loads(search_result)
            except json.JSONDecodeError:
                search_data = {}

            if search_data.get("total_matches", 0) > 0:
                # 已存在 → 追加来源页面
                match = search_data["matches"][0]
                existing_qn = match.get("qualified_name", name)
                entity_qn_map[name] = existing_qn
                logger.debug("  Phase1b: entity EXISTS '%s' (qn=%s, matched=%.2f)",
                             name, existing_qn, match.get("confidence", 0))
                # 追加新的来源页面
                await self._mcp_session.call_tool(
                    "sysml_update_entity",
                    {"qualified_name": existing_qn,
                     "append_source_sections": pages_list},
                )
                if aliases:
                    for alias in aliases:
                        await self._mcp_session.call_tool(
                            "sysml_add_alias",
                            {"qualified_name": existing_qn, "alias": alias},
                        )
                continue

            # 不存在 → 创建（记录所有来源页面）
            add_result = await self._mcp_session.call_tool(
                "sysml_add_entity", {
                    "entity_type": etype,
                    "name": name,
                    "parent_package": "",
                    "description": desc,
                    "aliases": aliases,
                    "source_sections": pages_list,
                    "source_text": desc,
                    "properties": {},
                    "supertypes": [],
                    "short_name": name,
                },
            )
            try:
                add_data = json.loads(add_result)
                if add_data.get("ok"):
                    created_entities += 1
                    entity_qn_map[name] = add_data.get("qualified_name", name)
                    logger.debug("  Phase1b: entity NEW '%s' <%s> (qn=%s)",
                                 name, etype, add_data.get("qualified_name", name))
            except json.JSONDecodeError:
                pass

        # 创建关系
        for r in relations:
            rtype = str(r.get("type", "Connection")).strip()
            source = str(r.get("source", "")).strip()
            target = str(r.get("target", "")).strip()
            desc = str(r.get("description", "")).strip()

            if not source or not target:
                continue

            # 解析端点名（可能已通过别名解析）
            src_qn = entity_qn_map.get(source, source)
            tgt_qn = entity_qn_map.get(target, target)

            rel_result = await self._mcp_session.call_tool(
                "sysml_add_relation", {
                    "relation_type": rtype.lower(),
                    "source": src_qn,
                    "target": tgt_qn,
                    "name": f"{source}_{target}_{rtype}",
                    "parent_package": "",
                    "description": desc,
                    "role_source": "",
                    "role_target": "",
                },
            )
            try:
                rel_data = json.loads(rel_result)
                if rel_data.get("ok"):
                    created_relations += 1
                    logger.debug("  Phase1b: relation NEW '%s' --[%s]--> '%s'",
                                 src_qn, rtype, tgt_qn)
            except json.JSONDecodeError:
                pass

        return {"entities": created_entities, "relations": created_relations}

    # ── Phase 3: Enrichment ──────────────────────────────────

    async def _enrich_entities(
        self, doc_name: str, processed_sections: set = None, errors: list = None
    ) -> None:
        """JSON 模式：并行 LLM 逐节生成富化指令 → 顺序 MCP 应用。支持断点续跑。"""
        if processed_sections is None:
            processed_sections = set()
        if errors is None:
            errors = []

        if self._mcp_session is None:
            return

        summary = await self._summary()
        total_entities = summary.get("total_entities", 0)
        if total_entities == 0:
            return

        entity_list = await self._mcp_session.call_tool(
            "sysml_list_entities", {"include_details": True}
        )
        try:
            entities_data = json.loads(entity_list)
        except json.JSONDecodeError:
            return

        entities = entities_data.get("entities", [])
        if not entities:
            return

        section_groups: dict = {}
        for e in entities:
            name = e.get("name", "")
            sections = e.get("source_sections") or e.get("source_section") or []
            if isinstance(sections, str):
                sections = [sections]
            if not sections:
                key = "__no_section__"
            else:
                first = str(sections[0])
                import re
                m = re.search(r'([\d]+\.[\d]+)', first)
                if m:
                    key = m.group(1)
                else:
                    key = first[:30]
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(e)

        logger.info("Enrichment: %d entities in %d sections",
                     len(entities), len(section_groups))

        batch_concurrency = self._get_config("BATCH_CONCURRENCY", 5)
        sem = asyncio.Semaphore(batch_concurrency)

        MAX_ENTITIES_PER_BATCH = 15

        async def _enrich_section(sec_key, sec_entities, doc_name, sem):
            async with sem:
                sec_names = [e.get("name", "") for e in sec_entities if e.get("name")]
                if not sec_names:
                    return None

                if sec_key in processed_sections:
                    logger.info("  Enrich section [%s]: SKIP (already processed)", sec_key)
                    return None

                entity_details = [{
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": (e.get("description") or "")[:200],
                } for e in sec_entities if e.get("name")]

                all_actions = []
                from langchain_core.messages import HumanMessage, SystemMessage

                # Split into batches to avoid overwhelming small model
                for batch_start in range(0, len(entity_details), MAX_ENTITIES_PER_BATCH):
                    batch = entity_details[batch_start:batch_start + MAX_ENTITIES_PER_BATCH]
                    chunk_label = f" ({batch_start//MAX_ENTITIES_PER_BATCH + 1}/{(len(entity_details)-1)//MAX_ENTITIES_PER_BATCH + 1})" if len(entity_details) > MAX_ENTITIES_PER_BATCH else ""
                    prompt = f"""文档: {doc_name}
小节: {sec_key}{chunk_label}
本小节实体列表:
{json.dumps(batch, ensure_ascii=False, indent=2)}

请为本小节实体输出JSON格式的富化操作。每个实体至少一条关系。
只在同小节实体之间创建关系。"""
                    try:
                        t_call = time.time()
                        response = await asyncio.wait_for(
                            self.light_llm.ainvoke([
                                SystemMessage(content=ENRICHMENT_JSON_PROMPT),
                                HumanMessage(content=prompt),
                            ]),
                            timeout=300,
                        )
                        raw = str(response.content) if hasattr(response, "content") else str(response)
                        logger.debug("  Phase3 LLM [%s] b%d (%.1fs): prompt=%dch resp=%dch",
                                     sec_key, batch_start//MAX_ENTITIES_PER_BATCH,
                                     time.time()-t_call, len(prompt), len(raw))
                        logger.debug("  Phase3 RESP [%s] b%d: %s", sec_key,
                                     batch_start//MAX_ENTITIES_PER_BATCH, raw[:500])
                    except asyncio.TimeoutError:
                        logger.warning("Enrichment LLM timeout for section [%s] batch %d", sec_key, batch_start//MAX_ENTITIES_PER_BATCH)
                        errors.append({"section": sec_key, "phase": "phase3", "error": "TimeoutError"})
                        continue
                    except Exception as e:
                        logger.warning("Enrichment LLM error for section [%s]: %s", sec_key, e)
                        errors.append({"section": sec_key, "phase": "phase3", "error": str(e)})
                        continue

                    batch_actions = self._parse_enrichment_json(raw) if raw else []
                    if isinstance(batch_actions, list):
                        all_actions.extend(batch_actions)

                if all_actions:
                    logger.info("  Enrich section [%s]: %d entities, %d actions generated",
                                 sec_key, len(sec_names), len(all_actions))
                return sec_key, all_actions

        section_items = sorted(section_groups.items())
        tasks = [_enrich_section(key, ents, doc_name, sem)
                  for key, ents in section_items]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if result is None or isinstance(result, Exception):
                if isinstance(result, Exception):
                    errors.append({"phase": "phase3", "error": str(result)})
                continue
            sec_key, actions = result
            if not actions:
                continue

            created_relations = 0
            for act in actions[:50]:
                action_type = act.get("action", "")
                try:
                    if action_type == "add_relation":
                        rel_type = act.get("type", "connection")
                        if rel_type not in ("connection", "interface", "allocation"):
                            rel_type = "connection"
                        rel_res = await self._mcp_session.call_tool("sysml_add_relation", {
                            "relation_type": rel_type,
                            "source": act.get("source", ""),
                            "target": act.get("target", ""),
                            "description": act.get("description", ""),
                        })
                        rdata = json.loads(rel_res)
                        if rdata.get("ok"):
                            created_relations += 1
                            logger.debug("  Phase3 REL: '%s' --[%s]--> '%s'",
                                         act.get("source", ""), rel_type, act.get("target", ""))
                    elif action_type == "add_alias":
                        await self._mcp_session.call_tool("sysml_add_alias", {
                            "qualified_name": act.get("entity", ""),
                            "alias": act.get("alias", ""),
                        })
                        logger.debug("  Phase3 ALIAS: '%s' ← '%s'",
                                     act.get("entity", ""), act.get("alias", ""))
                    elif action_type == "update_entity":
                        await self._mcp_session.call_tool("sysml_update_entity", {
                            "qualified_name": act.get("entity", ""),
                            "append_description": act.get("append_description", ""),
                        })
                except Exception as e:
                    logger.debug("Enrich action error: %s", e)

            logger.info("  Enrich section [%s]: %d relations created", sec_key, created_relations)
            processed_sections.add(sec_key)
            await self._trigger_save()
            try:
                self._save_build_state(
                    doc_name, "phase3", processed_sections=list(processed_sections))
            except Exception:
                pass

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

    async def _count_orphan_entities(self) -> int:
        """计算零关系实体的数量."""
        if self._mcp_session is None:
            return 0

        entity_list = await self._mcp_session.call_tool(
            "sysml_list_entities", {"include_details": False}
        )
        relation_list = await self._mcp_session.call_tool(
            "sysml_list_relations", {"include_details": True}
        )

        try:
            entities_data = json.loads(entity_list)
            relations_data = json.loads(relation_list)
        except json.JSONDecodeError:
            return 0

        entity_names = [e.get("name", "") for e in entities_data.get("entities", [])]
        connected: set = set()
        for rel in relations_data.get("relations", []):
            for end in rel.get("ends", []):
                ref = end.get("ref", "")
                if ref:
                    connected.add(ref)

        orphans = [n for n in entity_names if n not in connected]
        return len(orphans)

    # ── Phase 4: Graph Aggregation ─────────────────────────────

    async def _aggregate_graph(self, doc_name: str) -> None:
        """迭代合并连通分量，消除孤立子图，直到全图连通. 每轮后保存."""
        if self._mcp_session is None:
            return

        max_rounds = 20
        for round_idx in range(max_rounds):
            comps_result = await self._mcp_session.call_tool("sysml_connected_components", {})
            try:
                cc_data = json.loads(comps_result)
            except json.JSONDecodeError:
                break
            components = cc_data.get("components", [])
            if len(components) <= 1:
                logger.info("Phase 4: graph fully connected (%d component, round %d)",
                             len(components), round_idx + 1)
                await self._trigger_save()
                break

            smallest = components[0]
            target = components[-1] if len(components) >= 2 else components[0]

            logger.info("Phase 4 round %d: merging component size=%d into size=%d (%d total)",
                         round_idx + 1, smallest["size"], target["size"], len(components))

            small_entities = await self._get_entities_with_sections(smallest["entities"])
            target_entities = await self._get_entities_with_sections(target["entities"])

            bridge_candidates = []
            for s_name, s_sections in small_entities.items():
                for t_name, t_sections in target_entities.items():
                    common = set(s_sections) & set(t_sections)
                    if common:
                        bridge_candidates.append((s_name, t_name, list(common)[:5]))

            if bridge_candidates:
                logger.info("  Found %d bridge candidates via shared sections", len(bridge_candidates))
            else:
                bridge_candidates = [
                    (smallest["entities"][0], target["entities"][0], ["无共同章节"])
                ]

            created = await self._bridge_components(bridge_candidates)
            logger.info("  Created %d bridge relations", created)

            await self._trigger_save()

    async def _get_entities_with_sections(self, entity_names: list) -> dict:
        """获取实体的 source_sections 映射."""
        result = {}
        for name in entity_names:
            try:
                r = await self._mcp_session.call_tool(
                    "sysml_get_entity", {"entity_name": name}
                )
                data = json.loads(r)
                sections = data.get("source_sections", []) or []
                result[name] = [str(s) for s in sections if s]
            except Exception:
                result[name] = []
        return result

    async def _bridge_components(self, candidates: list) -> int:
        """并行 LLM 判断候选实体对是否存在有意义的关系并创建."""
        if not candidates:
            return 0

        model_name = self.light_model
        api_url = (config.settings.OPENAI_API_URL or "http://localhost:11434/v1").rstrip("/")
        batch_concurrency = self._get_config("BATCH_CONCURRENCY", 5)
        sem = asyncio.Semaphore(batch_concurrency)

        async def _classify_one(s_name, t_name, shared_sections):
            async with sem:
                prompt = f"""你是知识图谱关系审查员。判断以下两个实体之间是否存在有意义的关系。

实体A: {s_name}
实体B: {t_name}
共同出现的章节: {', '.join(shared_sections[:3])}

关系类型:
- allocation: A是B的一部分, B包含A, A属于B系统
- connection: A和B之间有物理连接或数据流
- None: 两者无直接关系

请只回答一个词: allocation, connection, 或 None"""
                try:
                    t_call = time.time()
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(
                            f"{api_url}/chat/completions",
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": prompt}],
                                "stream": False,
                                "options": {"num_predict": 8},
                            },
                        )
                        data = resp.json()
                        choice = data.get("choices", [{}])[0]
                        answer = (choice.get("message", {})
                                   .get("content", "")).strip().lower()
                        logger.debug("  Phase4 BRIDGE '%s' ↔ '%s' (%.1fs): %s",
                                     s_name, t_name, time.time()-t_call, answer or 'none')
                        return s_name, t_name, shared_sections, answer
                except Exception:
                    return s_name, t_name, shared_sections, None

        tasks = [_classify_one(s_name, t_name, shared)
                  for s_name, t_name, shared in candidates[:20]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        created = 0
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            s_name, t_name, shared_sections, answer = result
            if not answer or answer in ("none", "无", ""):
                continue

            rel_type = "allocation" if "allocation" in answer else "connection"
            try:
                rel_result = await self._mcp_session.call_tool(
                    "sysml_add_relation", {
                        "relation_type": rel_type,
                        "source": s_name,
                        "target": t_name,
                        "description": f"桥接关系: {s_name} ↔ {t_name} (同章: {', '.join(shared_sections[:2])})",
                    }
                )
                rel_data = json.loads(rel_result)
                if rel_data.get("ok"):
                    created += 1
                    logger.debug("  Bridge: %s --[%s]--> %s", s_name, rel_type, t_name)
            except Exception:
                pass

        return created

    # ── Light model unload ─────────────────────────────────────

    async def _unload_light_model(self) -> None:
        """显式卸载小模型以释放显存，为大模型富化阶段腾出空间."""
        base_url = config.settings.OPENAI_API_URL or "http://localhost:11434/v1"
        ollama_url = base_url.rstrip("/").rsplit("/v1", 1)[0]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 检查模型是否已加载
                ps_resp = await client.get(f"{ollama_url}/api/ps")
                if ps_resp.status_code == 200:
                    loaded_models = [m.get("name", "") for m in
                                     ps_resp.json().get("models", [])]
                    if self.light_model not in loaded_models:
                        logger.info("Light model '%s' already unloaded", self.light_model)
                        return

                # 发送 keep_alive=0 请求卸载模型（限制输出1 token，最少消耗）
                resp = await client.post(f"{ollama_url}/api/generate", json={
                    "model": self.light_model,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False,
                    "options": {"num_predict": 1},
                })
                if resp.status_code == 200:
                    logger.info("Light model '%s' unloaded (keep_alive=0)", self.light_model)
                else:
                    logger.warning("Light model unload returned %d: %s",
                                   resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("Light model unload skipped (not critical): %s", e)
        else:
            await asyncio.sleep(1)

    # ── Cross-section dedup ────────────────────────────────────

    async def _deduplicate_entities(self) -> None:
        """跨章节实体去重：获取合并建议 → 直接 MCP 合并（无 LLM，<1s）"""
        if self._mcp_session is None:
            return

        suggest_result = await self._mcp_session.call_tool(
            "sysml_suggest_merge", {"threshold": 0.5}
        )
        suggestions = json.loads(suggest_result)
        if not suggestions.get("ok") or not suggestions.get("suggestions"):
            logger.info("No merge suggestions found")
            return

        items = sorted(suggestions["suggestions"], key=lambda x: -x["confidence"])
        logger.info("Found %d merge suggestions, processing top 60", len(items))
        merged = 0

        for i, item in enumerate(items[:60]):
            a_qn = item["entity_a"]["qualified_name"]
            b_qn = item["entity_b"]["qualified_name"]
            a_name = item["entity_a"]["name"]
            b_name = item["entity_b"]["name"]
            conf = item["confidence"]

            try:
                result = json.loads(await self._mcp_session.call_tool(
                    "sysml_merge_entities", {"source": a_qn, "target": b_qn},
                ))
                if result.get("ok"):
                    merged += 1
                    if i < 10 or merged % 20 == 0:
                        logger.debug("  Merged %s → %s (%.2f)", a_name, b_name, conf)
            except Exception as e:
                logger.debug("  Merge skip %s→%s: %s", a_name, b_name, e)

        logger.info("Dedup complete: %d pairs merged", merged)

    # ── Persistence ────────────────────────────────────────────

    @property
    def _build_state_file(self) -> Path:
        return self.persist_dir / "knowledge_graph.build.json"

    def _load_build_state(self, doc_name: str) -> Dict[str, Any]:
        try:
            bsf = self._build_state_file
            if bsf.exists():
                with open(bsf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                docs = data.get("documents", {})
                if doc_name in docs:
                    return docs[doc_name]
        except Exception:
            pass
        return {}

    def _save_build_state(
        self, doc_name: str, phase: str,
        processed_sections: Optional[List[str]] = None,
        phase1_processed_sections: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
        errors: Optional[List[Dict]] = None,
    ) -> None:
        try:
            bsf = self._build_state_file
            data = {}
            if bsf.exists():
                with open(bsf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            docs = data.get("documents", {})
            entry = docs.get(doc_name, {})
            entry |= {"phase": phase, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if processed_sections is not None:
                entry["processed_sections"] = processed_sections
            if phase1_processed_sections is not None:
                entry["phase1_processed_sections"] = phase1_processed_sections
            if stats is not None:
                entry.setdefault("stats", {}).update(stats)
            if errors is not None:
                entry.setdefault("errors", []).extend(errors)
            docs[doc_name] = entry
            data["documents"] = docs
            data["version"] = 1
            tmp = bsf.with_suffix(".tmp")
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(bsf)
        except Exception as e:
            logger.warning("Failed to save build state: %s", e)

    async def _trigger_save(self) -> None:
        self._save_desired = True
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._do_save())

    async def _do_save(self) -> None:
        async with self._save_lock:
            while self._save_desired:
                self._save_desired = False
                await self._save_knowledge_graph()

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

    # ═══════════════════════════════════════════════════════════════
    # 递归级联 KG 构建 (cascading-recursive pipeline)
    # ═══════════════════════════════════════════════════════════════

    async def build_kg_recursive(self, rag_doc) -> Dict[str, Any]:
        """
        递归级联知识图谱构建管线:
        Phase 0: gemma4:31b 识别根实体和起点小节
        Phase 1: gemma4:31b 根小节完整提取
        Phase 2: 实体传播 → 全文搜索 → 构建队列
        Phase 3: 级联队列处理 (qwen3:8b)
        循环 Phase 2→3 直到队列为空
        最后: 跨章节去重 + 图聚合

        支持断点续跑: 完整序列化 build_queue/processed_pairs/实体队列状态
        """
        await self.initialize()

        doc_name = getattr(rag_doc, "doc_name", "")
        build = self._load_build_state(doc_name)
        pipeline = build.get("pipeline", "")
        phase = build.get("phase", "")

        if phase == "done" and pipeline == "recursive":
            logger.info("Document %s already fully built (recursive), reusing stats", doc_name)
            return build.get("stats", {})

        stats: Dict[str, Any] = {
            "entity_count_before": 0, "entity_count_after": 0,
            "relation_count_before": 0, "relation_count_after": 0,
            "pipeline": "recursive",
        }
        stats.update(build.get("stats", {}))

        try:
            summary = await self._summary()
            stats["entity_count_before"] = summary.get("total_entities", 0)
            stats["relation_count_before"] = summary.get("total_relations", 0)

            safe_doc_name = doc_name.replace(" ", "_").replace(".", "_")
            if safe_doc_name:
                await self._mcp_session.call_tool("sysml_add_entity", {
                    "entity_type": "Package", "name": safe_doc_name,
                })

            tree_state = DocumentTreeState(rag_doc)
            logger.info("Recursive: %d leaf pages in document tree", tree_state.total_pages)

            if tree_state.total_pages == 0:
                logger.warning("No content pages found in document")
                return stats

            # 构建 sections 映射 (parent_id → [nodes])
            sections = self._build_section_map(tree_state)

            # ── Phase 0: 根实体识别 ──
            root_entity = build.get("root_entity")
            start_sections = build.get("start_sections", [])
            if not root_entity or not start_sections:
                root_entity, start_sections = await self._identify_root(
                    rag_doc, tree_state, sections, doc_name
                )
                self._save_recursive_state(doc_name, "recursive_phase1",
                    root_entity=root_entity, start_sections=start_sections, stats=stats)
                logger.info("Phase 0 done: root=%s, start_sections=%s",
                             root_entity.get("name", "?"), start_sections)
            else:
                logger.info("Phase 0 already done (resume): root=%s", root_entity.get("name", "?"))

            # ── Phase 1: 根小节处理 ──
            processed_sections = set(build.get("processed_sections", []))
            if not processed_sections:
                new_entity_names = await self._process_root_sections(
                    tree_state, sections, root_entity, start_sections, doc_name
                )
                for sid in start_sections:
                    processed_sections.add(sid)
                self._save_recursive_state(doc_name, "recursive_phase2",
                    root_entity=root_entity, start_sections=start_sections,
                    processed_sections=list(processed_sections),
                    stats=stats)
                logger.info("Phase 1 done: %d entities discovered", len(new_entity_names))
            else:
                logger.info("Phase 1 already done: %d processed sections", len(processed_sections))
                new_entity_names = []  # will be rebuilt from Phase 2 search

            # ── 恢复队列 ──
            build_queue = deque()
            for pair in build.get("build_queue", []):
                build_queue.append(tuple(pair))
            processed_pairs = set(
                tuple(p) for p in build.get("processed_pairs", [])
            )
            section_entity_queue = deque(build.get("current_section_entity_queue", []))
            current_section = build.get("current_section")

            t0 = time.time()

            # ── Phase 2 → Phase 3 循环 ──
            iteration = 0
            max_iterations = 500
            while iteration < max_iterations:
                iteration += 1
                logger.debug("Recursive loop iteration %d: build_queue=%d, processed=%d",
                              iteration, len(build_queue), len(processed_sections))

                if not build_queue and not section_entity_queue:
                    # Phase 2: 传播新实体到未处理小节
                    if new_entity_names:
                        build_queue = await self._propagate_entities(
                            tree_state, sections, new_entity_names,
                            processed_sections, processed_pairs, build_queue, doc_name
                        )
                        new_entity_names = []
                        self._save_recursive_state(doc_name, "recursive_phase3",
                            root_entity=root_entity, start_sections=start_sections,
                            processed_sections=list(processed_sections),
                            build_queue=list(build_queue),
                            processed_pairs=[list(p) for p in processed_pairs],
                            stats=stats)
                        logger.info("Phase 2: queue size=%d after propagation", len(build_queue))

                    if not build_queue:
                        logger.info("Recursive pipeline: queue empty, build complete")
                        break
                    continue

                # Phase 3: 处理队列
                if section_entity_queue:
                    # 小节内实体级联
                    entity_name = section_entity_queue.popleft()
                    section_id = current_section
                    logger.info("Phase 3: intra-section cascade entity=%s section=%s (queue_remaining=%d)",
                                 entity_name, section_id, len(section_entity_queue))
                else:
                    # 从 build_queue 取新配对
                    entity_name, section_id = build_queue.popleft()
                    current_section = section_id
                    logger.info("Phase 3: new pair entity=%s section=%s (queue_remaining=%d)",
                                 entity_name, section_id, len(build_queue))

                pair = (entity_name, section_id)
                if pair in processed_pairs:
                    logger.debug("Phase 3: skip already processed pair %s", pair)
                    continue

                # 处理小节（如果尚未处理）
                if section_id not in processed_sections:
                    section_nodes = sections.get(section_id, [])
                    if section_nodes:
                        new_ents = await self._process_queued_section(
                            section_nodes, entity_name, section_id, doc_name
                        )
                        new_entity_names.extend(new_ents)
                        if not section_entity_queue:
                            for nd in section_nodes:
                                tree_state.mark_processed(nd.node_id)
                            processed_sections.add(section_id)
                            current_section = None
                            logger.info("Phase 3: section %s fully done, %d new entities",
                                         section_id, len(new_ents))
                    else:
                        processed_sections.add(section_id)

                processed_pairs.add(pair)

                if iteration % 10 == 0:
                    await self._trigger_save()
                    self._save_recursive_state(doc_name, phase or "recursive_phase3",
                        root_entity=root_entity, start_sections=start_sections,
                        processed_sections=list(processed_sections),
                        build_queue=list(build_queue),
                        processed_pairs=[list(p) for p in processed_pairs],
                        current_section=current_section,
                        current_section_entity_queue=list(section_entity_queue),
                        stats=stats)

            # ── 收尾: 跨章节去重 ──
            logger.info("Recursive pipeline: deduplicating across sections...")
            await self._deduplicate_entities()
            await self._trigger_save()
            stats["phase_dedup_time_s"] = round(time.time() - t0, 1)

            # ── 图聚合 ──
            logger.info("Recursive pipeline: aggregating graph...")
            await self._aggregate_graph(doc_name)
            await self._trigger_save()

            summary = await self._summary()
            stats["entity_count_after"] = summary.get("total_entities", 0)
            stats["relation_count_after"] = summary.get("total_relations", 0)
            stats["total_time_s"] = round(time.time() - t0, 1)

            self._save_recursive_state(doc_name, "done",
                root_entity=root_entity, start_sections=start_sections,
                processed_sections=list(processed_sections),
                stats=stats, errors=None)
            logger.info("Recursive KG build complete: entities %d→%d, relations %d→%d",
                         stats["entity_count_before"], stats["entity_count_after"],
                         stats["relation_count_before"], stats["relation_count_after"])
            return stats

        except KeyboardInterrupt:
            logger.warning("Recursive build interrupted, saving state...")
            await self._trigger_save()
            self._save_recursive_state(doc_name, "recursive_phase3",
                root_entity=build.get("root_entity", {}),
                start_sections=build.get("start_sections", []),
                processed_sections=list(processed_sections) if "processed_sections" in dir() else [],
                build_queue=list(build_queue) if "build_queue" in dir() else [],
                processed_pairs=[list(p) for p in processed_pairs] if "processed_pairs" in dir() else [],
                stats=stats)
            raise
        except Exception as e:
            logger.error("Recursive build error: %s\n%s", e, traceback.format_exc())
            await self._trigger_save()
            raise

    def _build_section_map(self, tree_state: DocumentTreeState) -> Dict[str, List[DocTreeNode]]:
        """构建 parent_id → [nodes] 映射（按小节分组）"""
        sections: Dict[str, List[DocTreeNode]] = {}
        for nid in tree_state._all_page_ids:
            if nid not in tree_state.nodes:
                continue
            node = tree_state.nodes[nid]
            parent_id = node.parent_id or "__root__"
            sections.setdefault(parent_id, []).append(node)
        return sections

    async def _identify_root(
        self, rag_doc, tree_state: DocumentTreeState,
        sections: Dict[str, List], doc_name: str
    ) -> tuple:
        """Phase 0: 用 gemma4:31b 识别根实体和起点小节"""
        from langchain_core.messages import HumanMessage, SystemMessage

        doc_title = getattr(rag_doc, "title", "") or doc_name
        tree_desc = json.dumps(tree_state.get_tree_structure(), ensure_ascii=False, indent=2)

        prompt = f"""你是技术文档分析专家。分析以下文档的目录结构，识别根实体和最佳构建起点。

## 文档信息
- 标题: {doc_title}

## 文档目录结构
{tree_desc}

## 任务
1. **根实体识别**: 文档描述的核心实体是什么？（文档主题：某个系统、设备、项目？）
   - name: 根实体名称（用文档中最正式的称谓）
   - type: SysML实体类型 (PartDef|ItemDef|Package)
   - description: 一句话描述

2. **起点小节选择**: 从上述章节中选择2-4个小节作为构建起点（对根实体描述最集中的小节）
   - start_sections: node_id 列表，如 ["n5", "n8"]

## 输出格式（仅JSON）"""
        prompt += """
{"root_entity":{"name":"...","type":"PartDef","description":"..."},"start_sections":["n5"],"reasoning":"..."}"""

        logger.debug("Phase 0: identifying root entity via %s", self.model)
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([
                    SystemMessage(content="你是技术文档分析专家。仅输出JSON，无其他内容。"),
                    HumanMessage(content=prompt),
                ]),
                timeout=180,
            )
            raw = str(response.content) if hasattr(response, "content") else str(response)
            logger.debug("Phase 0 LLM response: %s", raw[:500])

            result = self._parse_json_response(raw)
            root = result.get("root_entity", {})
            starts = result.get("start_sections", [])

            if not root.get("name"):
                root = {"name": doc_title or doc_name, "type": "PartDef",
                         "description": f"文档:{doc_title or doc_name}"}
            if not starts:
                starts = list(sections.keys())[:3]

            return root, starts
        except asyncio.TimeoutError:
            logger.warning("Phase 0 timed out, using defaults")
            return (
                {"name": doc_title or doc_name, "type": "PartDef",
                 "description": f"Document root entity for {doc_name}"},
                list(sections.keys())[:3],
            )
        except Exception as e:
            logger.error("Phase 0 LLM error: %s", e)
            return (
                {"name": doc_title or doc_name, "type": "PartDef",
                 "description": f"Document root entity for {doc_name}"},
                list(sections.keys())[:3],
            )

    async def _process_root_sections(
        self, tree_state: DocumentTreeState,
        sections: Dict[str, List], root_entity: Dict[str, Any],
        start_section_ids: List[str], doc_name: str,
    ) -> List[str]:
        """Phase 1: 用 gemma4:31b 提取根小节中的全部实体和关系"""
        from langchain_core.messages import HumanMessage, SystemMessage

        all_new_entity_names: List[str] = []
        root_name = root_entity.get("name", "")
        root_type = root_entity.get("type", "PartDef")
        root_desc = root_entity.get("description", "")

        for sid in start_section_ids:
            nodes = sections.get(sid, [])
            if not nodes:
                continue

            section_title = nodes[0].title[:60]
            text_parts = [
                f"## 页面{nd.page_start}: {nd.title}\n{nd.text}"
                for nd in nodes
            ]
            merged_text = "\n\n".join(text_parts)

            prompt = f"""你是SysML v2知识图谱专家。从文档小节中提取**所有**系统架构实体和关系。

## 文档根实体
- 名称: {root_name}
- 类型: {root_type}
- 描述: {root_desc}

## 当前小节
- ID: {sid}
- 标题: {section_title}

## 可用实体类型
PartDef(系统组件/模块/设备/机柜/服务器), AttributeDef(属性/参数/指标), PortDef(接口/端口), ItemDef(数据结构/信息流), RequirementDef(需求/约束), CommandDef(Shell命令/CLI操作), InterfaceDef, ConnectionDef

## 关系类型
Connection(物理连接/数据流), Interface(接口实现), Allocation(功能/资源分配)

## 小节内容
{merged_text}

## 输出格式 (仅JSON数组，无Markdown包裹)"""
            prompt += f"""
[
  {{"type":"PartDef","name":"实体名","description":"简短描述","aliases":["别名"],"source_section":"{section_title}"}},
  {{"type":"Connection","source":"源","target":"目标","description":"关系描述","source_section":"{section_title}"}}
]
无相关内容输出 []"""

            logger.debug("Phase 1: extracting root section %s via %s", sid, self.model)
            try:
                t_call = time.time()
                response = await asyncio.wait_for(
                    self.llm.ainvoke([
                        SystemMessage(content="你是SysML v2知识图谱专家。仅输出JSON数组，无其他内容。"),
                        HumanMessage(content=prompt),
                    ]),
                    timeout=300,
                )
                raw = str(response.content) if hasattr(response, "content") else str(response)
                logger.debug("Phase 1 LLM (root section %s, %.1fs): %s",
                              sid, time.time() - t_call, raw[:500])
            except asyncio.TimeoutError:
                logger.warning("Phase 1 timeout for root section %s", sid)
                continue
            except Exception as e:
                logger.error("Phase 1 LLM error for section %s: %s", sid, e)
                continue

            candidates, relations = self._parse_candidates(raw)

            # 记录新实体名
            for c in candidates:
                name = c.get("name", "").strip()
                if name:
                    all_new_entity_names.append(name)

            # 通过 MCP 创建实体和关系
            source_pages = [f"p{nd.page_start} {section_title}" for nd in nodes]
            section_key = section_title if section_title else f"Section {sid}"
            await self._process_candidates(
                candidates, relations, doc_name,
                nodes[0].page_start, section_key,
                source_pages=source_pages,
            )

            for nd in nodes:
                tree_state.mark_processed(nd.node_id)

            await self._trigger_save()

        return all_new_entity_names

    async def _propagate_entities(
        self, tree_state: DocumentTreeState,
        sections: Dict[str, List], new_entity_names: List[str],
        processed_sections: Set[str], processed_pairs: Set[Tuple[str, str]],
        build_queue: deque, doc_name: str,
    ) -> deque:
        """Phase 2: 全文搜索实体在新小节的提及，构建 (entity, section) 队列"""
        for entity_name in new_entity_names:
            if not entity_name:
                continue

            aliases: List[str] = []
            try:
                detail_raw = await self._mcp_session.call_tool(
                    "sysml_get_entity", {"entity_name": entity_name}
                )
                detail = json.loads(detail_raw)
                aliases = detail.get("aliases", []) or []
            except Exception:
                pass

            for section_id, nodes in sections.items():
                if section_id in processed_sections:
                    continue

                pair = (entity_name, section_id)
                if pair in processed_pairs:
                    continue

                already_queued = any(
                    qe == entity_name and qs == section_id
                    for qe, qs in build_queue
                )
                if already_queued:
                    continue

                found = False
                search_names = [entity_name] + aliases
                for nd in nodes:
                    text = nd.text
                    for sn in search_names:
                        if sn and len(sn) >= 2 and sn.lower() in text.lower():
                            found = True
                            break
                    if found:
                        break

                if found:
                    build_queue.append((entity_name, section_id))
                    logger.debug("  Phase 2: entity=%s found in section=%s (aliases=%s)",
                                  entity_name, section_id, aliases[:3])

        return build_queue

    async def _process_queued_section(
        self, nodes: List[DocTreeNode], entity_name: str,
        section_id: str, doc_name: str,
    ) -> List[str]:
        """Phase 3: 用 qwen3:8b 提取小节中与焦点实体相关的实体"""
        from langchain_core.messages import HumanMessage, SystemMessage

        section_title = nodes[0].title[:60]
        text_parts = [
            f"## 页面{nd.page_start}: {nd.title}\n{nd.text}"
            for nd in nodes
        ]
        merged_text = "\n\n".join(text_parts)

        prompt = f"""你是SysML v2知识图谱专家。从文档小节中提取与焦点实体相关的新知识。

## 焦点实体
- 名称: {entity_name}

## 当前小节
- 标题: {section_title}

## 实体类型
PartDef|AttributeDef|PortDef|ItemDef|RequirementDef|CommandDef|InterfaceDef|ConnectionDef

## 关系类型
Connection|Interface|Allocation

## 小节内容
{merged_text}

## 输出格式 (仅JSON数组)"""
        prompt += f"""
[
  {{"type":"PartDef","name":"实体名","description":"简短描述","aliases":["别名"],"source_section":"{section_title}"}},
  {{"type":"Connection","source":"源","target":"目标","description":"关系描述","source_section":"{section_title}"}}
]
无相关内容输出 []"""

        try:
            t_call = time.time()
            response = await asyncio.wait_for(
                self.light_llm.ainvoke([
                    SystemMessage(content="你是SysML v2知识图谱专家。仅输出JSON数组，无其他内容。"),
                    HumanMessage(content=prompt),
                ]),
                timeout=180,
            )
            raw = str(response.content) if hasattr(response, "content") else str(response)
            logger.debug("Phase 3 LLM (section %s, entity %s, %.1fs): %s",
                          section_id, entity_name, time.time() - t_call, raw[:400])
        except asyncio.TimeoutError:
            logger.warning("Phase 3 timeout for section %s", section_id)
            return []
        except Exception as e:
            logger.error("Phase 3 LLM error for section %s: %s", section_id, e)
            return []

        candidates, relations = self._parse_candidates(raw)
        new_entity_names = [c.get("name", "").strip() for c in candidates if c.get("name", "").strip()]

        # 创建实体和关系
        source_pages = [f"p{nd.page_start} {section_title}" for nd in nodes]
        section_key = section_title if section_title else f"Section {section_id}"
        await self._process_candidates(
            candidates, relations, doc_name,
            nodes[0].page_start, section_key,
            source_pages=source_pages,
        )

        return new_entity_names

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[str, Any]:
        """从 LLM 输出解析 JSON 对象（非数组）"""
        import re
        json_str = raw.strip()
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r'(\{.*\})', raw, re.DOTALL)
            if m:
                json_str = m.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    def _save_recursive_state(
        self, doc_name: str, phase: str,
        root_entity: Optional[Dict] = None,
        start_sections: Optional[List[str]] = None,
        processed_sections: Optional[List[str]] = None,
        build_queue: Optional[List[List]] = None,
        processed_pairs: Optional[List[List]] = None,
        current_section: Optional[str] = None,
        current_section_entity_queue: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
        errors: Optional[List[Dict]] = None,
    ) -> None:
        """扩展的断点续跑状态保存（含队列状态）"""
        try:
            bsf = self._build_state_file
            data = {}
            if bsf.exists():
                with open(bsf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            docs = data.get("documents", {})
            entry = docs.get(doc_name, {})
            entry.update({
                "phase": phase, "pipeline": "recursive",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            if root_entity is not None:
                entry["root_entity"] = root_entity
            if start_sections is not None:
                entry["start_sections"] = start_sections
            if processed_sections is not None:
                entry["processed_sections"] = processed_sections
            if build_queue is not None:
                entry["build_queue"] = build_queue
            if processed_pairs is not None:
                entry["processed_pairs"] = processed_pairs
            if current_section is not None:
                entry["current_section"] = current_section
            if current_section_entity_queue is not None:
                entry["current_section_entity_queue"] = current_section_entity_queue
            if stats is not None:
                entry.setdefault("stats", {}).update(stats)
            if errors is not None:
                entry.setdefault("errors", []).extend(errors)

            docs[doc_name] = entry
            data["documents"] = docs
            data["version"] = 2
            tmp = bsf.with_suffix(".tmp")
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(bsf)
        except Exception as e:
            logger.warning("Failed to save recursive build state: %s", e)


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
