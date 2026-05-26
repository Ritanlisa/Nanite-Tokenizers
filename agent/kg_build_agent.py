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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ── Enrichment Prompt (qwen3-vl:32b — adds aliases, properties, relations) ──

ENRICHMENT_PROMPT = """你是SysML v2知识图谱专家。审查已有的KG实体列表，重点添加关系和别名。

## ⚠️ 核心任务：关系创建（最重要）
每个实体应与系统中其他实体至少有一条关系。
- 对每对相关实体，判断关系类型并创建
- 关系类型: connection(物理连接), interface(接口), allocation(包含/分配)
- 示例: FT计算柜 allocation 计算处理分系统 (FT柜是计算系统的一部分)
- 示例: 交换柜 connection 交换设备 (交换设备安装在交换柜中)
- 示例: yhst allocation CMU (yhst命令在CMU上执行)
- 示例: 处理器数量 allocation 双路服务器 (服务器有处理器数量属性)
- **多创建关系，宁可冗余不要遗漏**

## 可用工具及精确参数

### mcp__sysml_get_entity
获取实体详情: entity_name(必填)

### mcp__sysml_add_alias
添加别名: qualified_name(必填), alias(必填)

### mcp__sysml_add_relation
创建关系: relation_type(必填), source(必填), target(必填)
  可选: name, parent_package, description, role_source, role_target

### mcp__sysml_update_entity
更新实体: qualified_name(必填)
  可选: append_description, update_properties

## 工作规则
1. 用 mcp__sysml_get_entity 了解实体
2. **重点: 为每对相关实体创建关系**
3. 补充别名是中英文通用的
4. 属性/端口类实体应关联到所属的 PartDef"""

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
        self.model = model or self._get_config("KG_EXTRACTION_MODEL", "qwen3-vl:32b")
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
                model_kwargs={"keep_alive": self._get_config("KG_KEEP_ALIVE", "30s")},
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
                timeout=120,
                streaming=False,
                model_kwargs={"keep_alive": self._get_config("KG_KEEP_ALIVE", "30s")},
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

        三阶段并行架构:
        1. 并行 light_llm 逐页扫描（Semaphore 控制并发，默认5路并发）
        2. 系统直接调 MCP — 搜索去重 + 创建实体/关系（顺序，无 LLM）
        3. 卸载小模型 → 强模型（llm）全局审查，补别名/属性/关系
        keep_alive=30s 确保小模型空闲后自动卸载，释放显存
        """
        await self.initialize()

        doc_name = getattr(rag_doc, "doc_name", "")
        stats = {
            "entity_count_before": 0, "entity_count_after": 0,
            "relation_count_before": 0, "relation_count_after": 0,
        }

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

        # ── Phase 1: 并行页提取 ──
        try:
            from tqdm import tqdm
            pbar = tqdm(total=tree_state.total_pages, desc="Phase 1: Page extraction",
                        unit="pg", ncols=120, file=sys.stderr,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        except ImportError:
            pbar = None

        page_count = 0
        entity_count = 0
        relation_count = 0
        t0 = time.time()

        batch_concurrency = self._get_config("BATCH_CONCURRENCY", 5)
        logger.info("Phase 1: parallel light_llm extraction with concurrency=%d",
                     batch_concurrency)
        sem = asyncio.Semaphore(batch_concurrency)

        async def _extract_one(nid: str) -> Any:
            async with sem:
                if nid not in tree_state.nodes:
                    return None
                node = tree_state.nodes[nid]
                if not node.text.strip():
                    tree_state.mark_processed(nid)
                    return None
                t_start = time.time()
                candidates, relations = await self._extract_page_candidates(
                    node.text, node.title, node.page_start
                )
                elapsed = time.time() - t_start
                return (nid, node, candidates, relations, elapsed)

        tasks = [_extract_one(nid) for nid in tree_state._all_page_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        llm_time = time.time() - t0
        logger.info("Phase 1a (%d parallel LLM calls) done in %.0fs", len(results), llm_time)

        for result in results:
            if result is None:
                if pbar:
                    pbar.update(1)
                continue
            if isinstance(result, Exception):
                logger.error("Phase 1 LLM extraction error: %s", result)
                if pbar:
                    pbar.update(1)
                continue

            nid, node, candidates, relations, elapsed = result
            page_count += 1

            created = await self._process_candidates(
                candidates, relations, doc_name, node.page_start, node.title
            )
            entity_count += created["entities"]
            relation_count += created["relations"]
            tree_state.mark_processed(nid)

            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"p{node.page_start} {node.title[:15]} | +{created['entities']}e {created['relations']}r {elapsed:.0f}s"
                )
            elif page_count % 5 == 0 or page_count <= 3:
                logger.info(
                    "  Page %d/%d [p%d %s]: %d entities, %d relations in %.1fs",
                    page_count, tree_state.total_pages,
                    node.page_start, node.title[:30],
                    created["entities"], created["relations"], elapsed,
                )

        if pbar:
            pbar.close()

        t1 = time.time()
        logger.info("Phase 1 complete: %d pages → %d entities, %d relations in %.0fs",
                     page_count, entity_count, relation_count, t1 - t0)

        # Checkpoint save after Phase 1
        await self._save_knowledge_graph()
        logger.info("Phase 1 KG checkpoint saved (%d entities, %d relations)",
                     entity_count, relation_count)

        # ── Phase 2: 跨章节去重 ──
        await self._deduplicate_entities()

        # ── 卸载小模型，释放显存给大模型 ──
        await self._unload_light_model()

        # ── Phase 3: 强模型补充别名/属性/关系 ──
        await self._enrich_entities(doc_name)

        # ── Save ──
        await self._save_knowledge_graph()

        summary = await self._summary()
        stats["entity_count_after"] = summary.get("total_entities", 0)
        stats["relation_count_after"] = summary.get("total_relations", 0)

        logger.info("KG build complete: entities %d→%d, relations %d→%d",
                     stats["entity_count_before"], stats["entity_count_after"],
                     stats["relation_count_before"], stats["relation_count_after"])
        return stats

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
                response = await asyncio.wait_for(
                    self.light_llm.ainvoke([
                        SystemMessage(content=EXTRACTION_CANDIDATES_PROMPT),
                        HumanMessage(content=prompt),
                    ]),
                    timeout=180,
                )
                raw = str(response.content) if hasattr(response, "content") else str(response)
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

    async def _process_candidates(
        self, candidates: list, relations: list,
        doc_name: str, page: int, section_title: str,
    ) -> dict:
        """系统直接调用 MCP 工具: 搜索去重 + 创建/更新实体和关系."""
        created_entities = 0
        created_relations = 0

        # 记录已有实体名 → QN 映射（避免重复创建）
        entity_qn_map: dict = {}

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
                # 已存在 → 更新
                match = search_data["matches"][0]
                existing_qn = match.get("qualified_name", name)
                entity_qn_map[name] = existing_qn
                if aliases:
                    for alias in aliases:
                        await self._mcp_session.call_tool(
                            "sysml_add_alias",
                            {"qualified_name": existing_qn, "alias": alias},
                        )
                continue

            # 不存在 → 创建
            add_result = await self._mcp_session.call_tool(
                "sysml_add_entity", {
                    "entity_type": etype,
                    "name": name,
                    "parent_package": "",
                    "description": desc,
                    "aliases": aliases,
                    "source_sections": [section_title or f"p{page}"],
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
            except json.JSONDecodeError:
                pass

        return {"entities": created_entities, "relations": created_relations}

    # ── Phase 3: Enrichment ──────────────────────────────────

    async def _enrich_entities(self, doc_name: str) -> None:
        """用强模型审查并补充别名/属性/关系."""
        if not self._unified_tools or self._mcp_session is None:
            return

        summary = await self._summary()
        if summary.get("total_entities", 0) == 0:
            return

        entity_list = await self._mcp_session.call_tool(
            "sysml_list_entities", {"include_details": False}
        )
        try:
            entities_data = json.loads(entity_list)
        except json.JSONDecodeError:
            return

        entity_names = [e.get("name", "") for e in entities_data.get("entities", [])]
        if not entity_names:
            return

        logger.info("Enrichment: reviewing %d entities with strong model", len(entity_names))

        agent = create_agent(
            model=self.llm,
            tools=self._unified_tools,
            system_prompt=ENRICHMENT_PROMPT,
            debug=config.settings.AGENT_VERBOSE,
            name="kg_enricher",
        )

        prompt = f"""文档: {doc_name}
已有实体列表 ({len(entity_names)}个):
{json.dumps(entity_names, ensure_ascii=False, indent=2)}

请逐一审查实体，补充别名、属性、关系。"""

        try:
            await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": self.max_iterations},
                ),
                timeout=self.timeout * 60,
            )
        except asyncio.TimeoutError:
            logger.warning("Enrichment timeout")
        except Exception as e:
            logger.warning("Enrichment error: %s", e)

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
