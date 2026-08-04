"""KG 元架构引导构建 — Phase 0 核心（kg-meta-architecture 计划 Task 1）。

两个核心类：
- ``MetaArchitecture``: 元架构数据类（实体类型层级 / 根节点 / 关系模式 / 约束），
  JSON 可序列化/反序列化，供后续 Phase 动态 Prompt 生成与后验过滤使用。
- ``GranularityAgent``: 三步流程 —— TOC 分析（程序化摘要）→ 小节采样（light_llm）
  → 元架构生成（llm）。复用 KGBuildAgent 的 light_llm / llm，不创建新的 LangChain Agent。

设计约束：
- 模块级不 import agent.kg_build_agent（避免触发 agent.tools 的模块级
  ``rag_engine = RAGEngine()`` 副作用）；langchain_core 消息类在方法内局部导入，
  与 KGBuildAgent._identify_root 的模式一致。
- ``MetaArchitecture`` 的 JSON 往返（to_dict/from_dict）不依赖任何 LLM。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetaArchitecture:
    """元架构：用户粒度描述 → LLM 确定的知识图谱 Schema 约束。

    字段：
      entity_types:           实体类型层级，元素形如
                              {"name": "PartDef", "parent": null, "level": 1,
                               "description": "部件/模块"}
      root_nodes:             根节点，元素形如 {"name": "系统概览", "type": "PartDef",
                               "description": "..."}
      relation_patterns:      关系模式，元素形如 {"source_type": "PartDef",
                               "target_type": "PartDef", "relation_type": "allocation",
                               "desc": "组成关系"}
      constraints:            自然语言约束，如 ["忽略温度参数", "不提取命令"]
      granularity_description: 原始用户粒度输入（CLI --granularity 参数）
      relation_network:       T2 统计报告挂载点（本任务可留 None）
    """

    entity_types: List[Dict[str, Any]] = field(default_factory=list)
    root_nodes: List[Dict[str, Any]] = field(default_factory=list)
    relation_patterns: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    granularity_description: str = ""
    relation_network: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为纯 JSON 兼容 dict（无 LLM / 无副作用）。"""
        return {
            "entity_types": list(self.entity_types),
            "root_nodes": list(self.root_nodes),
            "relation_patterns": list(self.relation_patterns),
            "constraints": list(self.constraints),
            "granularity_description": self.granularity_description,
            "relation_network": self.relation_network,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaArchitecture":
        """从 dict 反序列化（缺失字段容忍为默认值）。"""
        return cls(
            entity_types=list(data.get("entity_types") or []),
            root_nodes=list(data.get("root_nodes") or []),
            relation_patterns=list(data.get("relation_patterns") or []),
            constraints=list(data.get("constraints") or []),
            granularity_description=str(data.get("granularity_description") or ""),
            relation_network=data.get("relation_network"),
        )

    def to_json(self, **kwargs: Any) -> str:
        """序列化为 JSON 字符串（ensure_ascii=False，中文可读）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls, raw: str) -> "MetaArchitecture":
        """从 JSON 字符串反序列化。"""
        return cls.from_dict(json.loads(raw))


class GranularityAgent:
    """元架构引导构建的 Phase 0 核心：粒度描述 → MetaArchitecture。

    三步流程：
      Step 1 — TOC 分析：程序化摘要 ``tree_state.get_tree_structure()`` 的
               章节层次 / 密度（每章子树节点数 / 叶子页数）。
      Step 2 — 小节采样：light_llm 从 TOC 摘要挑选 3-5 个采样 node_id，
               从内存 sections（或 tree_state.nodes）读取文本；LLM 两次
               尝试均失败/有效节点不足时程序化回退（_fallback_sample：
               按内容密度选代表章节），保证采样环节不因 LLM 输出不稳定
               而整体失败。
      Step 3 — 元架构生成：llm 综合粒度描述 + TOC 摘要 + 采样文本 →
               MetaArchitecture JSON（严格 JSON 解析，参照
               KGBuildAgent._parse_json_response 模式）。

    模型选择与现有管线一致：Step 2 用 ``agent.light_llm``（快速小模型），
    Step 3 用 ``agent.llm``。Step 2 解析失败重试 1 次，仍失败则程序化
    回退采样（无 LLM）；Step 3 解析失败重试 1 次，仍失败则抛异常。
    异常（含回退也找不到任何有效节点）由 T4 的调用方捕获并降级为无
    meta 构建，保持向后兼容。
    """

    STEP2_SYSTEM_PROMPT = "你是技术文档分析专家。仅输出 JSON，无其他内容。"
    STEP3_SYSTEM_PROMPT = "你是知识图谱 Schema 设计专家。仅输出 JSON，无其他内容。"

    #: 采样文本单节上限（控制 Step 3 prompt 体积）
    SAMPLE_TEXT_MAX_CHARS = 6000
    #: TOC 摘要最多收录的目录条目数（控制 prompt 体积）
    TOC_MAX_ENTRIES = 80
    #: LLM 解析失败时的最大尝试次数（含首次，即重试 1 次）
    MAX_ATTEMPTS = 2

    def __init__(self, agent: Any, doc_name: str):
        """agent 为 KGBuildAgent 实例；light_llm/llm 惰性经其属性获取。"""
        self._agent = agent
        self.doc_name = doc_name

    # ── JSON 解析（参照 KGBuildAgent._parse_json_response 模式）──

    @staticmethod
    def _parse_json_object(raw: str) -> Dict[str, Any]:
        """从 LLM 输出解析 JSON 对象（非数组）；解析失败抛 JSONDecodeError。"""
        json_str = raw.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r"(\{.*\})", raw, re.DOTALL)
            if m:
                json_str = m.group(1)
        return json.loads(json_str)

    @staticmethod
    async def _ainvoke(llm: Any, messages: List[Any], timeout: int) -> str:
        """带超时的 ainvoke，统一提取响应文本（参照 _identify_root 模式）。"""
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
        return str(response.content) if hasattr(response, "content") else str(response)

    # ── Step 1: TOC 分析（程序化，无 LLM）──

    @staticmethod
    def _count_subtree(node: Dict[str, Any]) -> int:
        """子树节点总数（含自身）。"""
        children = node.get("children") or []
        return 1 + sum(GranularityAgent._count_subtree(c) for c in children)

    @staticmethod
    def _count_leaves(node: Dict[str, Any]) -> int:
        """子树叶子节点数（无 children 的节点视为叶子）。"""
        children = node.get("children") or []
        if not children:
            return 1
        return sum(GranularityAgent._count_leaves(c) for c in children)

    def _collect_toc_entries(
        self, node: Dict[str, Any], out: List[Dict[str, Any]], max_entries: int
    ) -> None:
        if len(out) >= max_entries:
            return
        entry: Dict[str, Any] = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "level": node.get("level", 0),
            "subtree_nodes": self._count_subtree(node),
            "leaf_pages": self._count_leaves(node),
        }
        page = node.get("page")
        if page:
            entry["page"] = page
        out.append(entry)
        for child in node.get("children") or []:
            self._collect_toc_entries(child, out, max_entries)

    def _summarize_toc(self, tree_state: Any) -> Dict[str, Any]:
        """提取章节层次/密度 → 供 LLM 的结构摘要（含截断标记）。"""
        tree = tree_state.get_tree_structure()
        summary: List[Dict[str, Any]] = []
        for node in tree.get("structure") or []:
            self._collect_toc_entries(node, summary, self.TOC_MAX_ENTRIES)
        return {
            "document": tree.get("document"),
            "total_pages": tree.get("total_pages"),
            "toc_nodes": len(summary),
            "truncated": len(summary) >= self.TOC_MAX_ENTRIES,
            "chapters": summary,
        }

    # ── 采样文本读取（内存优先；read_section MCP 工具在本代码库不存在，
    #    故不走 MCP 往返，与任务书"或直接 sections[nid] 若在内存"一致）──

    @staticmethod
    def _section_nodes(
        nid: str, tree_state: Any, sections: Optional[Dict[str, List[Any]]]
    ) -> List[Any]:
        """取 node_id 对应的节点列表：sections（Dict[str, List]）→ tree_state.nodes。"""
        if sections is not None:
            nodes = sections.get(nid)
            if isinstance(nodes, list):
                return nodes
            if nodes is not None:
                return [nodes]
        node = getattr(tree_state, "nodes", {}).get(nid)
        return [node] if node is not None else []

    @classmethod
    def _section_has_text(cls, nid: str, tree_state: Any, sections: Any) -> bool:
        return any(getattr(nd, "text", "") for nd in cls._section_nodes(nid, tree_state, sections))

    # ── Step 2: 小节采样（light_llm + 程序化回退）──

    @staticmethod
    def _first_leaf_id(node: Dict[str, Any]) -> Optional[str]:
        """深度优先取子树首个叶子节点（无 children）的 node_id；无则 None。"""
        children = node.get("children") or []
        if not children:
            return node.get("node_id")
        for child in children:
            leaf = GranularityAgent._first_leaf_id(child)
            if leaf is not None:
                return leaf
        return None

    @staticmethod
    def _toc_node_by_id(
        node: Dict[str, Any], nid: str
    ) -> Optional[Dict[str, Any]]:
        """在 TOC 结构树中按 node_id 查找节点（含后代）；未找到返回 None。"""
        if node.get("node_id") == nid:
            return node
        for child in node.get("children") or []:
            found = GranularityAgent._toc_node_by_id(child, nid)
            if found is not None:
                return found
        return None

    def _fallback_sample(
        self,
        tree_state: Any,
        sections: Optional[Dict[str, List[Any]]] = None,
        n: int = 3,
    ) -> List[str]:
        """程序化回退采样：LLM 采样失败时基于 TOC 结构选代表性节点（无 LLM）。

        策略（确定性）：
          1. 重新生成 TOC 摘要（_summarize_toc 为纯函数，结果与 Step 1 一致），
             目录条目按叶子页数（leaf_pages，内容密度）降序排序；
          2. 逐条目取其子树首个叶子 node_id（跨章节去重），取前 n 个不同章节；
          3. TOC 结构不可用/无章节时，直接取 tree_state.nodes 前 n 个节点；
          4. 有效性过滤（_section_has_text）；有效节点不足 2 个时从
             tree_state.nodes 全量补充。

        返回至少 2 个有效 node_id；仅当整棵树都没有有效节点时才返回空列表
        （此时才允许调用方降级为无 meta 构建）。
        """
        candidates: List[str] = []
        roots: List[Dict[str, Any]] = []
        try:
            tree = tree_state.get_tree_structure()
            roots = tree.get("structure") or []
            entries: List[Dict[str, Any]] = []
            for node in roots:
                self._collect_toc_entries(node, entries, self.TOC_MAX_ENTRIES)
            entries.sort(
                key=lambda e: (e.get("leaf_pages", 0), e.get("subtree_nodes", 0)),
                reverse=True,
            )
            for entry in entries:
                if len(candidates) >= n:
                    break
                nid = entry.get("node_id")
                if not nid:
                    continue
                raw: Optional[Dict[str, Any]] = None
                for root in roots:
                    raw = self._toc_node_by_id(root, nid)
                    if raw is not None:
                        break
                leaf_id = self._first_leaf_id(raw) if raw is not None else None
                if leaf_id and leaf_id not in candidates:
                    candidates.append(leaf_id)
        except Exception as e:  # noqa: BLE001 — 回退路径自身不因结构异常而失败
            logger.warning("GranularityAgent[%s] fallback TOC scan failed: %s",
                           self.doc_name, e)
        if not candidates:
            for nid in (getattr(tree_state, "nodes", None) or {}):
                if len(candidates) >= n:
                    break
                candidates.append(str(nid))

        valid = [i for i in candidates if self._section_has_text(i, tree_state, sections)]
        if len(valid) < 2:
            # 候选不足 2 个 → 从 tree_state.nodes 全量补充（除非整棵树不足）
            for nid in (getattr(tree_state, "nodes", None) or {}):
                if len(valid) >= 2:
                    break
                nid_s = str(nid)
                if nid_s not in candidates and self._section_has_text(
                    nid_s, tree_state, sections
                ):
                    valid.append(nid_s)
        return valid

    async def _sample_sections(
        self,
        granularity_description: str,
        toc_summary: Dict[str, Any],
        tree_state: Any,
        sections: Optional[Dict[str, List[Any]]],
    ) -> List[str]:
        """light_llm 决定采样 3-5 个 node_id；过滤无效 id 后返回有效列表。

        解析失败 / 有效采样不足时重试 1 次，仍失败则程序化回退采样
        （_fallback_sample，基于 TOC 内容密度选代表章节）；仅当回退也
        找不到任何有效节点时才抛 ValueError（由 T4 调用方降级）。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""你是技术文档分析专家。根据粒度要求选择最能代表内容细节的采样小节。

## 文档信息
- 文档名: {self.doc_name}

## 用户粒度要求
{granularity_description}

## 文档目录摘要
{json.dumps(toc_summary, ensure_ascii=False, indent=2)}

## 任务
从上述目录中选择 3-5 个 node_id 作为采样小节（优先覆盖不同主题、信息密度高的小节）。

## 输出格式（仅 JSON）
{{"sample_node_ids": ["n5", "n8"]}}"""

        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                raw = await self._ainvoke(
                    self._agent.light_llm,
                    [SystemMessage(content=self.STEP2_SYSTEM_PROMPT),
                     HumanMessage(content=prompt)],
                    timeout=180,
                )
                logger.debug("GranularityAgent[%s] Step2 raw (attempt %d): %s",
                             self.doc_name, attempt + 1, raw[:300])
                result = self._parse_json_object(raw)
                ids = result.get("sample_node_ids") or []
                if not isinstance(ids, list):
                    raise ValueError("sample_node_ids must be a JSON list")
                ids = [str(i) for i in ids]
                valid = [i for i in ids if self._section_has_text(i, tree_state, sections)]
                if len(valid) < 2:
                    raise ValueError(f"too few valid sample nodes (got {valid})")
                return valid[:5]
            except Exception as e:  # noqa: BLE001 — LLM 输出不可信，全部按解析失败重试
                last_err = e
                logger.warning("GranularityAgent[%s] Step2 attempt %d failed: %s",
                               self.doc_name, attempt + 1, e)
        # LLM 两次尝试均失败/有效节点不足 → 程序化回退采样（T8 修复）：
        # 保证采样环节不因 LLM 输出不稳定而整体失败（T8 实证：粗粒度描述
        # 下 LLM 常只输出顶层章节，经有效性过滤后不足 2 个）。
        fallback_ids = self._fallback_sample(tree_state, sections, n=3)
        if fallback_ids:
            logger.info("Fallback sampling used (LLM sampling failed): %d nodes",
                        len(fallback_ids))
            return fallback_ids
        raise ValueError(
            f"section sampling failed after {self.MAX_ATTEMPTS} attempts: {last_err}"
            " (fallback found no valid nodes in the whole tree)"
        )

    # ── Step 3: 元架构生成（llm）──

    async def _generate_meta_architecture(
        self,
        granularity_description: str,
        toc_summary: Dict[str, Any],
        samples: List[Dict[str, Any]],
    ) -> MetaArchitecture:
        """llm 综合粒度描述 + TOC 摘要 + 采样文本 → MetaArchitecture。

        解析失败 / 字段缺失时重试 1 次，仍失败抛 RuntimeError。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""你是知识图谱 Schema 设计专家。基于粒度要求与文档内容采样，确定元架构。

## 文档信息
- 文档名: {self.doc_name}

## 用户粒度要求
{granularity_description}

## 文档目录摘要
{json.dumps(toc_summary, ensure_ascii=False, indent=2)[:8000]}

## 采样小节内容
{json.dumps(samples, ensure_ascii=False, indent=2)[:12000]}

## 输出 Schema（仅 JSON）
{{
  "entity_types": [{{"name": "PartDef", "parent": null, "level": 1, "description": "部件/模块"}}],
  "root_nodes": [{{"name": "系统概览", "type": "PartDef", "description": "顶层系统"}}],
  "relation_patterns": [{{"source_type": "PartDef", "target_type": "PartDef", "relation_type": "allocation", "desc": "组成关系"}}],
  "constraints": ["自然语言约束，如忽略温度参数"]
}}

## 硬性规则
- entity_types 表达类型层级：level 1 为顶层类型；parent 指向父类型 name（顶层为 null）
- root_nodes 的 type 必须引用 entity_types 中的 name
- relation_patterns 的 relation_type 限选：
  allocation|connection|interface|containment|composition|reference|generalization|
  dependency|abstraction|realization|derive|trace|derivereqt|refine|satisfy|verify|
  copy|usecaseassociation|usecaseinclude|usecaseextend
- constraints 为自然语言粒度约束（不提取/忽略/详细到... 等）
- 仅输出 JSON 对象，无其他内容"""

        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                raw = await self._ainvoke(
                    self._agent.llm,
                    [SystemMessage(content=self.STEP3_SYSTEM_PROMPT),
                     HumanMessage(content=prompt)],
                    timeout=600,
                )
                logger.debug("GranularityAgent[%s] Step3 raw (attempt %d): %s",
                             self.doc_name, attempt + 1, raw[:400])
                result = self._parse_json_object(raw)
                for key in ("entity_types", "root_nodes", "relation_patterns"):
                    if not isinstance(result.get(key), list):
                        raise ValueError(f"field {key!r} missing or not a list")
                return MetaArchitecture(
                    entity_types=[dict(t) for t in result["entity_types"] if isinstance(t, dict)],
                    root_nodes=[dict(r) for r in result["root_nodes"] if isinstance(r, dict)],
                    relation_patterns=[dict(r) for r in result["relation_patterns"]
                                       if isinstance(r, dict)],
                    constraints=[str(c) for c in (result.get("constraints") or [])],
                    granularity_description=granularity_description,
                )
            except Exception as e:  # noqa: BLE001 — LLM 输出不可信，全部按解析失败重试
                last_err = e
                logger.warning("GranularityAgent[%s] Step3 attempt %d failed: %s",
                               self.doc_name, attempt + 1, e)
        raise RuntimeError(
            f"meta architecture generation failed after {self.MAX_ATTEMPTS} attempts: {last_err}"
        )

    # ── 主入口 ──

    async def determine_meta_architecture(
        self,
        granularity_description: str,
        tree_state: Any,
        sections: Optional[Dict[str, List[Any]]],
    ) -> MetaArchitecture:
        """三步流程：TOC 分析 → 小节采样 → 元架构生成。

        Args:
            granularity_description: 用户自然语言粒度描述（原始输入原样保存）。
            tree_state: DocumentTreeState 实例（提供 get_tree_structure()）。
            sections: node_id → [节点] 内存映射（与 build_kg_recursive 的
                _build_section_map 产物一致）；为 None 时回退读 tree_state.nodes。

        Returns:
            MetaArchitecture（granularity_description 已回填原始输入）。

        Raises:
            ValueError: Step 2 采样 LLM 两次尝试均失败/有效采样不足，且
                程序化回退（_fallback_sample）也找不到任何有效节点。
            RuntimeError: Step 3 生成 LLM 两次尝试均失败或字段缺失。
            由 T4 的调用方捕获并降级为无 meta 构建。
        """
        logger.info("GranularityAgent[%s]: determine meta architecture", self.doc_name)

        # Step 1 — TOC 分析（程序化，无 LLM）
        toc_summary = self._summarize_toc(tree_state)
        logger.info("GranularityAgent[%s]: TOC summary %d chapters",
                    self.doc_name, len(toc_summary.get("chapters", [])))

        # Step 2 — 小节采样（light_llm）+ 读文本（内存）
        sample_ids = await self._sample_sections(
            granularity_description, toc_summary, tree_state, sections
        )
        logger.info("GranularityAgent[%s]: sampled %d sections: %s",
                    self.doc_name, len(sample_ids), sample_ids)

        samples: List[Dict[str, Any]] = []
        for nid in sample_ids:
            nodes = self._section_nodes(nid, tree_state, sections)
            text = "\n\n".join(getattr(nd, "text", "") for nd in nodes).strip()
            if not text:
                continue
            samples.append({
                "node_id": nid,
                "title": getattr(nodes[0], "title", nid),
                "page": getattr(nodes[0], "page_start", 0),
                "text": text[: self.SAMPLE_TEXT_MAX_CHARS],
            })

        # Step 3 — 元架构生成（llm）
        meta = await self._generate_meta_architecture(
            granularity_description, toc_summary, samples
        )
        logger.info(
            "GranularityAgent[%s]: meta architecture done: %d entity types, "
            "%d roots, %d relation patterns, %d constraints",
            self.doc_name, len(meta.entity_types), len(meta.root_nodes),
            len(meta.relation_patterns), len(meta.constraints),
        )
        return meta


# ── 关系网络统计（T2：纯函数，无 I/O / 无 LLM / 无第三方依赖）──


def analyze_relation_network(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """关系网络的元认知统计报告（kg-meta-architecture 计划 T2）。

    输入 relations 的元素与 database/*/knowledge_graph.meta.json 的
    relations 数组一致：``{"source": str, "target": str, "type": str, "name": str}``。
    仅使用标准库（collections.deque / defaultdict / Counter），不引入 networkx。

    输出字段（与 MetaArchitecture.relation_network 的 Optional[Dict] 挂载点兼容）：
      relation_type_distribution: 关系类型 → 数量（{type: count}，键缺失记为 ""）
      num_relations:             输入关系总数（含自环与重复边）
      num_entities:              source + target 去重后的节点数
      hub_nodes:                 按 degree（去重无向边）降序的 top10，元素
                                 {"name": str, "degree": int}；空图返回 []
      num_components:            无向图连通分量数（BFS；孤立节点各为 1 个分量）
      component_sizes:           各分量节点数，降序
      max_depth:                 所有分量内各起点 BFS 可达最大层数（最长最短
                                 路径近似）的全局最大；空图返回 0
      self_loops:                source == target 的关系数（原始计数）

    实现约束：
      - 自环计入 degree（去重后每个自环节点计 1）与 self_loops（原始计数）；
        不参与 BFS 邻居传播（避免死循环）
      - a→b 与 b→a 视为同一条边（frozenset 去重），degree 基于去重后的无向边集
      - 空列表返回全零结构，不抛异常
    """
    from collections import Counter, defaultdict, deque

    type_counter: Counter = Counter()
    num_relations = 0
    self_loops = 0
    entities: set = set()
    edges: set = set()          # 无向去重边（不含自环），元素 frozenset({u, v}) 且 u != v
    self_loop_nodes: set = set()  # 出现自环的节点（去重）

    for rel in relations:
        src = rel.get("source") if isinstance(rel, dict) else None
        tgt = rel.get("target") if isinstance(rel, dict) else None
        if src is None or tgt is None:
            continue  # 防御：跳过畸形关系（正常数据不含）
        num_relations += 1
        type_counter[rel.get("type", "")] += 1
        entities.add(src)
        entities.add(tgt)
        if src == tgt:
            self_loops += 1
            self_loop_nodes.add(src)
        else:
            edges.add(frozenset((src, tgt)))

    # degree：基于去重无向边集（自环去重后每个节点计 1）
    degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        u, v = tuple(edge)
        degree[u] += 1
        degree[v] += 1
    for node in self_loop_nodes:
        degree[node] += 1

    # 邻接表（自环不参与传播）；连通分量 BFS
    adj: Dict[str, set] = defaultdict(set)
    for edge in edges:
        u, v = tuple(edge)
        adj[u].add(v)
        adj[v].add(u)

    visited: set = set()
    components: List[List[str]] = []
    for node in entities:
        if node in visited:
            continue
        queue = deque([node])
        visited.add(node)
        comp: List[str] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    # max_depth：每个分量内所有起点的 BFS 最大层数，取全局最大
    def _bfs_max_depth(start: str, adjacency: Dict[str, set]) -> int:
        seen = {start}
        queue = deque([start])
        depth = {start: 0}
        farthest = 0
        while queue:
            cur = queue.popleft()
            for nb in adjacency.get(cur, ()):
                if nb in seen:
                    continue
                seen.add(nb)
                depth[nb] = depth[cur] + 1
                if depth[nb] > farthest:
                    farthest = depth[nb]
                queue.append(nb)
        return farthest

    max_depth = 0
    for comp in components:
        for node in comp:
            max_depth = max(max_depth, _bfs_max_depth(node, adj))

    component_sizes = sorted((len(c) for c in components), reverse=True)
    hub_nodes = [
        {"name": name, "degree": deg}
        for name, deg in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:10]

    return {
        "relation_type_distribution": dict(type_counter),
        "num_relations": num_relations,
        "num_entities": len(entities),
        "hub_nodes": hub_nodes,
        "num_components": len(components),
        "component_sizes": component_sizes,
        "max_depth": max_depth,
        "self_loops": self_loops,
    }
