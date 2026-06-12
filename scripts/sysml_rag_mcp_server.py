#!/usr/bin/env python3
"""
SysML RAG MCP 服务器
——面向 SysML v2 模型的知识图谱构建与检索工具

提供工具：
  # 实体 CRUD
  sysml_add_entity        - 创建实体（含别名/属性/来源）
  sysml_update_entity     - 补充/合并实体信息
  sysml_delete_entity     - 删除实体
  # 搜索
  sysml_search_entity     - 多策略搜索（精确→别名→子串→正则→Token重叠）
  sysml_get_entity        - 实体详情
  sysml_normalize_name    - 名称归一化
  sysml_list_entities     - 全部实体列表
  sysml_add_alias         - 追加别名
  # 关系 CRUD
  sysml_add_relation      - 创建关系
  sysml_delete_relation   - 删除关系
  sysml_get_connections   - 实体关联查询
  sysml_list_relations    - 全部关系列表
  # 合并/去重
  sysml_suggest_merge     - 全局去重建议
  sysml_merge_entities    - 执行合并
  # 模型 I/O
  sysml_load_model        - 加载 .sysml 文件
  sysml_save_model        - 保存 .sysml 文件
  sysml_export_submodel   - 子模型导出
  sysml_model_summary     - 全局统计
  sysml_semantic_search   - 语义搜索
  sysml_import_doc        - 从文档导入

可作为独立 MCP 服务器运行，供 Build Agent 通过 MCP 协议调用。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 计算项目根目录并加入 sys.path（必须在导入 sysml 之前）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Optional, Set, Tuple

from sysml.sysml_model import (
    Package, SysMLElement, Namespace, Definition, Usage, Doc,
    ConnectionUsage, InterfaceUsage, AllocationUsage,
    ContainmentUsage, CompositionUsage, ReferenceUsage,
    GeneralizationUsage, DependencyUsage, AbstractionUsage,
    RealizationUsage, DeriveUsage, TraceUsage,
    DeriveReqtUsage, RefineUsage, SatisfyUsage,
    VerifyUsage, CopyUsage, UseCaseAssociationUsage,
    UseCaseIncludeUsage, UseCaseExtendUsage,
    InterfaceDef, AllocationDef,
    PartDef, PartUsage, AttributeDef, AttributeUsage,
    PortDef, PortUsage, ItemDef, ItemUsage,
    RequirementDef, RequirementUsage,
    ConnectionDef, CommandDef,
)
from sysml.sysml_manager import SysMLManager, AliasRegistry
from sysml.hv_resolver import HVResolver

try:
    from scripts.demo_doc_to_sysml import build_sysml_model_from_doc_tree
except Exception:
    try:
        from demo_doc_to_sysml import build_sysml_model_from_doc_tree
    except Exception:
        build_sysml_model_from_doc_tree = None

# ── 全局模型注册表 ───────────────────────────────────────────
_global_manager: Optional[SysMLManager] = None
_loaded_files: Dict[str, str] = {}  # file_stem → file_path


def _get_manager() -> SysMLManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = SysMLManager(workspace_root=ROOT_DIR)
    return _global_manager


def _entity_type_name(entity: Any) -> str:
    """返回人类可读的实体类型名称（子类型排在父类型前，避免 isinstance 误判）"""
    type_map = {
        PartDef: "部件定义", PartUsage: "部件使用",
        AttributeDef: "属性定义", AttributeUsage: "属性使用",
        PortDef: "端口定义", PortUsage: "端口使用",
        ItemDef: "项定义", ItemUsage: "项使用",
        RequirementDef: "需求定义", RequirementUsage: "需求使用",
        CommandDef: "命令定义",
        InterfaceUsage: "接口使用", InterfaceDef: "接口定义",
        AllocationUsage: "分配使用", AllocationDef: "分配定义",
        ContainmentUsage: "包含使用", CompositionUsage: "组合使用",
        ReferenceUsage: "引用使用",
        GeneralizationUsage: "泛化使用", DependencyUsage: "依赖使用",
        AbstractionUsage: "抽象使用", RealizationUsage: "实现使用",
        DeriveUsage: "派生使用", TraceUsage: "跟踪使用",
        DeriveReqtUsage: "派生需求使用", RefineUsage: "细化使用",
        SatisfyUsage: "满足使用", VerifyUsage: "验证使用",
        CopyUsage: "复制使用", UseCaseAssociationUsage: "用例关联使用",
        UseCaseIncludeUsage: "用例包含使用", UseCaseExtendUsage: "用例扩展使用",
        ConnectionDef: "连接定义", ConnectionUsage: "连接使用",
        Package: "包",
    }
    for cls, name in type_map.items():
        if isinstance(entity, cls):
            return name
    return type(entity).__name__


def _entity_summary(entity: SysMLElement, include_body: bool = False) -> Dict[str, Any]:
    """将实体序列化为结构化摘要"""
    info: Dict[str, Any] = {
        "name": getattr(entity, "name", ""),
        "type": _entity_type_name(entity),
        "class": type(entity).__name__,
        "qualified_name": entity.qualified_name,
    }

    if hasattr(entity, "short_name") and entity.short_name:
        info["short_name"] = entity.short_name

    if isinstance(entity, Definition):
        if entity.is_abstract:
            info["abstract"] = True
        if entity.is_variation:
            info["variation"] = True
        if entity.supertypes:
            info["supertypes"] = entity.supertypes
    elif isinstance(entity, Usage):
        if entity.type_refs:
            info["type"] += f" : {', '.join(entity.type_refs)}"
        if entity.subsetted:
            info["subsets"] = entity.subsetted
        if entity.redefined:
            info["redefines"] = entity.redefined
        if entity.value_expr:
            info["value"] = entity.value_expr

    if isinstance(entity, ConnectionUsage):
        if entity.ends:
            info["ends"] = [
                {"ref": e.ref, "role": e.role} for e in entity.ends
            ]

    # 从 metadata 获取 source_sections
    mgr = _get_manager()
    meta = mgr._entity_metadata.get(entity.qualified_name, {})
    if meta.get("source_sections"):
        info["source_sections"] = meta["source_sections"]

    # 从 alias registry 获取别名列表
    qn = entity.qualified_name
    if qn in mgr._alias_registry._aliases_by_entity:
        entity_aliases = [a for a in mgr._alias_registry._aliases_by_entity[qn] if a != entity.name]
        if entity_aliases:
            info["aliases"] = entity_aliases

    if include_body:
        members = getattr(entity, "members", None)
        if members:
            member_summaries = []
            for m in members:
                member_summaries.append(_entity_summary(m, include_body=False))
            if member_summaries:
                info["members"] = member_summaries

    return info


def _find_any_element(mgr: SysMLManager, name: str) -> Optional[SysMLElement]:
    """Find any element (Definition, Usage, Package) by name, alias, or instance name."""
    # 1. Alias registry lookup (normalized)
    qn = mgr._alias_registry.lookup(name)
    if qn:
        elem = mgr.find_entity_by_qn(qn)
        if elem is not None:
            return elem

    # 2. Exact name match in definitions
    found = mgr.find_definition(name)
    if found is not None:
        return found

    # 3. Exact name match in all elements (including usages)
    def search_in(ns):
        for m in ns.members:
            if getattr(m, "name", "") == name:
                return m
            if isinstance(m, Namespace):
                inner = search_in(m)
                if inner:
                    return inner
        return None

    for elem in mgr.root_elements:
        if getattr(elem, "name", "") == name:
            return elem
        if isinstance(elem, Namespace):
            inner = search_in(elem)
            if inner:
                return inner

    # 4. P2: substring / prefix match in all entity names
    name_lower = name.lower().strip()
    candidates: List[Tuple[SysMLElement, float]] = []

    def collect_candidates(ns, depth=0):
        for m in ns.members:
            m_name = getattr(m, "name", "") or ""
            m_name_lower = m_name.lower()
            score = 0.0
            if name_lower and m_name_lower:
                if m_name_lower == name_lower:
                    score = 0.99
                elif m_name_lower.startswith(name_lower):
                    score = 0.8 + 0.1 * (len(name_lower) / max(len(m_name_lower), 1))
                elif name_lower in m_name_lower:
                    score = 0.6 + 0.1 * (len(name_lower) / max(len(m_name_lower), 1))
                # P2: short query → check if any word token matches
                elif len(name_lower) <= 4:
                    words = re.split(r'[_\s]+', m_name_lower)
                    if any(name_lower in w for w in words):
                        score = 0.5
            if score > 0.4:
                candidates.append((m, score))
            if isinstance(m, Namespace):
                collect_candidates(m, depth + 1)

    for elem in mgr.root_elements:
        e_name = getattr(elem, "name", "") or ""
        e_lower = e_name.lower()
        if name_lower and e_lower:
            if e_lower == name_lower:
                candidates.append((elem, 0.99))
            elif e_lower.startswith(name_lower):
                candidates.append((elem, 0.8))
            elif name_lower in e_lower:
                candidates.append((elem, 0.6))
        if isinstance(elem, Namespace):
            collect_candidates(elem)

    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    # 5. Fuzzy search fallback via search_entities
    results = mgr.search_entities(name, threshold=0.5)
    if results:
        best = results[0]
        return mgr.find_entity_by_qn(best["qualified_name"])

    return None


def _entity_detail_full(entity: SysMLElement, mgr: SysMLManager) -> Dict[str, Any]:
    """Generate full entity detail: all SysML properties, members, metadata, aliases."""
    info = _entity_summary(entity, include_body=True)
    qn = entity.qualified_name

    # Metadata
    meta = mgr.get_entity_metadata(qn)
    if any(v for v in meta.values() if v):
        info["metadata"] = {}
        if meta.get("description"):
            info["metadata"]["description"] = meta["description"]
        if meta.get("source_sections"):
            info["metadata"]["source_sections"] = meta["source_sections"]
        if meta.get("source_text"):
            info["metadata"]["source_text"] = meta["source_text"]
        if meta.get("properties"):
            info["metadata"]["properties"] = meta["properties"]

    # Aliases
    aliases = mgr._alias_registry.get_aliases(qn)
    if aliases:
        info["aliases"] = aliases

    # Definition-specific
    if isinstance(entity, Definition):
        if entity.supertypes:
            info["supertypes"] = entity.supertypes
        if entity.is_abstract:
            info["abstract"] = True
        if entity.is_variation:
            info["variation"] = True

    # Usage-specific
    if isinstance(entity, Usage):
        if entity.direction:
            info["direction"] = entity.direction.value
        if entity.multiplicity:
            info["multiplicity"] = entity.multiplicity.to_text()
        if entity.type_refs:
            info["type_refs"] = entity.type_refs
        if entity.subsetted:
            info["subsetted"] = entity.subsetted
        if entity.redefined:
            info["redefined"] = entity.redefined
        if entity.value_expr is not None:
            info["value"] = entity.value_expr
        if entity.is_derived:
            info["derived"] = True
        if entity.is_constant:
            info["constant"] = True
        if entity.is_reference:
            info["reference"] = True

    # ConnectionUsage-specific
    if isinstance(entity, ConnectionUsage) and entity.ends:
        info["ends"] = [
            {"ref": e.ref, "role": e.role} for e in entity.ends
        ]

    return info


def _build_k_layer_graph(mgr: SysMLManager, entity_name: str, k: int) -> Dict[str, Any]:
    """BFS traversal from entity_name up to k layers. Returns layered graph structure."""
    all_relations = mgr.get_all_relations()

    # Build adjacency: entity_name -> list of {relation, neighbor, my_role, neighbor_role}
    adjacency = defaultdict(list)
    for rel in all_relations:
        ends = getattr(rel, "ends", None) or []
        if not ends:
            continue

        rel_summary = {
            "name": getattr(rel, "name", ""),
            "qualified_name": rel.qualified_name,
            "type_name": _entity_type_name(rel),
            "class": type(rel).__name__,
        }
        rel_meta = mgr.get_entity_metadata(rel.qualified_name)
        if rel_meta and rel_meta.get("description"):
            rel_summary["description"] = rel_meta["description"]

        for i, end_i in enumerate(ends):
            for j, end_j in enumerate(ends):
                if i == j:
                    continue
                adjacency[end_i.ref].append({
                    "relation": rel_summary,
                    "neighbor": end_j.ref,
                    "my_role": end_i.role,
                    "neighbor_role": end_j.role,
                })

    visited: Set[str] = set()
    all_nodes: Dict[str, Any] = {}
    all_edges: List[Dict[str, Any]] = []
    layers: List[Dict[str, Any]] = []

    current_frontier = {entity_name}

    for level in range(k + 1):
        if not current_frontier:
            break

        layer_node_ids: List[str] = []
        next_frontier: Set[str] = set()

        for ename in sorted(current_frontier):
            if ename in visited:
                continue
            visited.add(ename)
            layer_node_ids.append(ename)

            # Get entity details for this node
            entity = _find_any_element(mgr, ename)
            if entity:
                all_nodes[ename] = _entity_detail_full(entity, mgr)
            else:
                all_nodes[ename] = {
                    "name": ename, "class": "Unknown", "qualified_name": ename,
                    "present": False, "note": "Referenced entity not found in model",
                }

            # Explore neighbors for next level
            if level < k:
                for edge in adjacency.get(ename, []):
                    neighbor = edge["neighbor"]
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    # Record edge (deduplicate by from/to pair)
                    edge_key = (ename, neighbor, edge["relation"]["name"])
                    if edge_key not in {(e["from"], e["to"], e["relation"]["name"]) for e in all_edges}:
                        all_edges.append({
                            "from": ename,
                            "to": neighbor,
                            "relation": edge["relation"],
                            "from_role": edge["my_role"],
                            "to_role": edge["neighbor_role"],
                        })

        layers.append({"level": level, "node_ids": layer_node_ids})
        current_frontier = next_frontier

    return {
        "depth": k,
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "nodes": all_nodes,
        "edges": all_edges,
        "layers": layers,
    }


def _fuzzy_match_name(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """简单模糊匹配：支持子串和拼音首字母（简化版用子串+小写匹配）"""
    results: List[Tuple[str, float]] = []
    query_lower = query.lower().strip()

    for cand in candidates:
        cand_lower = cand.lower()
        score = 0.0

        # 精确匹配
        if cand_lower == query_lower:
            score = 1.0
        # 以查询开头
        elif cand_lower.startswith(query_lower):
            score = 0.9
        # 包含查询
        elif query_lower in cand_lower:
            # 查询越长，匹配越精确
            score = 0.7 + 0.2 * (len(query_lower) / max(len(cand_lower), 1))
        # 单词匹配
        elif query_lower.replace("_", " ") in cand_lower.replace("_", " "):
            score = 0.6
        # 每个词至少出现
        elif all(w in cand_lower for w in query_lower.split()):
            score = 0.5

        if score > 0:
            results.append((cand, score))

    results.sort(key=lambda x: -x[1])
    return results[:20]


# ══════════════════════════════════════════════════════════════
# MCP 工具函数
# ══════════════════════════════════════════════════════════════

def _sysml_save_model(file_path: Optional[str] = None) -> Dict[str, Any]:
    """保存当前模型到 .sysml 文件"""
    mgr = _get_manager()
    mgr.save_to_file(file_path)
    return {"ok": True, "file": str(mgr.current_model_file)}


def sysml_load_model(file_path: str) -> Dict[str, Any]:
    """
    加载一个 .sysml 模型文件到全局管理器。

    Args:
        file_path: .sysml 文件路径

    Returns:
        加载状态摘要
    """
    mgr = _get_manager()
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return {"ok": False, "error": f"File not found: {file_path}"}

    try:
        mgr.load_from_file(str(path))
        # 记录单个已加载文件
        _loaded_files[path.stem] = str(path)
        entity_count = len(mgr.get_all_entities())
        rel_count = len(mgr.get_all_relations())
        # 重建超变量解析器
        global _HV_RESOLVER
        _HV_RESOLVER = HVResolver()
        return {
            "ok": True,
            "file": str(path),
            "entities": entity_count,
            "relations": rel_count,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def sysml_list_entities(include_details: bool = False) -> Dict[str, Any]:
    """
    列出当前已加载模型中的所有实体定义。

    Args:
        include_details: 是否包含详细成员信息

    Returns:
        实体列表及计数
    """
    mgr = _get_manager()
    entities = mgr.get_all_entities()

    result = {
        "total": len(entities),
        "entities": [],
    }

    type_count: Dict[str, int] = {}
    for entity in entities:
        summary = _entity_summary(entity, include_body=include_details)
        result["entities"].append(summary)
        t = _entity_type_name(entity)
        type_count[t] = type_count.get(t, 0) + 1

    result["type_distribution"] = type_count
    return result


def sysml_list_relations(include_details: bool = False) -> Dict[str, Any]:
    """
    列出当前已加载模型中的所有关系（连接、接口、分配）。

    Args:
        include_details: 是否包含详细成员信息

    Returns:
        关系列表及计数
    """
    mgr = _get_manager()
    relations = mgr.get_all_relations()

    result = {
        "total": len(relations),
        "relations": [],
    }

    for rel in relations:
        summary = _entity_summary(rel, include_body=include_details)
        result["relations"].append(summary)

    return result


def sysml_search_entity(query: str, threshold: float = 0.3,
                       regex_pattern: Optional[str] = None) -> Dict[str, Any]:
    """
    按名称多策略搜索实体（精确→归一化→别名→子串→正则→Token重叠）。

    Args:
        query: 搜索关键词
        threshold: 最低匹配阈值 (0.0~1.0)
        regex_pattern: 可选正则表达式辅助匹配

    Returns:
        匹配的实体列表及置信度
    """
    mgr = _get_manager()
    results = mgr.search_entities(query, threshold=threshold, regex_pattern=regex_pattern)

    return {
        "ok": True,
        "query": query,
        "threshold": threshold,
        "total_matches": len(results),
        "matches": results,
    }


def sysml_get_entity(entity_name: str) -> Dict[str, Any]:
    """
    获取指定实体的详细信息（含子特征和下级成员树）。

    Args:
        entity_name: 实体名称或限定名

    Returns:
        实体详情
    """
    mgr = _get_manager()
    entity = mgr.find_definition(entity_name)

    if entity is None:
        # 尝试用限定名查找
        found = mgr.find_element(qualified_name=entity_name)
        if found is not None and isinstance(found, Definition):
            entity = found

    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    summary = _entity_summary(entity, include_body=True)

    # 收集关联关系
    relations = []
    all_rels = mgr.get_all_relations()
    for rel in all_rels:
        if isinstance(rel, (ConnectionUsage, InterfaceUsage, AllocationUsage)):
            # 检查该实体是否出现在连接的任一端
            if hasattr(rel, "ends") and rel.ends:
                for end in rel.ends:
                    if end.ref == entity_name or end.ref == entity.name:
                        relations.append(_entity_summary(rel, include_body=False))
                        break

    result = summary
    result["ok"] = True
    result["related_connections"] = relations
    result["related_count"] = len(relations)
    return result


def sysml_get_connections(entity_name: str) -> Dict[str, Any]:
    """
    获取与指定实体相关的所有连接关系。

    Args:
        entity_name: 实体名称

    Returns:
        连接列表
    """
    mgr = _get_manager()
    entity = mgr.find_definition(entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}", "connections": []}

    connections = []
    all_rels = mgr.get_all_relations()
    for rel in all_rels:
        if isinstance(rel, (ConnectionUsage, InterfaceUsage, AllocationUsage)):
            if hasattr(rel, "ends") and rel.ends:
                involved = False
                for end in rel.ends:
                    if end.ref == entity_name or end.ref == entity.name:
                        involved = True
                        break
                if involved:
                    connections.append(_entity_summary(rel, include_body=False))

    return {
        "ok": True,
        "entity": entity_name,
        "total_connections": len(connections),
        "connections": connections,
    }


def sysml_export_submodel(entity_name: str, depth: int = 2) -> Dict[str, Any]:
    """
    导出一个以指定实体为核心的子模型视图。

    从实体出发，收集其成员及关联实体，生成局部 SysML 文本。

    Args:
        entity_name: 核心实体名称
        depth: 导出深度（默认2层）

    Returns:
        包含 SysML 文本段及关联实体列表
    """
    mgr = _get_manager()
    entity = mgr.find_definition(entity_name)

    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    collected: Set[str] = set()

    def collect_refs(item: Any, d: int):
        if d <= 0:
            return
        name = getattr(item, "name", "")
        if name:
            collected.add(name)

        if isinstance(item, ConnectionUsage) and hasattr(item, "ends"):
            for end in getattr(item, "ends", []) or []:
                collected.add(getattr(end, "ref", str(end)))

        members = getattr(item, "members", None)
        if members:
            for m in members:
                collect_refs(m, d - 1)

    collect_refs(entity, depth)

    # 收集关联的连接
    all_rels = mgr.get_all_relations()
    for rel in all_rels:
        if isinstance(rel, ConnectionUsage) and hasattr(rel, "ends"):
            for end in list(rel.ends or []):
                if end.ref == entity.name:
                    collect_refs(rel, 1)
                    break

    # 生成 SysML 文本
    lines: List[str] = []
    lines.append(f"// Submodel view centered on: {entity.name}")
    lines.append(f"// Depth: {depth}, Referenced entities: {len(collected)}")
    lines.append("")
    lines.append(entity.to_text())

    # 关联实体文本
    for ref_name in sorted(collected):
        if ref_name == entity.name:
            continue
        found = mgr.find_definition(ref_name)
        if found:
            lines.append("")
            lines.append(f"// Referenced: {ref_name}")
            lines.append(found.to_text())

    sysml_text = "\n".join(lines)

    return {
        "ok": True,
        "center_entity": entity_name,
        "depth": depth,
        "referenced_entities": sorted(list(collected)),
        "sysml_text": sysml_text,
    }


def sysml_import_doc(file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    从文档文件（PDF/DOCX/TXT/MD/XLSX）导入并构建 SysML 模型。

    内部调用 demo_doc_to_sysml 的文档树→SysML 转换逻辑。

    Args:
        file_path: 源文档路径
        output_path: 输出 .sysml 文件路径（可选，默认同名）

    Returns:
        导入结果摘要
    """

    if build_sysml_model_from_doc_tree is None:
        return {"ok": False, "error": "demo_doc_to_sysml import not available"}

    try:
        mgr, payload = build_sysml_model_from_doc_tree(file_path)

        if output_path is None:
            output_path = str(Path(file_path).with_suffix(".sysml"))

        mgr.save_to_file(output_path)

        # 同时加载到全局管理器
        global_mgr = _get_manager()
        # 合并元素
        for elem in mgr.root_elements:
            # 简单追加（实际可由用户决定是否替换）
            global_mgr.root_elements.append(elem)

        entity_count = len(mgr.get_all_entities())
        rel_count = len(mgr.get_all_relations())

        return {
            "ok": True,
            "source_document": file_path,
            "output_sysml": output_path,
            "entities_extracted": entity_count,
            "relations_extracted": rel_count,
            "document_title": payload.get("title", ""),
            "page_count": payload.get("page_count", 0),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def sysml_semantic_search(query: str, search_content: bool = True) -> Dict[str, Any]:
    """
    在已加载的 SysML 模型中执行语义搜索。

    搜索策略：
    1. 在实体名称/类型中匹配
    2. 可选：在实体成员树中深入匹配

    Args:
        query: 搜索关键词
        search_content: 是否搜索成员内容

    Returns:
        匹配项列表
    """
    mgr = _get_manager()
    entities = mgr.get_all_entities()

    results: List[Dict[str, Any]] = []

    for entity in entities:
        score = 0.0
        reasons: List[str] = []

        name = getattr(entity, "name", "")
        if name:
            name_matches = _fuzzy_match_name(query, [name])
            if name_matches:
                score = max(score, name_matches[0][1])
                reasons.append(f"name_match: {name_matches[0][1]:.2f}")

        type_name = _entity_type_name(entity)
        if query.lower() in type_name.lower():
            score = max(score, 0.7)
            reasons.append(f"type_match: {type_name}")

        if hasattr(entity, "short_name") and entity.short_name:
            if query.lower() in str(entity.short_name).lower():
                score = max(score, 0.6)
                reasons.append("short_name_match")

        # 搜索成员
        if search_content:
            members = getattr(entity, "members", None)
            if members:
                member_texts = []
                for m in members:
                    member_texts.append(f"{_entity_type_name(m)} {getattr(m, 'name', '')}")
                combined = " ".join(member_texts)
                if query.lower() in combined.lower():
                    score = max(score, 0.5)
                    reasons.append("member_content_match")

        if score > 0:
            summary = _entity_summary(entity, include_body=False)
            summary["match_score"] = round(score, 3)
            summary["match_reasons"] = reasons
            results.append(summary)

    results.sort(key=lambda x: -x["match_score"])

    return {
        "ok": True,
        "query": query,
        "total_matches": len(results),
        "results": results[:30],
    }


def sysml_model_summary() -> Dict[str, Any]:
    """
    获取当前已加载模型的全局摘要。

    Returns:
        模型统计信息
    """
    mgr = _get_manager()
    entities = mgr.get_all_entities()
    relations = mgr.get_all_relations()

    type_dist: Dict[str, int] = {}
    for e in entities:
        t = _entity_type_name(e)
        type_dist[t] = type_dist.get(t, 0) + 1

    root_elements = [
        {"name": getattr(e, "name", ""), "type": _entity_type_name(e)}
        for e in mgr.root_elements
    ]

    return {
        "ok": True,
        "loaded_files": list(_loaded_files.values()),
        "total_entities": len(entities),
        "total_relations": len(relations),
        "type_distribution": type_dist,
        "root_elements": root_elements,
    }


def sysml_connected_components() -> Dict[str, Any]:
    """
    计算知识图谱的连通分量（子图），返回各分量包含的实体名称列表。
    """
    mgr = _get_manager()
    entities = mgr.get_all_entities()
    relations = mgr.get_all_relations()

    adj: Dict[str, set] = {}
    for e in entities:
        name = getattr(e, "name", "")
        adj.setdefault(name, set())

    for rel in relations:
        ends = getattr(rel, "ends", None) or []
        refs = [end.ref for end in ends if end.ref]
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                adj.setdefault(refs[i], set()).add(refs[j])
                adj.setdefault(refs[j], set()).add(refs[i])

    visited: set = set()
    components = []
    for name in adj:
        if name in visited:
            continue
        queue = [name]
        visited.add(name)
        comp = []
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(comp)

    components.sort(key=len)
    return {
        "ok": True,
        "total_entities": len(entities),
        "total_components": len(components),
        "components": [{"size": len(c), "entities": c} for c in components],
    }


# ── Bracket expansion helpers ─────────────────────────────────

import re as _re


def _expand_bracket_name(name: str) -> list[str]:
    """展开方括号: ion[0-99] → [ion0,...,ion99]; mn[1,3] → [mn1,mn3]"""
    result = [name]
    pat_range = _re.compile(r'^(.+)\[(\d+)-(\d+)\](.*)$')
    while True:
        changed = False
        new_result = []
        for n in result:
            m = pat_range.match(n)
            if m:
                prefix, start, end, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                for i in range(start, end + 1):
                    new_result.append(f"{prefix}{i}{suffix}")
                changed = True
            else:
                new_result.append(n)
        result = new_result
        if not changed:
            break
    pat_list = _re.compile(r'^(.+)\[([\d,]+)\](.*)$')
    while True:
        changed = False
        new_result = []
        for n in result:
            m = pat_list.match(n)
            if m:
                prefix, nums, suffix = m.group(1), m.group(2), m.group(3)
                for num in nums.split(','):
                    new_result.append(f"{prefix}{num.strip()}{suffix}")
                changed = True
            else:
                new_result.append(n)
        result = new_result
        if not changed:
            break
    return result


def _expand_relation_pair(source: str, target: str) -> list[tuple[str, str]]:
    """扩展关系: src[0-2] conn tgt[0-1] → 3×2=6 对"""
    src_list = _expand_bracket_name(source)
    tgt_list = _expand_bracket_name(target)
    if len(src_list) == 1 and len(tgt_list) == 1:
        return [(source, target)]
    return [(s, t) for s in src_list for t in tgt_list]


def _sanitize_name(name: str) -> str:
    """清理实体名: 去除所有非 SysML 标识符字符"""
    for ch in ['（', '）', '(', ')', '<', '>', '"', '\'', ' ', '/', '\\', ',', ';', ':']:
        name = name.replace(ch, '_')
    # Collapse consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_') or "_"


# ══════════════════════════════════════════════════════════════
# 新增：实体检索工具
# ══════════════════════════════════════════════════════════════
# 新增：实体 CRUD 工具
# ══════════════════════════════════════════════════════════════

def sysml_add_entity(
    entity_type: str,
    name: str,
    parent_package: Optional[str] = None,
    description: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    source_sections: Optional[List[str]] = None,
    source_text: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    supertypes: Optional[List[str]] = None,
    short_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建新实体（PartDef/AttributeDef/RequirementDef/PortDef/ItemDef/Package 等）。

    Args:
        entity_type: 实体类型 (PartDef, AttributeDef, PortDef, ItemDef, RequirementDef 及对应的 Usage 变体, Package)
        name: 实体名称
        parent_package: 父包限定名（可选，不指定则放在顶层）
        description: 实体描述
        aliases: 别名列表
        source_sections: 来源章节列表
        source_text: 原始文本
        properties: 属性字典
        supertypes: 父类型列表（仅 Definition）
        short_name: 短名称

    Returns:
        创建结果
    """
    name = _sanitize_name(name)
    if short_name:
        short_name = _sanitize_name(short_name)
    mgr = _get_manager()
    names = _expand_bracket_name(name)
    if len(names) > 1:
        results = []
        for n in names:
            element = mgr.add_entity_with_metadata(
                entity_type=entity_type, name=n,
                parent_package=parent_package, description=description,
                aliases=aliases, source_sections=source_sections,
                source_text=source_text, properties=properties,
                supertypes=supertypes, short_name=n,
            )
            if element is None:
                results.append({"ok": False, "error": f"Invalid entity_type: {entity_type}"})
            else:
                results.append({
                    "ok": True, "qualified_name": element.qualified_name,
                    "name": element.name, "type": type(element).__name__,
                })
        return {"ok": True, "expanded_from": name, "count": len(results), "items": results}
    element = mgr.add_entity_with_metadata(
        entity_type=entity_type, name=name,
        parent_package=parent_package, description=description,
        aliases=aliases, source_sections=source_sections,
        source_text=source_text, properties=properties,
        supertypes=supertypes, short_name=short_name,
    )
    if element is None:
        return {"ok": False, "error": f"Invalid entity_type: {entity_type}"}
    return {
        "ok": True,
        "qualified_name": element.qualified_name,
        "name": element.name,
        "type": type(element).__name__,
    }


def sysml_update_entity(
    qualified_name: str,
    append_description: Optional[str] = None,
    append_source_sections: Optional[List[str]] = None,
    merge_aliases: Optional[List[str]] = None,
    update_properties: Optional[Dict[str, Any]] = None,
    new_name: Optional[str] = None,
    supertypes: Optional[List[str]] = None,
    type_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    更新实体信息（追加描述、合并别名、添加来源章节、更新属性）。

    Args:
        qualified_name: 实体限定名
        append_description: 追加描述文本
        append_source_sections: 追加来源章节
        merge_aliases: 合并别名列表
        update_properties: 更新属性字典
        new_name: 重命名实体
        supertypes: 更新父类型 (Definition)
        type_refs: 更新类型引用 (Usage)

    Returns:
        更新结果
    """
    mgr = _get_manager()
    ok = mgr.update_entity_metadata(
        qualified_name=qualified_name,
        append_description=append_description,
        append_source_sections=append_source_sections or [],
        merge_aliases=merge_aliases or [],
        update_properties=update_properties or {},
        new_name=new_name or "",
        supertypes=supertypes or [],
        type_refs=type_refs or [],
    )
    if not ok:
        return {"ok": False, "error": f"Entity not found: {qualified_name}"}
    return {"ok": True, "qualified_name": qualified_name}


def sysml_delete_entity(qualified_name: str) -> Dict[str, Any]:
    """
    删除指定实体及其元数据、别名。

    Args:
        qualified_name: 实体限定名

    Returns:
        删除结果
    """
    mgr = _get_manager()
    ok = mgr.delete_entity(qualified_name)
    if not ok:
        return {"ok": False, "error": f"Entity not found: {qualified_name}"}
    return {"ok": True, "deleted": qualified_name}


def sysml_normalize_name(name: str) -> Dict[str, Any]:
    """
    对名称进行归一化处理（去标点、小写、全半角统一）。

    Args:
        name: 原始名称

    Returns:
        归一化结果
    """
    normalized = AliasRegistry.normalize(name)
    return {"ok": True, "original": name, "normalized": normalized}


def sysml_add_alias(qualified_name: str, alias: str) -> Dict[str, Any]:
    """
    为已有实体追加别名。

    Args:
        qualified_name: 实体限定名
        alias: 新别名

    Returns:
        操作结果
    """
    mgr = _get_manager()
    ok = mgr.add_alias(qualified_name, alias)
    if not ok:
        return {"ok": False, "error": f"Entity not found: {qualified_name}"}
    return {"ok": True, "qualified_name": qualified_name, "alias_added": alias}


# ══════════════════════════════════════════════════════════════
# 新增：关系 CRUD 工具
# ══════════════════════════════════════════════════════════════

def sysml_add_relation(
    relation_type: str,
    source: str,
    target: str,
    name: Optional[str] = None,
    parent_package: Optional[str] = None,
    description: Optional[str] = None,
    role_source: Optional[str] = None,
    role_target: Optional[str] = None,
    source_sections: Optional[List[str]] = None,
    source_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建关系（connection / interface / allocation）。

    Args:
        relation_type: 关系类型 ("connection", "interface", "allocation")
        source: 源实体名称
        target: 目标实体名称
        name: 关系名称（可选）
        parent_package: 父包限定名（可选）
        description: 关系描述
        role_source: 源端角色名
        role_target: 目标端角色名
        source_sections: 来源小节列表（可选）
        source_text: 来源原文（可选）

    Returns:
        创建结果
    """
    source = _sanitize_name(source)
    target = _sanitize_name(target)
    mgr = _get_manager()
    pairs = _expand_relation_pair(source, target)
    if len(pairs) > 1:
        results = []
        for s, t in pairs:
            rel = mgr.add_relation(
                relation_type=relation_type, source_name=s, target_name=t,
                name=name, parent_package=parent_package,
                description=description, role_source=role_source,
                role_target=role_target,
                source_sections=source_sections, source_text=source_text,
            )
            if rel is None:
                results.append({"ok": False, "error": f"Invalid relation_type: {relation_type}"})
            else:
                results.append({
                    "ok": True, "qualified_name": rel.qualified_name,
                    "name": rel.name, "type": type(rel).__name__,
                    "source": s, "target": t,
                })
        return {"ok": True, "expanded_from": f"{source}→{target}", "count": len(results), "items": results}
    rel = mgr.add_relation(
        relation_type=relation_type, source_name=source, target_name=target,
        name=name, parent_package=parent_package,
        description=description, role_source=role_source,
        role_target=role_target,
        source_sections=source_sections, source_text=source_text,
    )
    if rel is None:
        return {"ok": False, "error": f"Invalid relation_type: {relation_type}"}
    return {
        "ok": True,
        "qualified_name": rel.qualified_name,
        "name": rel.name,
        "type": type(rel).__name__,
        "source": source,
        "target": target,
    }


def sysml_delete_relation(name: str, parent_package: Optional[str] = None) -> Dict[str, Any]:
    """
    删除指定关系。

    Args:
        name: 关系名称
        parent_package: 父包限定名（可选）

    Returns:
        删除结果
    """
    mgr = _get_manager()
    ok = mgr.delete_relation(name, parent_package=parent_package)
    if not ok:
        return {"ok": False, "error": f"Relation not found: {name}"}
    return {"ok": True, "deleted": name}


# ══════════════════════════════════════════════════════════════
# 新增：合并/去重工具
# ══════════════════════════════════════════════════════════════

def sysml_suggest_merge(threshold: float = 0.6) -> Dict[str, Any]:
    """
    分析当前模型中的重复实体，返回合并建议。

    Args:
        threshold: 最低置信度阈值 (0.0~1.0)

    Returns:
        合并建议列表
    """
    mgr = _get_manager()
    suggestions = mgr.suggest_merges(threshold=threshold)
    return {
        "ok": True,
        "threshold": threshold,
        "total_suggestions": len(suggestions),
        "suggestions": suggestions,
    }


def sysml_merge_entities(source: str, target: str) -> Dict[str, Any]:
    """
    将 source 实体合并到 target 实体（转移别名、描述、来源章节、属性）。

    Args:
        source: 源实体限定名（将被删除）
        target: 目标实体限定名（保留）

    Returns:
        合并结果
    """
    mgr = _get_manager()
    result_qn = mgr.merge_entities(source, target)
    if result_qn is None:
        return {"ok": False, "error": "Merge failed: source or target not found"}
    return {"ok": True, "merged_from": source, "merged_into": target, "result": result_qn}


# ══════════════════════════════════════════════════════════════
# 超变量 / 文档章节工具
# ══════════════════════════════════════════════════════════════

_HV_RESOLVER: Optional[HVResolver] = None


def _get_resolver() -> HVResolver:
    global _HV_RESOLVER
    if _HV_RESOLVER is None:
        _HV_RESOLVER = HVResolver()
    return _HV_RESOLVER


def _get_hv_config(mgr: SysMLManager, hv_id: str) -> Optional[dict]:
    """Extract HV configuration (source, selector, unit, hv_type) from the SysML model."""
    HV_TYPES = {"InstantVariable", "BlockVariable", "ModifierVariable", "Parameter", "Operation"}

    def walk(items):
        for item in (items or []):
            name = getattr(item, "name", None)
            if name == hv_id:
                type_refs = getattr(item, "type_refs", []) or []
                hv_type = next((t for t in type_refs if t in HV_TYPES), None)
                if hv_type:
                    info = {"name": name, "hv_type": hv_type}
                    for m in getattr(item, "members", []) or []:
                        m_name = getattr(m, "name", "")
                        value = getattr(m, "value_expr", None)
                        if m_name and value is not None:
                            info[m_name] = str(value).strip("'\" ")
    return info


def _build_hv_lookup(mgr: SysMLManager, text: str) -> dict[str, tuple[str, str]]:
    """Build a lookup dict {hv_id: (source, selector)} from all HV tags in text."""
    import re
    lookup = {}
    for m in re.finditer(r'<(\w+):([a-zA-Z_][a-zA-Z0-9_]*):([^>]+)>', text):
        hv_id = m.group(2)
        if hv_id not in lookup:
            cfg = _get_hv_config(mgr, hv_id)
            if cfg:
                lookup[hv_id] = (cfg.get("source", ""), cfg.get("selector", ""))
    return lookup


def _find_sections(manager: SysMLManager) -> list[PartUsage]:
    """Collect all DocumentSection instances from the model tree."""
    sections = []
    entities = manager.get_all_entities()
    # Also check root-level usage elements
    for e in manager.root_elements:
        if isinstance(e, PartUsage):
            entities.append(e)
        # Walk into packages
        def walk(ns):
            for m in getattr(ns, "members", []) or []:
                if isinstance(m, PartUsage):
                    entities.append(m)
                if hasattr(m, "members"):
                    walk(m)
        if hasattr(e, "members"):
            walk(e)
    for e in entities:
        type_refs = getattr(e, "type_refs", []) or []
        if "DocumentSection" in type_refs:
            sections.append(e)
    return sections


def _get_doc_texts(part: PartUsage) -> list[str]:
    """Get all doc texts from a part's members."""
    texts = []
    for m in getattr(part, "members", []) or []:
        if isinstance(m, Doc):
            texts.append(m.text)
    return texts


def _get_attr_value(part: PartUsage, attr_name: str) -> str:
    """Get a string attribute value from a part's members (quotes stripped)."""
    for m in getattr(part, "members", []) or []:
        if isinstance(m, AttributeUsage) and m.name == attr_name:
            raw = getattr(m, "value_expr", None)
            if raw:
                return str(raw).strip("'\" ")
    return ""


def sysml_search_sections(query: str, max_results: int = 20) -> Dict[str, Any]:
    """
    搜索文档章节（按标题或 doc 文本内容）。

    在所有 DocumentSection 实例中搜索匹配名称、标题或文档文本的章节。

    Args:
        query: 搜索关键词
        max_results: 最大返回数

    Returns:
        匹配的章节列表（含 ID、标题、摘要）
    """
    mgr = _get_manager()
    sections = _find_sections(mgr)
    query_lower = query.lower().strip()
    results = []

    for sec in sections:
        title = _get_attr_value(sec, "title")
        chapter = _get_attr_value(sec, "chapter")
        name = getattr(sec, "name", "") or ""
        docs = _get_doc_texts(sec)

        score = 0.0
        reason = ""
        match_text = ""

        # Name match
        if query_lower in name.lower():
            score = max(score, 1.0)
            reason = "name_match"
            match_text = name
        # Title match
        if title and query_lower in title.lower():
            if score < 0.9:
                score = max(score, 0.9)
                reason = "title_match"
                match_text = title
        # Doc text match
        for d in docs:
            if d and query_lower in d.lower():
                idx = d.lower().index(query_lower)
                start = max(0, idx - 40)
                end = min(len(d), idx + len(query) + 40)
                match_text = ("..." if start > 0 else "") + d[start:end] + ("..." if end < len(d) else "")
                if score < 0.7:
                    score = max(score, 0.7)
                    reason = "doc_match"
                break

        if score > 0:
            results.append({
                "id": name,
                "title": title or name,
                "chapter": chapter,
                "score": round(score, 3),
                "match_reason": reason,
                "snippet": match_text[:200],
            })

    results.sort(key=lambda x: -x["score"])
    results = results[:max_results]

    return {
        "ok": True,
        "query": query,
        "total_matches": len(results),
        "sections": results,
    }


def sysml_get_section(section_id: str, enrich: bool = True) -> Dict[str, Any]:
    """
    获取文档章节详情。

    返回章节的标题、章节号、原始 doc 文本，以及可选的超变量解析后的富文本。
    当 `enrich=True` 时，doc 文本中的 `<instvar:id:Name>` 等标签会被替换为实时值。

    Args:
        section_id: 章节 ID（如 sec4_1, sec5_3_1）
        enrich: 是否解析并注入超变量值

    Returns:
        章节详情（含原始文本和富文本）
    """
    mgr = _get_manager()
    sections = _find_sections(mgr)
    target = None
    for sec in sections:
        if getattr(sec, "name", "") == section_id:
            target = sec
            break

    if target is None:
        return {"ok": False, "error": f"Section not found: {section_id}"}

    title = _get_attr_value(target, "title")
    chapter = _get_attr_value(target, "chapter")
    name = getattr(target, "name", "") or ""
    docs = _get_doc_texts(target)
    full_text = "\n".join(docs)

    result = {
        "ok": True,
        "id": name,
        "title": title or name,
        "chapter": chapter,
        "doc_count": len(docs),
        "raw_text": full_text,
    }

    if enrich:
        resolver = _get_resolver()
        mgr = _get_manager()
        lookup = _build_hv_lookup(mgr, full_text)
        enriched = resolver.enrich_text(full_text, lookup=lookup)
        result["enriched_text"] = enriched

    return result


def sysml_resolve_hv(hv_id: str) -> Dict[str, Any]:
    """
    解析超变量，返回当前实时值。

    根据 Hypervariable 的 source/selector 配置，通过对应协议驱动获取值。

    Args:
        hv_id: 超变量 ID（如 fan_speed, SystemHealth, ThresholdParam）

    Returns:
        hv_id 对应的当前值
    """
    mgr = _get_manager()
    cfg = _get_hv_config(mgr, hv_id)
    if cfg is None:
        return {"ok": False, "error": f"Hypervariable not found: {hv_id}"}

    source = cfg.get("source", "")
    selector = cfg.get("selector", "")
    unit = cfg.get("unit", "")

    if not source and not selector:
        return {"ok": True, "id": hv_id, "type": cfg.get("hv_type", ""), "value": f"[{hv_id}]"}

    resolver = _get_resolver()
    value = resolver.resolve(source, selector)

    if unit and value and not value.startswith("[") and not value.startswith("["):
        value = f"{value} {unit}".strip()

    return {
        "ok": True,
        "id": hv_id,
        "type": cfg.get("hv_type", ""),
        "value": value,
        "source": source,
        "selector": selector,
    }


def sysml_set_parameter(param_id: str, value: str) -> Dict[str, Any]:
    """
    设置参数型超变量（Parameter）的值。

    向 source 指向的目标写入新值，影响系统行为。

    Args:
        param_id: 参数 ID（如 fan_curve, threshold）
        value: 要设置的字符串值

    Returns:
        写入结果
    """
    mgr = _get_manager()
    cfg = _get_hv_config(mgr, param_id)
    if cfg is None:
        return {"ok": False, "error": f"Parameter not found: {param_id}"}

    hv_type = cfg.get("hv_type", "")
    if hv_type != "Parameter":
        return {"ok": False, "error": f"'{param_id}' is a {hv_type}, not a Parameter"}

    source = cfg.get("source", "")
    selector = cfg.get("selector", "")
    resolver = _get_resolver()
    ok = resolver.write(source, selector, value)
    return {
        "ok": ok,
        "id": param_id,
        "set_value": value,
    }


def sysml_execute_operation(op_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    执行操作型超变量（Operation）。

    触发 source 指向的可执行操作，返回执行结果。

    Args:
        op_id: 操作 ID（如 underclock, LogDownload）
        params: 操作参数字典（可选）

    Returns:
        执行结果
    """
    mgr = _get_manager()
    cfg = _get_hv_config(mgr, op_id)
    if cfg is None:
        return {"ok": False, "error": f"Operation not found: {op_id}"}

    hv_type = cfg.get("hv_type", "")
    if hv_type != "Operation":
        return {"ok": False, "error": f"'{op_id}' is a {hv_type}, not an Operation"}

    source = cfg.get("source", "")
    selector = cfg.get("selector", "")
    resolver = _get_resolver()
    result = resolver.execute(source, selector, params or {})
    return {
        "ok": True,
        "id": op_id,
        "result": result,
    }


def sysml_list_hvs(entity_name: Optional[str] = None) -> Dict[str, Any]:
    """
    列出超变量（Hypervariable）。

    如果不指定 entity_name，返回所有超变量；如果指定，返回该实体的 ref 槽位中关联的超变量。

    Args:
        entity_name: 实体名称（可选）。指定时列出该实体通过 ref 关联的超变量。

    Returns:
        超变量列表
    """
    mgr = _get_manager()
    HV_TYPES = {"InstantVariable", "BlockVariable", "ModifierVariable", "Parameter", "Operation"}

    def walk(items):
        """Walk all elements including usages."""
        visited = set()
        def _w(items):
            for item in (items or []):
                iid = id(item)
                if iid in visited:
                    continue
                visited.add(iid)
                yield item
                for m in (getattr(item, "members", None) or []):
                    yield from _w([m])
        return _w(items)

    if entity_name:
        # List HVs linked via ref slots on this entity
        for item in walk(mgr.root_elements):
            if getattr(item, "name", None) != entity_name:
                continue
            hvs = []
            for m in getattr(item, "members", []) or []:
                if getattr(m, "is_reference", False):
                    hvs.append({
                        "slot": m.name,
                        "target": getattr(m, "value_expr", None) or "",
                        "type": (getattr(m, "type_refs", [None]) or [None])[0],
                    })
            return {"ok": True, "entity_name": entity_name, "total": len(hvs), "hypervariables": hvs}
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    # List ALL HVs in the model
    all_hvs = []
    for item in walk(mgr.root_elements):
        type_refs = getattr(item, "type_refs", []) or []
        for tr in type_refs:
            if tr in HV_TYPES:
                all_hvs.append({
                    "id": getattr(item, "name", ""),
                    "type": tr,
                    "qualified_name": getattr(item, "qualified_name", ""),
                })
                break
    return {"ok": True, "total": len(all_hvs), "hypervariables": all_hvs}


# ══════════════════════════════════════════════════════════════
# P0 手工注入工具 —— command / hostname / cabinet
# ══════════════════════════════════════════════════════════════

def sysml_add_command(
    name: str,
    command_text: str,
    target_device: Optional[str] = None,
    description: Optional[str] = None,
    invocation: Optional[str] = None,
    source_section: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    parent_package: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建一条运维命令实体 (CommandDef)。

    将 Shell 命令、CLI 操作等以 CommandDef 类型存入知识图谱，
    自动创建与 target_device 的 "executes" 连接。

    Args:
        name: 命令名 (如 "yhst", "smu_tranfer_cmd")
        command_text: 命令文本
        target_device: 命令执行目标设备名 (如 "CMU", "SMU", "ManagementNode")
        description: 命令功能描述
        invocation: 完整调用示例
        source_section: 文档出处 (如 "6.3")
        aliases: 别名列表
        parent_package: 父包限定名 (默认 "Commands")

    Returns:
        创建结果: qualified_name + 连接信息
    """
    mgr = _get_manager()
    pkg = parent_package or "Commands"
    props = {"command_text": command_text}
    if target_device:
        props["target_device"] = target_device
    if invocation:
        props["invocation"] = invocation
    if source_section:
        props["source_section"] = source_section

    element = mgr.add_entity_with_metadata(
        entity_type="CommandDef",
        name=name,
        parent_package=pkg,
        description=description,
        properties=props,
        aliases=aliases,
        source_sections=[source_section] if source_section else None,
    )
    if element is None:
        return {"ok": False, "error": "Failed to create CommandDef entity"}

    result = {
        "ok": True,
        "qualified_name": element.qualified_name,
        "name": element.name,
        "type": "CommandDef",
    }

    # Auto-connect to target_device
    if target_device:
        target = _find_any_element(mgr, target_device)
        if target:
            rel = mgr.add_relation(
                relation_type="connection",
                source_name=element.name,
                target_name=target_device,
                name=f"{element.name}_runs_on_{target_device}",
                parent_package=pkg,
                role_source="executed",
                role_target="target",
            )
            if rel:
                result["connected_to"] = target_device
                result["connection"] = rel.qualified_name

    return result


def sysml_set_hostname(
    entity_name: str,
    hostname: str,
) -> Dict[str, Any]:
    """
    为 SysML 实体设置 hostname 标识并注册别名。

    对 ManagementNode、LoginNode 等设置 hostname=mn0，
    使得 sysml_retrieve("mn0") 可以通过别名匹配到该实体。

    Args:
        entity_name: 实体名称 (如 "ManagementNode")
        hostname: 主机名标识 (如 "mn0")

    Returns:
        操作结果
    """
    mgr = _get_manager()

    entity = _find_any_element(mgr, entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    qn = entity.qualified_name

    # Update metadata with hostname property
    ok = mgr.update_entity_metadata(
        qualified_name=qn,
        update_properties={"_hostname": hostname},
    )
    if not ok:
        return {"ok": False, "error": f"Failed to update metadata for: {entity_name}"}

    # Register hostname as alias so sysml_retrieve("mn0") works
    mgr.add_alias(qn, hostname)

    return {
        "ok": True,
        "entity": entity_name,
        "qualified_name": qn,
        "hostname": hostname,
        "aliases": mgr._alias_registry.get_aliases(qn),
    }


def sysml_add_cabinet_instance(
    cabinet_id: str,
    cabinet_type: str = "CustomCabinet",
    sub_units: Optional[List[str]] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    parent_package: Optional[str] = None,
) -> Dict[str, Any]:
    """
    基于抽象机柜类型创建具体机柜实例。

    例如将 CustomCabinet 实例化为 R1P3，并创建 a/b/c/d 子机柜。

    Args:
        cabinet_id: 机柜编号 (如 "R1P3")
        cabinet_type: 机柜类型 (如 "CustomCabinet", "StandardCabinet")
        sub_units: 子机柜列表 (如 ["R1P3a", "R1P3b", "R1P3c", "R1P3d"])
        description: 机柜描述
        location: 物理位置
        parent_package: 父包 (默认 "Cabinets")

    Returns:
        创建的所有实体 qualified_names
    """
    mgr = _get_manager()
    pkg = parent_package or "Cabinets"
    props = {"cabinet_type": cabinet_type}
    if location:
        props["location"] = location

    cabinet = mgr.add_entity_with_metadata(
        entity_type="PartUsage",
        name=cabinet_id,
        parent_package=pkg,
        description=description,
        properties=props,
        aliases=[cabinet_id],
    )
    if cabinet is None:
        return {"ok": False, "error": f"Failed to create cabinet: {cabinet_id}"}

    # Set type_refs via update
    mgr.update_entity_metadata(
        qualified_name=cabinet.qualified_name,
        type_refs=[cabinet_type],
    )

    result = {
        "ok": True,
        "cabinet": cabinet.qualified_name,
        "type_ref": cabinet_type,
    }

    # Create sub-units
    if sub_units:
        sub_qns = []
        for sub in sub_units:
            sub_entity = mgr.add_entity_with_metadata(
                entity_type="PartUsage",
                name=sub,
                parent_package=pkg,
                description=f"{cabinet_id} 子机柜 {sub}",
                aliases=[sub],
            )
            if sub_entity:
                mgr.update_entity_metadata(
                    qualified_name=sub_entity.qualified_name,
                    type_refs=["PhysicalComponent"],
                )
            if sub_entity:
                sub_qns.append(sub_entity.qualified_name)
                # Connect sub-unit to parent cabinet
                mgr.add_relation(
                    relation_type="connection",
                    source_name=sub,
                    target_name=cabinet_id,
                    name=f"{sub}_in_{cabinet_id}",
                    parent_package=pkg,
                    role_source="child",
                    role_target="parent",
                )
        result["sub_units"] = sub_qns

    return result


# ══════════════════════════════════════════════════════════════
# P1 补充工具 —— chapter_ref / quantity / ip / display_name
# ══════════════════════════════════════════════════════════════

def sysml_add_chapter_ref(
    entity_name: str,
    chapter: str,
    section_title: str = "",
    page_range: str = "",
) -> Dict[str, Any]:
    """
    为实体添加文档章节引用，建立实体→文档出处的可追溯链接。

    Args:
        entity_name: 实体名称
        chapter: 章节号 (如 "6.3", "1.4")
        section_title: 章节标题 (如 "系统布局")
        page_range: 页码范围 (如 "7-9")

    Returns:
        操作结果
    """
    mgr = _get_manager()
    entity = _find_any_element(mgr, entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    qn = entity.qualified_name
    ref_text = f"第{chapter}节"
    if section_title:
        ref_text += f" {section_title}"
    if page_range:
        ref_text += f" (页码{page_range})"

    # Append to source_sections
    mgr.update_entity_metadata(
        qualified_name=qn,
        append_source_sections=[ref_text],
    )

    # Store chapter in properties for structured access
    meta = mgr.get_entity_metadata(qn)
    chapters = meta.get("properties", {}).get("_chapters", [])
    chapters.append(chapter)
    mgr.update_entity_metadata(
        qualified_name=qn,
        update_properties={"_chapters": chapters, "_source_chapter": chapter},
    )

    return {
        "ok": True,
        "entity": entity_name,
        "chapter_ref": ref_text,
    }


def sysml_add_quantity(
    entity_name: str,
    count: int,
    unit: str = "",
) -> Dict[str, Any]:
    """
    为实体设置精确的数量信息。

    Args:
        entity_name: 实体名称
        count: 数量
        unit: 单位 (如 "个", "台", "套")

    Returns:
        操作结果
    """
    mgr = _get_manager()
    entity = _find_any_element(mgr, entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    qn = entity.qualified_name
    mgr.update_entity_metadata(
        qualified_name=qn,
        update_properties={"_count": count, "_count_unit": unit or "个"},
    )

    # Also set multiplicity on Usage entities
    if isinstance(entity, Usage):
        from sysml.sysml_model import Multiplicity
        entity.multiplicity = Multiplicity(lower=str(count), upper=str(count))

    return {
        "ok": True,
        "entity": entity_name,
        "count": count,
        "unit": unit or "个",
    }


def sysml_add_ip_config(
    entity_name: str,
    ip_address: str = "",
    subnet: str = "",
    gateway: str = "",
    dns: str = "",
    description: str = "",
) -> Dict[str, Any]:
    """
    为实体添加 IP 网络配置信息。

    Args:
        entity_name: 实体名称 (如 ManagementNode, CMU)
        ip_address: IP 地址
        subnet: 子网掩码
        gateway: 网关
        dns: DNS 服务器
        description: 网络描述

    Returns:
        操作结果
    """
    mgr = _get_manager()
    entity = _find_any_element(mgr, entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    qn = entity.qualified_name
    ip_config = {}
    if ip_address:
        ip_config["ip"] = ip_address
    if subnet:
        ip_config["subnet"] = subnet
    if gateway:
        ip_config["gateway"] = gateway
    if dns:
        ip_config["dns"] = dns
    if description:
        ip_config["description"] = description

    mgr.update_entity_metadata(
        qualified_name=qn,
        update_properties={"_ip_config": ip_config},
    )

    # Register IP as alias for easy lookup
    if ip_address:
        mgr.add_alias(qn, ip_address)

    return {
        "ok": True,
        "entity": entity_name,
        "ip_config": ip_config,
    }


def sysml_set_display_name(
    entity_name: str,
    display_name: str,
) -> Dict[str, Any]:
    """
    为实体设置中文显示名并注册别名，防止 CamelCase 丢失语义。

    Args:
        entity_name: 实体名称 (如 "compute_module")
        display_name: 中文显示名 (如 "计算模块")

    Returns:
        操作结果
    """
    mgr = _get_manager()
    entity = _find_any_element(mgr, entity_name)
    if entity is None:
        return {"ok": False, "error": f"Entity not found: {entity_name}"}

    qn = entity.qualified_name
    mgr.update_entity_metadata(
        qualified_name=qn,
        update_properties={"_display_name": display_name},
    )

    # Register display_name as alias
    mgr.add_alias(qn, display_name)

    return {
        "ok": True,
        "entity": entity_name,
        "display_name": display_name,
        "aliases": mgr._alias_registry.get_aliases(qn),
    }


# ══════════════════════════════════════════════════════════════
# 核心检索工具 —— sysml_retrieve
# ══════════════════════════════════════════════════════════════

def sysml_retrieve(name: str, k: int = 2) -> Dict[str, Any]:
    """
    根据类名、别名或实例名检索SysML实体，返回完整属性和k层关系图。

    这是检索SysML模型的核心工具。一次调用即可获取：
    - 实体的完整属性（定义、元数据、成员树、别名）
    - k层关系图（BFS遍历的邻接实体和连接关系）

    Args:
        name: 类名、别名或实例名（支持中文/英文/拼音）
        k: 关系图遍历深度。
           0 = 仅实体本身
           1 = 直接关联关系
           2 = 两层关系（默认）
           3+ = 多层关系

    Returns:
        实体的完整属性和k层关系图
    """
    mgr = _get_manager()

    entity = _find_any_element(mgr, name)
    if entity is None:
        return {
            "ok": False,
            "error": f"Entity not found: '{name}'. Try a different name, alias, or check that a model is loaded.",
            "search_attempted": name,
        }

    entity_detail = _entity_detail_full(entity, mgr)
    graph = _build_k_layer_graph(mgr, entity.name, k)

    return {
        "ok": True,
        "matched_by": entity.name,
        "entity": entity_detail,
        "relationship_graph": graph,
    }


# ══════════════════════════════════════════════════════════════
# MCP 服务器入口
# ══════════════════════════════════════════════════════════════

# 工具元数据（供 MCP/LangChain 使用）
TOOL_DEFINITIONS = {
    # ── 核心检索 ──
    "sysml_retrieve": {
        "function": sysml_retrieve,
        "description": "根据类名/别名/实例名检索SysML实体，返回完整属性（定义、元数据、成员树、别名）和k层关系图（BFS遍历的邻接实体和连接关系）。这是检索SysML模型的主工具。",
        "parameters": {
            "name": {"type": "string", "description": "类名、别名或实例名（支持中文、英文、拼音）"},
            "k": {"type": "integer", "description": "关系图遍历深度: 0=仅实体本身, 1=直接关联, 2=两层关系, ...", "default": 2},
        },
    },
    # ── 模型 I/O ──
    "sysml_load_model": {
        "function": sysml_load_model,
        "description": "加载一个 .sysml 模型文件到当前会话",
        "parameters": {
            "file_path": {"type": "string", "description": ".sysml 文件路径"},
        },
    },
    "sysml_save_model": {
        "function": _sysml_save_model,
        "description": "保存当前模型到 .sysml 文件（同时写入 .meta.json 元数据）",
        "parameters": {
            "file_path": {"type": "string", "description": "输出 .sysml 路径（默认使用当前文件）", "default": None},
        },
    },
    # ── 模型 I/O ──
    "sysml_load_model": {
        "function": sysml_load_model,
        "description": "加载一个 .sysml 模型文件到当前会话",
        "parameters": {
            "file_path": {"type": "string", "description": ".sysml 文件路径"},
        },
    },
    "sysml_save_model": {
        "function": _sysml_save_model,
        "description": "保存当前模型到 .sysml 文件（同时写入 .meta.json 元数据）",
        "parameters": {
            "file_path": {"type": "string", "description": "输出 .sysml 路径（默认使用当前文件）", "default": None},
        },
    },
    # ── 实体 CRUD ──
    "sysml_add_entity": {
        "function": sysml_add_entity,
        "description": "创建新实体（PartDef/AttributeDef/RequirementDef/PortDef/ItemDef/Package 及对应的 Usage），含别名、描述、来源、属性",
        "parameters": {
            "entity_type": {"type": "string", "description": "实体类型: PartDef, PartUsage, AttributeDef, AttributeUsage, PortDef, PortUsage, ItemDef, ItemUsage, RequirementDef, RequirementUsage, Package"},
            "name": {"type": "string", "description": "实体名称"},
            "parent_package": {"type": "string", "description": "父包限定名（可选）", "default": None},
            "description": {"type": "string", "description": "实体描述（可选）", "default": None},
            "aliases": {"type": "array", "items": {"type": "string"}, "description": "别名列表（可选）", "default": None},
            "source_sections": {"type": "array", "items": {"type": "string"}, "description": "来源章节列表（可选）", "default": None},
            "source_text": {"type": "string", "description": "原始出处文本（可选）", "default": None},
            "properties": {"type": "object", "description": "属性字典（可选）", "default": None},
            "supertypes": {"type": "array", "items": {"type": "string"}, "description": "父类型列表（仅 Definition）", "default": None},
            "short_name": {"type": "string", "description": "短名称（可选）", "default": None},
        },
    },
    "sysml_update_entity": {
        "function": sysml_update_entity,
        "description": "更新实体（追加描述、合并别名、追加来源、更新属性、重命名）",
        "parameters": {
            "qualified_name": {"type": "string", "description": "实体限定名"},
            "append_description": {"type": "string", "description": "追加描述文本（可选）", "default": None},
            "append_source_sections": {"type": "array", "items": {"type": "string"}, "description": "追加来源章节（可选）", "default": None},
            "merge_aliases": {"type": "array", "items": {"type": "string"}, "description": "合并别名（可选）", "default": None},
            "update_properties": {"type": "object", "description": "更新属性（可选）", "default": None},
            "new_name": {"type": "string", "description": "重命名（可选）", "default": None},
            "supertypes": {"type": "array", "items": {"type": "string"}, "description": "更新父类型（可选）", "default": None},
            "type_refs": {"type": "array", "items": {"type": "string"}, "description": "更新类型引用（可选）", "default": None},
        },
    },
    "sysml_delete_entity": {
        "function": sysml_delete_entity,
        "description": "删除指定实体（含别名和元数据）",
        "parameters": {
            "qualified_name": {"type": "string", "description": "实体限定名"},
        },
    },
    "sysml_normalize_name": {
        "function": sysml_normalize_name,
        "description": "对名称进行归一化处理（去标点、小写、全半角统一），用于名称比较",
        "parameters": {
            "name": {"type": "string", "description": "原始名称"},
        },
    },
    "sysml_add_alias": {
        "function": sysml_add_alias,
        "description": "为已有实体追加别名",
        "parameters": {
            "qualified_name": {"type": "string", "description": "实体限定名"},
            "alias": {"type": "string", "description": "新别名"},
        },
    },
    # ── 搜索 ──
    "sysml_search_entity": {
        "function": sysml_search_entity,
        "description": "多策略搜索实体（精确→归一化→别名→子串→正则→Token重叠），返回匹配列表与置信度",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"},
            "threshold": {"type": "number", "description": "最低置信度阈值 (0.0~1.0)", "default": 0.3},
            "regex_pattern": {"type": "string", "description": "可选正则辅助匹配", "default": None},
        },
    },
    "sysml_list_entities": {
        "function": sysml_list_entities,
        "description": "列出当前模型中所有实体定义（部件/属性/需求等）",
        "parameters": {
            "include_details": {"type": "boolean", "description": "是否包含详细成员", "default": False},
        },
    },
    "sysml_get_entity": {
        "function": sysml_get_entity,
        "description": "获取实体详情（含子特征和关联关系）",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
        },
    },
    # ── 关系 CRUD ──
    "sysml_add_relation": {
        "function": sysml_add_relation,
        "description": "创建关系（connection/interface/allocation），可选来源追踪",
        "parameters": {
            "relation_type": {"type": "string", "description": "关系类型: connection, interface, allocation, containment, composition, reference, generalization, dependency, abstraction, realization, derive, trace, derivereqt, refine, satisfy, verify, copy, usecaseassociation, usecaseinclude, usecaseextend"},
            "source": {"type": "string", "description": "源实体名称"},
            "target": {"type": "string", "description": "目标实体名称"},
            "name": {"type": "string", "description": "关系名称（可选）", "default": None},
            "parent_package": {"type": "string", "description": "父包限定名（可选）", "default": None},
            "description": {"type": "string", "description": "关系描述（可选）", "default": None},
            "role_source": {"type": "string", "description": "源端角色名（可选）", "default": None},
            "role_target": {"type": "string", "description": "目标端角色名（可选）", "default": None},
            "source_sections": {"type": "array", "items": {"type": "string"}, "description": "来源小节列表（可选）", "default": None},
            "source_text": {"type": "string", "description": "来源原文（可选）", "default": None},
        },
    },
    "sysml_delete_relation": {
        "function": sysml_delete_relation,
        "description": "删除指定关系",
        "parameters": {
            "name": {"type": "string", "description": "关系名称"},
            "parent_package": {"type": "string", "description": "父包限定名（可选）", "default": None},
        },
    },
    "sysml_get_connections": {
        "function": sysml_get_connections,
        "description": "获取与指定实体相关的所有连接关系",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
        },
    },
    "sysml_list_relations": {
        "function": sysml_list_relations,
        "description": "列出当前模型中所有关系（连接/接口/分配）",
        "parameters": {
            "include_details": {"type": "boolean", "description": "是否包含详细信息", "default": False},
        },
    },
    # ── 合并/去重 ──
    "sysml_suggest_merge": {
        "function": sysml_suggest_merge,
        "description": "分析模型中的重复实体，返回合并建议列表（基于名称相似度和共享别名）",
        "parameters": {
            "threshold": {"type": "number", "description": "最低置信度阈值 (0.0~1.0)", "default": 0.6},
        },
    },
    "sysml_merge_entities": {
        "function": sysml_merge_entities,
        "description": "将 source 实体合并到 target 实体（转移别名、描述、来源、属性后删除 source）",
        "parameters": {
            "source": {"type": "string", "description": "源实体限定名（将被删除）"},
            "target": {"type": "string", "description": "目标实体限定名（保留）"},
        },
    },
    # ── 其他 ──
    "sysml_export_submodel": {
        "function": sysml_export_submodel,
        "description": "导出以实体为中心的子模型视图（SysML 文本）",
        "parameters": {
            "entity_name": {"type": "string", "description": "核心实体名称"},
            "depth": {"type": "integer", "description": "导出深度", "default": 2},
        },
    },
    "sysml_import_doc": {
        "function": sysml_import_doc,
        "description": "从文档（PDF/DOCX/TXT/MD/XLSX）导入并构建 SysML 模型",
        "parameters": {
            "file_path": {"type": "string", "description": "源文档路径"},
            "output_path": {"type": "string", "description": "输出 .sysml 路径（可选）", "default": None},
        },
    },
    "sysml_semantic_search": {
        "function": sysml_semantic_search,
        "description": "在 SysML 模型中执行语义搜索（名称+内容）",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"},
            "search_content": {"type": "boolean", "description": "是否搜索成员内容", "default": True},
        },
    },
    "sysml_model_summary": {
        "function": sysml_model_summary,
        "description": "获取当前已加载模型的全局摘要统计",
        "parameters": {},
    },
    "sysml_connected_components": {
        "function": sysml_connected_components,
        "description": "计算知识图谱连通分量（子图），返回各分量实体列表（按大小升序）",
        "parameters": {},
    },
    # ── 超变量 / 文档章节 ──
    "sysml_search_sections": {
        "function": sysml_search_sections,
        "description": "搜索文档章节（按标题或 doc 文本内容），返回匹配的章节 ID、标题和摘要",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "最大返回数", "default": 20},
        },
    },
    "sysml_get_section": {
        "function": sysml_get_section,
        "description": "获取文档章节详情。返回原始 doc 文本，选择性地返回超变量解析后的富文本（enrich=True 时自动注入实时值）",
        "parameters": {
            "section_id": {"type": "string", "description": "章节 ID（如 sec4_1, sec5_3_1）"},
            "enrich": {"type": "boolean", "description": "是否解析并注入超变量值", "default": True},
        },
    },
    "sysml_resolve_hv": {
        "function": sysml_resolve_hv,
        "description": "解析超变量，返回当前实时值。支持 InstantVariable（即时读取）、BlockVariable（阻塞读取）",
        "parameters": {
            "hv_id": {"type": "string", "description": "超变量 ID（如 fan_speed, SystemHealth）"},
        },
    },
    "sysml_set_parameter": {
        "function": sysml_set_parameter,
        "description": "设置参数型超变量（Parameter）的值，影响系统行为",
        "parameters": {
            "param_id": {"type": "string", "description": "参数 ID（如 fan_curve, threshold）"},
            "value": {"type": "string", "description": "要设置的字符串值"},
        },
    },
    "sysml_execute_operation": {
        "function": sysml_execute_operation,
        "description": "执行操作型超变量（Operation），触发可执行操作并返回执行结果",
        "parameters": {
            "op_id": {"type": "string", "description": "操作 ID（如 underclock, LogDownload）"},
            "params": {"type": "object", "description": "操作参数（可选）", "default": None},
        },
    },
    "sysml_list_hvs": {
        "function": sysml_list_hvs,
        "description": "列出超变量。不指定 entity_name 时返回所有超变量；指定时返回该实体 ref 槽位关联的超变量",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称（可选）", "default": None},
        },
    },
    # ── P0 注入工具 ──
    "sysml_add_command": {
        "function": sysml_add_command,
        "description": "创建运维命令实体(CommandDef)。将Shell命令/CLI操作存入知识图谱，自动连接目标设备。",
        "parameters": {
            "name": {"type": "string", "description": "命令名 (如 yhst, smu_tranfer_cmd)"},
            "command_text": {"type": "string", "description": "命令文本"},
            "target_device": {"type": "string", "description": "执行目标设备名", "default": None},
            "description": {"type": "string", "description": "命令功能描述", "default": None},
            "invocation": {"type": "string", "description": "完整调用示例", "default": None},
            "source_section": {"type": "string", "description": "文档出处 (如 6.3)", "default": None},
            "aliases": {"type": "array", "items": {"type": "string"}, "description": "别名列表", "default": None},
            "parent_package": {"type": "string", "description": "父包 (默认 Commands)", "default": None},
        },
    },
    "sysml_set_hostname": {
        "function": sysml_set_hostname,
        "description": "为实体设置hostname标识并注册别名，使sysml_retrieve可通过主机名匹配实体。",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称 (如 ManagementNode)"},
            "hostname": {"type": "string", "description": "主机名 (如 mn0)"},
        },
    },
    "sysml_add_cabinet_instance": {
        "function": sysml_add_cabinet_instance,
        "description": "基于抽象机柜类型创建具体机柜实例。例如将CustomCabinet实例化为R1P3并创建a/b/c/d子机柜。",
        "parameters": {
            "cabinet_id": {"type": "string", "description": "机柜编号 (如 R1P3)"},
            "cabinet_type": {"type": "string", "description": "机柜类型 (如 CustomCabinet)", "default": "CustomCabinet"},
            "sub_units": {"type": "array", "items": {"type": "string"}, "description": "子机柜列表", "default": None},
            "description": {"type": "string", "description": "机柜描述", "default": None},
            "location": {"type": "string", "description": "物理位置", "default": None},
            "parent_package": {"type": "string", "description": "父包 (默认 Cabinets)", "default": None},
        },
    },
    # ── P1 补充工具 ──
    "sysml_add_chapter_ref": {
        "function": sysml_add_chapter_ref,
        "description": "为实体添加文档章节引用，建立实体→文档出处的可追溯链接。",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
            "chapter": {"type": "string", "description": "章节号 (如 6.3, 1.4)"},
            "section_title": {"type": "string", "description": "章节标题", "default": ""},
            "page_range": {"type": "string", "description": "页码范围", "default": ""},
        },
    },
    "sysml_add_quantity": {
        "function": sysml_add_quantity,
        "description": "为实体设置精确的数量信息。如216个计算结点、2个管理结点。",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
            "count": {"type": "integer", "description": "数量"},
            "unit": {"type": "string", "description": "单位 (如 个)", "default": "个"},
        },
    },
    "sysml_add_ip_config": {
        "function": sysml_add_ip_config,
        "description": "为实体添加IP网络配置信息。主机IP、子网、网关等。",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称 (如 ManagementNode, CMU)"},
            "ip_address": {"type": "string", "description": "IP地址", "default": ""},
            "subnet": {"type": "string", "description": "子网掩码", "default": ""},
            "gateway": {"type": "string", "description": "网关", "default": ""},
            "dns": {"type": "string", "description": "DNS服务器", "default": ""},
            "description": {"type": "string", "description": "网络描述", "default": ""},
        },
    },
    "sysml_set_display_name": {
        "function": sysml_set_display_name,
        "description": "为实体设置中文显示名并注册别名，防止CamelCase丢失语义。如compute_module → 计算模块。",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称 (如 compute_module)"},
            "display_name": {"type": "string", "description": "中文显示名 (如 计算模块)"},
        },
    },
}


# ── Agent tool filtering lists ───────────────────────────────
# QA / Retrieval agents: sysml_retrieve + load
QUERY_AGENT_TOOL_NAMES = [
    "sysml_retrieve",
    "sysml_load_model",
    "sysml_add_command",
    "sysml_set_hostname",
    "sysml_add_cabinet_instance",
    "sysml_add_chapter_ref",
    "sysml_add_quantity",
    "sysml_add_ip_config",
    "sysml_set_display_name",
    "sysml_resolve_hv",
]

# Build agents: all tools available (via ENTITY_TOOL_NAMES etc in kg_build_agent.py)
# Keep existing lists from kg_build_agent.py for compatibility
BUILD_ENTITY_TOOL_NAMES = [
    "sysml_search_entity",
    "sysml_add_entity",
    "sysml_update_entity",
    "sysml_add_alias",
    "sysml_normalize_name",
    "sysml_list_entities",
    "sysml_model_summary",
]
BUILD_RELATION_TOOL_NAMES = [
    "sysml_add_relation",
    "sysml_search_entity",
    "sysml_get_connections",
    "sysml_list_relations",
    "sysml_list_entities",
    "sysml_model_summary",
]
BUILD_MERGE_TOOL_NAMES = [
    "sysml_suggest_merge",
    "sysml_merge_entities",
    "sysml_list_entities",
    "sysml_search_entity",
]
BUILD_ALL_TOOL_NAMES = list(TOOL_DEFINITIONS.keys())


def _run_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """根据工具名称和参数执行并返回 JSON 字符串"""
    tool_def = TOOL_DEFINITIONS.get(tool_name)
    if tool_def is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

    func = tool_def["function"]
    try:
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


# ── MCP stdio 服务器 ────────────────────────────────────────
# 遵循 MCP 协议：通过 stdin/stdout 收发 JSON-RPC
def _mcp_serve() -> None:
    """启动 MCP stdio 服务器"""
    print("[SysML RAG MCP] Server starting on stdio...", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")

        # JSONRPC notification — 不响应
        if req_id is None:
            continue

        if method == "tools/list":
            tools = []
            for name, defn in TOOL_DEFINITIONS.items():
                properties = {}
                required = []
                for k, v in defn["parameters"].items():
                    entry = {"type": v["type"], "description": v.get("description", "")}
                    if "default" in v:
                        entry["default"] = v["default"]
                    else:
                        required.append(k)
                    if v.get("items"):
                        entry["items"] = v["items"]
                    properties[k] = entry
                tools.append({
                    "name": name,
                    "description": defn["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                })
            response = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result_text = _run_tool(tool_name, arguments)
            response = json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }, ensure_ascii=False)

        elif method == "initialize":
            response = json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "sysml-rag-mcp",
                        "version": "0.1.0",
                    },
                },
            })

        else:
            response = json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

        print(response, flush=True)


# ── CLI 入口 ─────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="SysML RAG 检索 MCP 服务器 / 命令行工具",
    )
    sub = parser.add_subparsers(dest="command")

    # serve: 启动 MCP 服务器
    sub.add_parser("serve", help="启动 MCP stdio 服务器")

    # run: 单次工具调用
    run_parser = sub.add_parser("run", help="执行单次工具调用")
    run_parser.add_argument("tool", choices=list(TOOL_DEFINITIONS.keys()), help="工具名称")
    run_parser.add_argument("--args", default="{}", help="JSON 参数字符串")

    args = parser.parse_args()

    if args.command == "serve":
        _mcp_serve()
    elif args.command == "run":
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError:
            print(f"Invalid JSON args: {args.args}", file=sys.stderr)
            sys.exit(1)
        result = _run_tool(args.tool, tool_args)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()