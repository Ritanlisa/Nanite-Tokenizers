#!/usr/bin/env python3
"""
SysML RAG 检索 MCP 服务器
——面向 SysML v2 模型的知识检索工具

提供工具：
  1. sysml_load_model       - 加载 .sysml 模型文件
  2. sysml_list_entities    - 列出所有实体（部件/属性/需求等）
  3. sysml_list_relations   - 列出所有关系（连接/接口/分配）
  4. sysml_search_entity    - 按名称搜索实体（支持模糊匹配）
  5. sysml_get_entity       - 获取实体详情（含子特征）
  6. sysml_get_connections  - 获取某实体的所有连接关系
  7. sysml_export_submodel  - 导出子模型（以实体为中心的局部视图）
  8. sysml_import_doc       - 从文档导入并构建 SysML 模型
  9. sysml_semantic_search  - 基于内容的语义搜索（遍历树结构）

可作为独立 MCP 服务器运行，也可在 agent/tools.py 中注册为 LangChain 工具。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 确保项目根在 path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sysml.sysml_model import (
    Package, SysMLElement, Namespace, Definition, Usage,
    ConnectionUsage, ConnectionEnd, InterfaceUsage, AllocationUsage,
    InterfaceDef, AllocationDef,
    PartDef, PartUsage, AttributeDef, AttributeUsage,
    PortDef, PortUsage, ItemDef, ItemUsage,
    RequirementDef, RequirementUsage,
    ConnectionDef,
)
from sysml.sysml_manager import SysMLManager

# ── 全局模型注册表 ───────────────────────────────────────────
_global_manager: Optional[SysMLManager] = None
_loaded_files: Dict[str, str] = {}  # file_stem → file_path


def _get_manager() -> SysMLManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = SysMLManager(workspace_root=ROOT_DIR)
    return _global_manager


def _entity_type_name(entity: Any) -> str:
    """返回人类可读的实体类型名称"""
    type_map = {
        PartDef: "部件定义", PartUsage: "部件使用",
        AttributeDef: "属性定义", AttributeUsage: "属性使用",
        PortDef: "端口定义", PortUsage: "端口使用",
        ItemDef: "项定义", ItemUsage: "项使用",
        ConnectionDef: "连接定义", ConnectionUsage: "连接使用",
        InterfaceUsage: "接口使用", InterfaceDef: "接口定义",
        AllocationUsage: "分配使用", AllocationDef: "分配定义",
        RequirementDef: "需求定义", RequirementUsage: "需求使用",
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

    if include_body and hasattr(entity, "members"):
        member_summaries = []
        for m in entity.members:
            member_summaries.append(_entity_summary(m, include_body=False))
        if member_summaries:
            info["members"] = member_summaries

    return info


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


def sysml_search_entity(query: str) -> Dict[str, Any]:
    """
    按名称模糊搜索实体。

    Args:
        query: 搜索关键词（支持部分名称、拼音、首字母等）

    Returns:
        匹配的实体列表
    """
    mgr = _get_manager()
    entities = mgr.get_all_entities()
    entity_names: Dict[str, Definition] = {}

    for entity in entities:
        name = getattr(entity, "name", "")
        if name:
            # 可能有重名的情况，保留第一个
            if name not in entity_names:
                entity_names[name] = entity

    matches = _fuzzy_match_name(query, list(entity_names.keys()))

    result = {
        "query": query,
        "total_matches": len(matches),
        "matches": [],
    }

    for name, score in matches:
        entity = entity_names[name]
        summary = _entity_summary(entity, include_body=False)
        summary["match_score"] = round(score, 3)
        result["matches"].append(summary)

    return result


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
            for end in item.ends:
                collected.add(end.ref)

        if hasattr(item, "members"):
            for m in item.members:
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
    try:
        from scripts.demo_doc_to_sysml import build_sysml_model_from_doc_tree
    except ImportError as exc:
        return {"ok": False, "error": f"Failed to import doc_to_sysml: {exc}"}

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
                reasons.append(f"short_name_match")

        # 搜索成员
        if search_content and hasattr(entity, "members"):
            member_texts = []
            for m in entity.members:
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

    # 顶层元素
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


# ══════════════════════════════════════════════════════════════
# MCP 服务器入口
# ══════════════════════════════════════════════════════════════

# 工具元数据（供 MCP/LangChain 使用）
TOOL_DEFINITIONS = {
    "sysml_load_model": {
        "function": sysml_load_model,
        "description": "加载一个 .sysml 模型文件到当前会话",
        "parameters": {
            "file_path": {"type": "string", "description": ".sysml 文件路径"},
        },
    },
    "sysml_list_entities": {
        "function": sysml_list_entities,
        "description": "列出当前模型中所有实体定义（部件/属性/需求等）",
        "parameters": {
            "include_details": {"type": "boolean", "description": "是否包含详细成员", "default": False},
        },
    },
    "sysml_list_relations": {
        "function": sysml_list_relations,
        "description": "列出当前模型中所有关系（连接/接口/分配）",
        "parameters": {
            "include_details": {"type": "boolean", "description": "是否包含详细信息", "default": False},
        },
    },
    "sysml_search_entity": {
        "function": sysml_search_entity,
        "description": "按名称模糊搜索实体",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"},
        },
    },
    "sysml_get_entity": {
        "function": sysml_get_entity,
        "description": "获取实体详情（含子特征和关联关系）",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
        },
    },
    "sysml_get_connections": {
        "function": sysml_get_connections,
        "description": "获取与指定实体相关的所有连接关系",
        "parameters": {
            "entity_name": {"type": "string", "description": "实体名称"},
        },
    },
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
}


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
    print("[SysML RAG MCP] Server starting on stdio...", flush=True)

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

        if method == "tools/list":
            tools = []
            for name, defn in TOOL_DEFINITIONS.items():
                tools.append({
                    "name": name,
                    "description": defn["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            k: {"type": v["type"], "description": v["description"]}
                            for k, v in defn["parameters"].items()
                        },
                        "required": list(defn["parameters"].keys()),
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