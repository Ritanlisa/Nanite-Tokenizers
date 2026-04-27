"""
Demo: 从文档树提取内容到 SysML v2 (.sysml) 文件
——基于 test_doc_tree_debug_gui.py 的构建数据库算法

提取策略：
1. 构建文档树（复用 _build_payload_and_rag_doc_from_file）
2. 按章节（chapter）分节遍历内容节点（content）
3. 每节创建一个 SysML Package
4. 从节标题和 markdown 内容中提取实体（part/attribute/requirement）
5. 在同一 Package 内建立实体间的关系（connection/containment）
6. 序列化为 .sysml 文本并持久化
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sysml.sysml_model import (
    Package,
    PartDef,
    AttributeDef,
    AttributeUsage,
    RequirementDef,
    ConnectionUsage,
    ConnectionEnd,
)
from sysml.sysml_manager import SysMLManager

from scripts.test_doc_tree_debug_gui import (
    _build_payload_and_rag_doc_from_file,
    SUPPORTED_RAG_EXTENSIONS,
)

# 项目根目录，用于 SysML 管理器的 workspace_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 实体提取正则 ──────────────────────────────────────────────
# 识别常见中文技术文档中的实体模式
_RE_PART_PATTERN = re.compile(
    r"(?:部件|组件|零件|模块|装置|总成|系统|子系统|单元|机构|元件|设备|仪器)"
    r"[\s：:]*[「『\"']?([\u4e00-\u9fff\w\-+/]+)[」』\"']?",
)
_RE_ATTRIBUTE_PATTERN = re.compile(
    r"(?:参数|属性|特性|指标|规格|尺寸|重量|功率|电压|电流|温度|压力|频率|速度|容量|精度)"
    r"[\s：:]*[「『\"']?([\u4e00-\u9fff\w\-+/]+)[」』\"']?",
)
_RE_REQUIREMENT_PATTERN = re.compile(
    r"(?:要求|需求|条件|约束|指标要求|技术条件|规范)"
    r"[\s：:]*[「『\"']?([\u4e00-\u9fff\w\-+/]+)[」』\"']?",
)

# SysML 标识符只允许 [a-zA-Z_][a-zA-Z0-9_]* 或 '...'
def _safe_sysml_name(raw: str) -> str:
    """将任意文本转换为合法的 SysML 名称（优先取英文，否则拼音/编号）"""
    raw = raw.strip()
    if not raw:
        return "unnamed"
    # 提取英文字母、数字、下划线
    ascii_part = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_")
    if ascii_part and re.match(r"^[a-zA-Z_]", ascii_part):
        return ascii_part[:64]
    # 全中文/特殊字符 → 用引号包裹
    safe = re.sub(r"['\n\r\t]+", "", raw)[:64]
    if safe:
        return f"'{safe}'"
    return "unnamed"


def _extract_entities_from_text(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    从文本中提取实体类别。
    返回: (parts, attributes, requirements)，每个元素为 (原始标签, 规范化名称)
    """
    parts: List[Tuple[str, str]] = []
    attrs: List[Tuple[str, str]] = []
    reqs: List[Tuple[str, str]] = []

    seen_parts: Set[str] = set()
    seen_attrs: Set[str] = set()
    seen_reqs: Set[str] = set()

    for m in _RE_PART_PATTERN.finditer(text):
        label = m.group(1).strip()
        name = _safe_sysml_name(label)
        if name not in seen_parts and name != "unnamed":
            seen_parts.add(name)
            parts.append((label, name))

    for m in _RE_ATTRIBUTE_PATTERN.finditer(text):
        label = m.group(1).strip()
        name = _safe_sysml_name(label)
        if name not in seen_attrs and name != "unnamed":
            seen_attrs.add(name)
            attrs.append((label, name))

    for m in _RE_REQUIREMENT_PATTERN.finditer(text):
        label = m.group(1).strip()
        name = _safe_sysml_name(label)
        if name not in seen_reqs and name != "unnamed":
            seen_reqs.add(name)
            reqs.append((label, name))

    return parts, attrs, reqs


# ── 文档树 → SysML 模型 ─────────────────────────────────────
def _collect_section_nodes(
    nodes: List[Dict[str, Any]],
    out_sections: List[Dict[str, Any]],
) -> None:
    """
    递归收集章节节点及其下属内容节点。
    每个 section 包含：
    - chapter: 章节节点信息
    - content_nodes: 该章节下的所有内容节点
    - markdown_text: 汇总的 markdown 文本
    """
    for node in nodes:
        category = str(node.get("category") or "")
        if category == "chapter":
            content_nodes: List[Dict[str, Any]] = []
            _collect_content_nodes(node, content_nodes)
            if content_nodes:
                content_nodes.sort(
                    key=lambda item: int((item.get("metadata") or {}).get("page") or 10**9)
                )
                merged_text = "\n\n".join(
                    str((item.get("variables") or {}).get("markdown_text") or "")
                    for item in content_nodes
                )
                out_sections.append({
                    "chapter_title": str(node.get("title") or ""),
                    "chapter_meta": dict(node.get("metadata") or {}),
                    "content_nodes": content_nodes,
                    "markdown_text": merged_text,
                })
        # 继续深入子级（有些文档章下无直接 content，但有子 chapter）
        children = node.get("children") or []
        if isinstance(children, list) and children:
            _collect_section_nodes(children, out_sections)


def _collect_content_nodes(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    """递归收集 category=="content" 的节点"""
    if str(node.get("category") or "") == "content":
        out.append(node)
    for child in list(node.get("children") or []):
        _collect_content_nodes(child, out)


def _safe_section_package_name(title: str, index: int) -> str:
    """将章节标题转换为合法的 SysML Package 名称"""
    if not title or not title.strip():
        return f"Section{index + 1}"
    # 尝试提取英文
    ascii_part = re.sub(r"[^a-zA-Z0-9_]+", "_", title.strip()).strip("_")
    if ascii_part and re.match(r"^[a-zA-Z_]", ascii_part):
        return ascii_part[:64]
    # 中文标题 → 用序号
    return f"Section{index + 1}"


def build_sysml_model_from_doc_tree(file_path: str) -> Tuple[SysMLManager, Dict[str, Any]]:
    """
    从文档文件构建 SysML v2 模型。

    步骤：
    1. 使用文档树构建器获取结构化内容
    2. 按章节分节
    3. 每节创建 Package，内含提取到的实体
    4. 建立实体内关系
    """
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"文件不存在: {source}")
    ext = source.suffix.lower()
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        raise RuntimeError(f"不支持的文档类型: {ext}")

    print(f"[DocToSysML] 构建文档树: {source.name}")
    payload, rag_doc = _build_payload_and_rag_doc_from_file(
        str(source),
        include_build_debug=False,
        emit_summary=True,
        emit_tree=False,
        emit_boundary_debug=False,
    )

    tree = payload.get("tree") or []
    if not tree:
        raise RuntimeError("文档树为空，无法提取 SysML 模型")

    print(f"[DocToSysML] 文档标题: {payload.get('title') or source.name}")
    print(f"[DocToSysML] 页数: {payload.get('page_count')}")

    # 收集章节节点
    sections: List[Dict[str, Any]] = []
    _collect_section_nodes(tree, sections)

    if not sections:
        # 如果没有章节划分，将整个文档作为一个 section
        all_content: List[Dict[str, Any]] = []
        _collect_content_nodes({"children": tree, "category": "", "title": "", "metadata": {}}, all_content)
        merged_text = "\n\n".join(
            str((item.get("variables") or {}).get("markdown_text") or "")
            for item in all_content
        )
        sections = [{
            "chapter_title": payload.get("title") or source.stem,
            "chapter_meta": {},
            "content_nodes": all_content,
            "markdown_text": merged_text,
        }]

    print(f"[DocToSysML] 识别到 {len(sections)} 个章节/节")

    # 创建 SysML 管理器
    mgr = SysMLManager(workspace_root=PROJECT_ROOT)

    # 根 Package：以文档标题命名
    doc_title = payload.get("title") or source.stem
    root_pkg_name = _safe_sysml_name(doc_title)
    root_pkg = Package(root_pkg_name, short_name=source.stem[:32])
    mgr.add_element(root_pkg)

    total_entities = 0
    total_relations = 0

    # 全局实体注册表（用于跨节关系）
    global_parts: Dict[str, PartDef] = {}
    global_attrs: Dict[str, AttributeDef] = {}
    global_reqs: Dict[str, RequirementDef] = {}

    for idx, section in enumerate(sections):
        title = section["chapter_title"]
        markdown_text = section["markdown_text"]
        if not markdown_text.strip():
            continue

        # 创建节的 Package
        section_pkg_name = _safe_section_package_name(title, idx)
        section_pkg = Package(section_pkg_name, short_name=f"s{idx + 1}")
        root_pkg.add_member(section_pkg)

        # 提取实体
        parts, attrs, reqs = _extract_entities_from_text(markdown_text)

        section_parts: Dict[str, PartDef] = {}
        section_attrs: Dict[str, AttributeDef] = {}
        section_reqs: Dict[str, RequirementDef] = {}

        # 创建 Part 定义
        for label, name in parts:
            if name in global_parts:
                # 引用已有定义
                section_parts[name] = global_parts[name]
                continue
            part = PartDef(name)
            section_pkg.add_member(part)
            global_parts[name] = part
            section_parts[name] = part
            total_entities += 1

        # 创建 Attribute 定义
        for label, name in attrs:
            if name in global_attrs:
                section_attrs[name] = global_attrs[name]
                continue
            attr = AttributeDef(name)
            section_pkg.add_member(attr)
            global_attrs[name] = attr
            section_attrs[name] = attr
            total_entities += 1

        # 创建 Requirement 定义
        for label, name in reqs:
            if name in global_reqs:
                section_reqs[name] = global_reqs[name]
                continue
            req = RequirementDef(name)
            section_pkg.add_member(req)
            global_reqs[name] = req
            section_reqs[name] = req
            total_entities += 1

        # 建立关联（同一节内的部件/属性间）
        if len(section_parts) >= 2:
            part_names = list(section_parts.keys())
            for i in range(min(len(part_names) - 1, 5)):  # 最多 5 条连接
                conn = ConnectionUsage(name=f"conn_{section_pkg_name}_{i + 1}")
                conn.ends = [
                    ConnectionEnd(part_names[i]),
                    ConnectionEnd(part_names[i + 1]),
                ]
                section_pkg.add_member(conn)
                total_relations += 1

        # 属性归属到部件
        for part_name in list(section_parts.keys())[:3]:  # 前 3 个部件
            for attr_name in list(section_attrs.keys())[:2]:  # 前 2 个属性
                attr_usage = AttributeUsage(
                    name=f"{attr_name}_of_{part_name}",
                    type_refs=[attr_name],
                )
                section_parts[part_name].add_member(attr_usage)
                total_entities += 1

    print(f"[DocToSysML] 共提取 {total_entities} 个实体, {total_relations} 个关系")
    return mgr, payload


def export_document_to_sysml(
    file_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    将文档导出为 SysML v2 .sysml 文件。

    Args:
        file_path: 源文档路径
        output_path: 输出 .sysml 文件路径，默认为文档同名 .sysml

    Returns:
        输出文件的路径
    """
    mgr, payload = build_sysml_model_from_doc_tree(file_path)

    if output_path is None:
        source = Path(file_path)
        output_path = str(source.with_suffix(".sysml"))

    mgr.save_to_file(output_path)
    print(f"[DocToSysML] 模型已保存到: {output_path}")
    return output_path


# ── CLI 入口 ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="从文档提取 SysML v2 模型并导出 .sysml 文件",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="输入文档路径（支持 PDF/DOCX/TXT/MD/XLSX 等）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 .sysml 文件路径（默认与输入同名，多文件时需配合 --outdir）",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="输出目录（多文件时必须指定）",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="同时打印 SysML 文本到 stdout",
    )

    args = parser.parse_args()
    inputs = args.input

    if len(inputs) > 1 and args.output:
        print("警告: 多文件模式下 -o 参数将被忽略，请使用 --outdir")
        args.output = None

    for input_path in inputs:
        source = Path(input_path)
        if not source.exists():
            print(f"跳过不存在的文件: {input_path}")
            continue

        output_path = args.output
        if output_path is None and args.outdir:
            out_dir = Path(args.outdir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(out_dir / f"{source.stem}.sysml")

        try:
            output_path = export_document_to_sysml(str(source), output_path)
            if args.print and output_path:
                text = Path(output_path).read_text(encoding="utf-8")
                print(f"\n{'=' * 60}")
                print(f"SysML 模型: {output_path}")
                print(f"{'=' * 60}")
                print(text)
        except Exception as exc:
            print(f"处理失败 [{input_path}]: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()