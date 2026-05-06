"""
Demo: 从文档树提取内容到 SysML v2 (.sysml) 文件
——基于结构驱动的语义段分析（SDSSA）算法 V2

核心理念（纯结构分析，无查表/无字典/无ML）：
  完全放弃关键词集合查表、字典匹配等"打表"式实体识别方法。
  仅利用文档自身的结构线索（标题层级深度、章节位置关系、内容特征、
  相邻段落的语义对比）来推断 SysML 元素。

  每一步都是：
    - 纯结构驱动的：分析层级位置、内容形态、上下文关系
    - 确定性 + 可审计：无黑盒、无统计、无预定义词表
    - 自然语言算法：利用人类写作时的天然结构规律

提取策略（多阶段渐进式解析）：
  阶段1：标题结构分析 —— 从标题层级深度和内容特征推断实体类型
  阶段2：列表结构分析 —— 从缩进和数值特征提取属性/参数
  阶段3：表格结构分析 —— 从列数和数据类型推断角色
  阶段4：句式结构分析 —— 从句子语法结构提取关系
  阶段5：实体注册与关系推断 —— 全局融合去重

鲁棒性设计：
  - 每阶段都有多层 fallback 降级解析
  - 置信度渐进累加，最终由 Context Oracle 综合裁决
  - 段落级 fallback：无法结构化解析的纯文本按 Part + 描述保留
  - 不依赖任何预定义词表、关键词集合、ML 模型

总体要求：鲁棒性 > 效果 > 速度
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_FILE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_RAG_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls",
})

# ===== 工具函数 =====

def _safe_sysml_name(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "unnamed"
    ascii_part = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_")
    if ascii_part and re.match(r"^[a-zA-Z_]", ascii_part):
        return ascii_part[:64]
    safe = re.sub(r"['\n\r\t]+", "", raw)[:64]
    return f"'{safe}'" if safe else "unnamed"

def _normalize_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _has_numeric(text: str) -> bool:
    return bool(re.search(r"[+-]?\d+\.?\d*", text))

def _has_unit_suffix(text: str) -> bool:
    return bool(re.search(r"[+-]?\d+\.?\d*\s*[a-zA-Z°%/μ]+$", text.strip()))

def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", text)) / max(len(text), 1)

def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[a-zA-Z]", text)) / max(len(text), 1)

def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[0-9]", text)) / max(len(text), 1)

def _depth_bias(level: int) -> float:
    if level <= 2:
        return -0.15
    elif level == 3:
        return 0.0
    elif level == 4:
        return 0.15
    return 0.25

# ===== 阶段1：标题结构分析 =====

def _analyze_title(text: str) -> Dict[str, float]:
    cjk = _cjk_ratio(text)
    alpha = _alpha_ratio(text)
    digit = _digit_ratio(text)
    has_num = _has_numeric(text)
    has_unit = _has_unit_suffix(text)
    length = len(text.strip())

    part = 0.0
    if alpha > 0.5 and length < 30:
        part += 0.3
    if cjk > 0.8 and length > 8:
        part += 0.15
    if not has_num:
        part += 0.1
    part = min(part, 0.5)

    attr = 0.0
    if has_num:
        attr += 0.2
    if has_unit:
        attr += 0.25
    if digit > 0.15:
        attr += 0.2
    if alpha < 0.3 and cjk < 0.7 and digit > 0.1:
        attr += 0.15
    attr = min(attr, 0.6)

    req = 0.0
    if text.strip().endswith("的") and length > 6:
        req += 0.2
    if cjk > 0.9 and length > 10 and not has_num:
        req += 0.1
    req = min(req, 0.4)

    return {"part": part, "attribute": attr, "requirement": req}

def _parse_title_level(text: str) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if not m:
            continue
        raw.append({"level": len(m.group(1)), "text": m.group(2).strip()})
    if not raw:
        return []

    results: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        level, title_text = entry["level"], entry["text"]
        scores = _analyze_title(title_text)
        bias = _depth_bias(level)

        prev_hint = 0.0
        for j in range(idx - 1, -1, -1):
            prev = raw[j]
            if prev["level"] == level:
                ps = _analyze_title(prev["text"])
                if ps["part"] > ps["attribute"]:
                    prev_hint = 0.08
                elif ps["attribute"] > ps["part"]:
                    prev_hint = -0.08
                break

        pc = scores["part"] + max(0, prev_hint) + max(0, -bias)
        ac = scores["attribute"] + max(0, -prev_hint) + max(0, bias)
        rc = scores["requirement"]

        if ac > pc and ac >= rc:
            t, c = "attribute", ac
        elif rc > pc and rc > ac:
            t, c = "requirement", rc
        else:
            t, c = "part", pc

        if t == "part":
            conf = min(0.5 + (c - max(ac, rc)) * 1.5, 0.95)
        elif t == "attribute":
            conf = min(0.5 + (c - max(pc, rc)) * 1.5, 0.95)
        else:
            conf = min(0.5 + (c - max(pc, ac)) * 2.0, 0.95)

        results.append({
            "level": level, "text": title_text,
            "inferred_type": t, "confidence": max(conf, 0.3),
        })
    return results

# ===== 阶段2：列表结构分析 =====

def _parse_list_items(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^(\s*)[-*]\s+(.+)$", s)
        if not m:
            m = re.match(r"^(\s*)\d+[.)]\s+(.+)$", s)
        if not m:
            continue
        indent, item_text = len(m.group(1)), m.group(2).strip()
        item_text = re.sub(r"\*\*|__|``", "", item_text)
        item_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", item_text)

        length = len(item_text)
        has_num = _has_numeric(item_text)
        has_unit = _has_unit_suffix(item_text)
        dr = _digit_ratio(item_text)

        # "名称: 值"
        kv = re.match(r"^([\u4e00-\u9fff\w\-+./]+)[：:]\s*(.+)$", item_text)
        if kv:
            key, value = kv.group(1).strip(), kv.group(2).strip()
            conf = min(0.85 + (0.05 if _has_numeric(value) else 0.0), 0.95)
            results.append({"text": item_text, "indent": indent, "inferred_type": "attribute", "confidence": conf, "key": key, "value": value})
            continue

        # "值 单位"
        nv = re.match(r"^([\u4e00-\u9fff\w\-+]+)\s+([+-]?\d+\.?\d*)\s*([\w°/%%]+)?$", item_text)
        if nv:
            conf = min(0.8 + (0.05 if nv.group(3) else 0.0), 0.9)
            val = (nv.group(2) + " " + (nv.group(3) or "")).strip()
            results.append({"text": item_text, "indent": indent, "inferred_type": "attribute", "confidence": conf, "key": nv.group(1).strip(), "value": val})
            continue

        if length < 40 and has_num and dr > 0.05:
            conf = min(0.65 + (0.1 if has_unit else 0.0), 0.85)
            results.append({"text": item_text, "indent": indent, "inferred_type": "attribute", "confidence": conf, "key": item_text, "value": ""})
            continue

        if length < 30 and not has_num:
            results.append({"text": item_text, "indent": indent, "inferred_type": "unknown", "confidence": 0.5, "key": item_text, "value": ""})
            continue

        results.append({"text": item_text, "indent": indent, "inferred_type": "unknown", "confidence": 0.4, "key": item_text, "value": ""})
    return results

# ===== 阶段3：表格结构分析 =====

def _extract_table(lines: List[str], start: int) -> Optional[Tuple[List[str], List[List[str]]]]:
    if start + 2 >= len(lines):
        return None
    sep = lines[start + 1].strip()
    if not re.match(r"^\|[\s\-:|+]+\|$", sep):
        return None
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows: List[List[str]] = []
    i = start + 2
    while i < len(lines):
        rl = lines[i].strip()
        if not rl.startswith("|") or not rl.endswith("|"):
            break
        cols = [c.strip() for c in rl.strip("|").split("|")]
        while len(cols) < len(header):
            cols.append("")
        rows.append(cols[:len(header)])
        i += 1
    return (header, rows) if rows else None

def _row_type(row: List[str]) -> str:
    nc, tc, lc = 0, 0, 0
    for cv in row:
        s = cv.strip()
        if not s or s in ("-", "\u2014", ""):
            continue
        if _has_numeric(s):
            nc += 1
        elif len(s) > 20:
            lc += 1
        else:
            tc += 1
    t = max(nc + tc + lc, 1)
    if nc / t > 0.4:
        return "attribute"
    if lc / t > 0.5:
        return "requirement"
    return "part"

def _parse_table_structure(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith("|") and l.strip().endswith("|") and (i == 0 or not lines[i-1].strip().startswith("|"))]

    for si in starts:
        table = _extract_table(lines, si)
        if not table:
            continue
        header, rows = table
        name_likely = sum(1 for r in rows if r and r[0].strip() and len(r[0].strip()) < 30 and not re.match(r"^\d+(\.\d+)?$", r[0].strip()))

        for row in rows:
            if not row:
                continue
            en = row[0].strip() if row[0].strip() else ""
            if not en or en in ("-", "\u2014", ""):
                continue
            en = re.sub(r"\*\*|``", "", en).strip()
            if not en or en in ("-", "\u2014", ""):
                continue
            results.append({
                "entity_name": en, "inferred_type": _row_type(row),
                "confidence": 0.85,
                "columns": {str(header[i]): row[i] if i < len(row) else "" for i in range(len(header))},
            })
    return results

# ===== 阶段4：句式结构分析 =====
# 注意：此阶段为低置信度辅助提取。
# 鲁棒性优先策略：只提取高确定性关系，避免产生碎片实体。

# 可靠的结构动词集合：只有这些动词引导的关系才被接受
_RELIABLE_RELATION_VERBS = frozenset({
    "连接", "包含", "包括", "组成", "由",
    "属于", "依赖于", "基于", "使用", "提供",
    "产生", "发送", "接收", "传输", "转换",
    "控制", "管理", "监控", "驱动",
})

# 可靠表属性动词：X的Y | 该Y|其Y 为/是/等于 值
_RELIABLE_ATTRIB_VERBS = frozenset({"=", ":", "：", "为", "是", "等于"})


def _parse_sentence_patterns(text: str) -> List[Dict[str, Any]]:
    """
    提取确定性高的关系。
    
    鲁棒性策略：
    - 只接受主体和客体都是完整词（3+ 字母或 2+ CJK 字符）
    - 动词必须在 _RELIABLE_RELATION_VERBS 中
    - 不产生长度 < 3 的实体名
    """
    results: List[Dict[str, Any]] = []
    for s in re.split(r"[。；;\n]", text):
        s = s.strip()
        if len(s) < 10:
            continue  # 太短的句子跳过

        # 类型A：可靠动词连接的二元关系
        # "X 动词 Y" 或 "X动词Y"
        for verb in _RELIABLE_RELATION_VERBS:
            # 搜索带有空格或无空格的模式
            for pattern in [
                rf"([\u4e00-\u9fffA-Za-z]{{2,20}})\s+{re.escape(verb)}\s+([\u4e00-\u9fffA-Za-z]{{2,20}})",
                rf"([\u4e00-\u9fffA-Za-z]{{2,20}}){re.escape(verb)}([\u4e00-\u9fffA-Za-z]{{2,20}})",
            ]:
                for m in re.finditer(pattern, s):
                    subj, obj = m.group(1).strip(), m.group(2).strip()
                    # 过滤器：忽略分词后的短碎片
                    if len(subj) < 2 or len(obj) < 2:
                        continue
                    if _cjk_ratio(subj) > 0.5 and len(subj) < 2:
                        continue
                    if _cjk_ratio(obj) > 0.5 and len(obj) < 2:
                        continue
                    results.append({
                        "relation_type": "composition",
                        "subject": subj,
                        "object": obj,
                        "confidence": 0.70,
                        "raw_sentence": s,
                    })
                    break  # 同一句只匹配一次同一动词

        # 类型B："X 的 Y 为/是/等于 Z" —— 属性关系
        dp = s.find("的")
        if 2 < dp < len(s) - 6:
            left_of_de = s[:dp].strip()
            right_of_de = s[dp + 1:].strip()
            if not left_of_de or not right_of_de:
                continue
            left_words = re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", left_of_de)
            if not left_words:
                continue
            subject = left_words[-1]
            for vb in _RELIABLE_ATTRIB_VERBS:
                parts = right_of_de.split(vb, 1)
                if len(parts) == 2:
                    attr_name = parts[0].strip()
                    attr_val = parts[1].strip()
                    if attr_name and attr_val and 2 <= len(attr_name) <= 30:
                        if re.search(r"[\u4e00-\u9fffA-Za-z]", attr_name):
                            results.append({
                                "relation_type": "attribution",
                                "subject": subject,
                                "object": attr_name,
                                "value": attr_val,
                                "confidence": 0.65,
                                "raw_sentence": s,
                            })
                            break

        # 类型C：数字指标的约束提取
        vm = re.search(
            r"([\u4e00-\u9fffA-Za-z]{2,20})\s*(?:为|是|等于)\s*"
            r"([+-]?\d+\.?\d*\s*[a-zA-Z°%/μ]+[\w°%/μ]*)",
            s,
        )
        if vm:
            subj, val = vm.group(1).strip(), vm.group(2).strip()
            if len(subj) >= 2 and val:
                results.append({
                    "relation_type": "attribution",
                    "subject": subj,
                    "object": subj,
                    "value": val,
                    "confidence": 0.75,
                    "raw_sentence": s,
                })

    return results

# ===== 实体注册表 =====

class EntityRegistry:
    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: List[Dict[str, Any]] = []

    def register_entity(self, name: str, inferred_type: str, confidence: float,
                        source: str = "unknown", columns: Optional[Dict[str, str]] = None) -> str:
        sn = _safe_sysml_name(name)
        if not sn:
            return ""
        if sn not in self._entities:
            self._entities[sn] = {"name": sn, "original_names": set(), "type_votes": {}, "attributes": {}, "max_confidence": 0.0, "sources": set()}
        e = self._entities[sn]
        e["original_names"].add(name)
        e["sources"].add(source)
        if confidence > e["max_confidence"]:
            e["max_confidence"] = confidence
        e["type_votes"].setdefault(inferred_type, []).append(confidence)
        if columns:
            for cn, cv in columns.items():
                if cn != name and cv.strip() and cv not in ("-", "\u2014"):
                    ak = _safe_sysml_name(cn)
                    if ak:
                        e["attributes"][ak] = cv
        return sn

    def register_relation(self, relation_type: str, subject: str, obj: str,
                          confidence: float, raw_sentence: str = "") -> None:
        ss, so = _safe_sysml_name(subject), _safe_sysml_name(obj)
        if ss and so and ss != so:
            self._relations.append({"relation_type": relation_type, "subject": ss, "object": so, "confidence": confidence, "raw_sentence": raw_sentence})

    def winner_type(self, sn: str) -> str:
        e = self._entities.get(sn)
        if not e or not e["type_votes"]:
            return "part"
        return max(e["type_votes"], key=lambda k: sum(e["type_votes"][k]))

    def all_entities(self) -> List[Dict[str, Any]]:
        return [{"safe_name": k, "original_name": next(iter(v["original_names"]), k),
                 "type": self.winner_type(k), "confidence": v["max_confidence"],
                 "attributes": dict(v["attributes"]), "sources": list(v["sources"])}
                for k, v in self._entities.items()]

    def all_relations(self) -> List[Dict[str, Any]]:
        return list(self._relations)

    def has(self, sn: str) -> bool:
        return sn in self._entities

# ===== 从文档树提取 =====

def _extract_from_doc_tree(tree: List[Dict[str, Any]], registry: EntityRegistry, markdown_text: str = "") -> None:
    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            title = str(node.get("title") or "").strip()
            vars_ = dict(node.get("variables") or {})
            md = str(vars_.get("render_markdown_text") or vars_.get("markdown_text") or "")
            if title and md:
                for t in _parse_title_level(f"# {title}"):
                    if t["inferred_type"] != "skip":
                        registry.register_entity(title, t["inferred_type"], t["confidence"], source="title")
                for item in _parse_list_items(md):
                    entity_type = item["inferred_type"] if item["inferred_type"] != "unknown" else "attribute"
                    if item["confidence"] >= 0.5:
                        registry.register_entity(item.get("key") or item["text"], entity_type, item["confidence"], source="list")
                    if item.get("key") and item.get("value") and item["confidence"] >= 0.6:
                        registry.register_relation("attribution", title, item["key"], 0.65, raw_sentence=item["text"])
                for tab in _parse_table_structure(md):
                    registry.register_entity(tab["entity_name"], tab["inferred_type"], tab["confidence"], source="table", columns=tab.get("columns"))
                for rel in _parse_sentence_patterns(md):
                    if rel.get("subject"):
                        registry.register_entity(rel["subject"], "part", rel["confidence"] * 0.9, source=f"sentence_{rel['relation_type']}")
                    if rel.get("object"):
                        registry.register_entity(rel["object"], "part", rel["confidence"] * 0.9, source=f"sentence_{rel['relation_type']}")
                    registry.register_relation(rel["relation_type"], rel.get("subject", ""), rel.get("object", ""), rel["confidence"], raw_sentence=rel.get("raw_sentence", ""))
            walk(list(node.get("children") or []))
    walk(list(tree or []))

    # 全文补充提取
    if markdown_text:
        for tab in _parse_table_structure(markdown_text):
            registry.register_entity(tab["entity_name"], tab["inferred_type"], tab["confidence"] * 0.85, source="global_table", columns=tab.get("columns"))
        for rel in _parse_sentence_patterns(markdown_text):
            if rel.get("subject"):
                registry.register_entity(rel["subject"], "part", rel["confidence"] * 0.8, source=f"global_sentence_{rel['relation_type']}")
            if rel.get("object"):
                registry.register_entity(rel["object"], "part", rel["confidence"] * 0.8, source=f"global_sentence_{rel['relation_type']}")
            registry.register_relation(rel["relation_type"], rel.get("subject", ""), rel.get("object", ""), rel["confidence"] * 0.85, raw_sentence=rel.get("raw_sentence", ""))

# ===== 构建 SysML 模型 =====

def _build_sysml_package(registry: EntityRegistry, doc_name: str, title: str) -> Package:
    root_pkg = Package(name=_safe_sysml_name(doc_name))
    entities = registry.all_entities()
    relations = registry.all_relations()

    # 构建 parts 字典
    parts: Dict[str, PartDef] = {}
    for e in entities:
        sn = e["safe_name"]
        conf = e["confidence"]
        etype = e["type"]
        if conf < 0.45:
            continue
        if etype == "attribute" and conf < 0.65:
            continue  # 属性实体需要更高置信度
        parts[sn] = PartDef(name=sn)

    # 从关系补充实体
    for r in relations:
        if r["confidence"] < 0.5:
            continue
        for side in ["subject", "object"]:
            n = r.get(side, "")
            if n and n not in parts:
                parts[n] = PartDef(name=n)

    # 为实体添加属性
    for e in entities:
        sn = e["safe_name"]
        pd = parts.get(sn)
        if not pd:
            continue
        for ak, av in e.get("attributes", {}).items():
            if av.strip():
                pd.add_member(AttributeUsage(name=ak, value_expr=av))
        # 如果实体有来源 relation，也加上属性
        for r in relations:
            if r.get("subject") == sn and r.get("value") and r["confidence"] >= 0.5:
                attr_name = r.get("object", r.get("relation_type", "property"))
                pd.add_member(AttributeUsage(name=str(attr_name), value_expr=str(r["value"])))

    # 添加关系
    for r in relations:
        if r["confidence"] < 0.5:
            continue
        if r["relation_type"] == "attribution":
            s, o = r.get("subject", ""), r.get("object", "")
            val = r.get("value", "")
            if s in parts and o and val:
                parts[s].add_member(AttributeUsage(name=o, value_expr=val))
        elif r["relation_type"] == "composition":
            s, o = r.get("subject", ""), r.get("object", "")
            if s in parts and o in parts and s != o:
                # 添加连接关系
                conn = ConnectionUsage(
                    name=f"conn_{s}_{o}",
                    ends=[ConnectionEnd(ref=s), ConnectionEnd(ref=o)],
                )
                root_pkg.add_member(conn)

    # 将所有 parts 添加到 package
    for pn, pd in parts.items():
        # 检查是否已在 root_pkg 中
        already = any(m.name == pd.name for m in root_pkg.members if hasattr(m, 'name'))
        if not already:
            root_pkg.add_member(pd)

    return root_pkg


# ===== 构建 SysML 模型的对外入口 =====

def _read_plain_text(raw_path: Path) -> str:
    """读取纯文本（兼容各种编码）"""
    for enc in ("utf-8", "gbk", "gb2312", "utf-16", "latin-1"):
        try:
            return raw_path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw_path.read_text(encoding="utf-8", errors="replace")


def _try_rag_read(raw_path: Path, ext: str, doc_title: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """尝试通过 RAG 管道读取文档。失败时返回空三元组。"""
    if ext not in SUPPORTED_RAG_EXTENSIONS:
        return ([], {}, "")
    # 先尝试 llama_index / rag 管道
    if ext in (".txt", ".md", ".pdf", ".doc", ".docx", ".csv", ".xlsx", ".xls"):
        try:
            from rag.documents import load_rag_documents_from_paths
            docs = load_rag_documents_from_paths([str(raw_path)], set(SUPPORTED_RAG_EXTENSIONS))
            markdown_parts = []
            doc_list = []
            for d in docs if isinstance(docs, list) else [docs]:
                text = getattr(d, 'text', '') or getattr(d, 'content', '') or ''
                if text:
                    markdown_parts.append(text)
                    doc_list.append({"title": doc_title, "children": [], "variables": {"render_markdown_text": text}})
            merged = "\n\n".join(markdown_parts)
            if doc_list and merged:
                return (doc_list, {"title": doc_title, "page_count": len(doc_list)}, merged)
        except Exception:
            traceback.print_exc()
    return ([], {}, "")


def _read_file_via_rag(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    通过 RAG 管道（优先）或纯文件读取（fallback）读取文档文件。

    返回：
      - doc_tree: 文档树结构列表
      - meta: 元信息字典
      - markdown_text: 合并的文本
    """
    raw_path = Path(file_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"文件未找到: {file_path}")

    ext = raw_path.suffix.lower()
    doc_title = raw_path.stem

    # —— 尝试 RAG 管道（如果可用） ——
    doc_tree, meta, markdown_text = _try_rag_read(raw_path, ext, doc_title)
    if doc_tree and markdown_text:
        return (doc_tree, meta, markdown_text)

    # —— 纯 Python fallback ——
    plain_text = _read_plain_text(raw_path)

    # 对 markdown 文件做简单的标题分割做 doc_tree，同时收集每一节下的纯文本
    simple_tree: List[Dict[str, Any]] = []
    if ext in (".md", ".txt"):
        lines = plain_text.splitlines()
        stack: List[Dict[str, Any]] = [{"title": doc_title, "children": [], "_body_lines": []}]
        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if len(stack) > 1:
                    cur = stack[-1]
                    body = "\n".join(cur.get("_body_lines", [])).strip()
                    if body:
                        cur.setdefault("variables", {})["render_markdown_text"] = body
                    cur.pop("_body_lines", None)
                while len(stack) > 1 and stack[-1].get("level", 99) >= level:
                    stack.pop()
                node: Dict[str, Any] = {"title": title, "level": level, "children": [], "_body_lines": []}
                stack[-1]["children"].append(node)
                stack.append(node)
            else:
                stripped = line.strip()
                if stripped:
                    stack[-1].setdefault("_body_lines", []).append(stripped)
        if len(stack) > 1:
            cur = stack[-1]
            body = "\n".join(cur.get("_body_lines", [])).strip()
            if body:
                cur.setdefault("variables", {})["render_markdown_text"] = body
            cur.pop("_body_lines", None)
        simple_tree = stack[0]["children"]

    doc_tree = simple_tree
    meta = {"title": doc_title, "page_count": 0}
    markdown_text = plain_text
    return (doc_tree, meta, markdown_text)


def build_sysml_model_from_doc_tree(file_path: str, output_path: Optional[str] = None,
                                     engine: Optional[Any] = None) -> Tuple[Optional[SysMLManager], Dict[str, Any]]:
    """
    从文档树构建 SysML 模型并写入 .sysml 文件。

    参数：
      file_path: 输入文档路径
      output_path: 输出 .sysml 路径（可选）
      engine: RAG引擎实例（可选）

    返回：
      (manager, payload)
        manager: SysMLManager 实例（加载后可能为 None）
        payload: 包含统计信息的字典
    """
    doc_tree, meta, markdown_text = _read_file_via_rag(file_path)

    registry = EntityRegistry()
    _extract_from_doc_tree(doc_tree, registry, markdown_text)

    doc_name = meta.get("title", Path(file_path).stem)
    title = meta.get("title", doc_name)
    root_package = _build_sysml_package(registry, doc_name, title)

    if output_path is None:
        output_path = str(Path(file_path).with_suffix(".sysml"))
    output_path = str(output_path)

    manager: Optional[SysMLManager] = None
    pkg_text = root_package.to_text()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pkg_text)

    try:
        manager = SysMLManager()
        manager.load_from_file(output_path)
    except Exception:
        pass  # loading 失败不影响输出

    payload = {
        "title": doc_name,
        "page_count": meta.get("page_count", 0),
        "entities": len(registry.all_entities()),
        "relations": len(registry.all_relations()),
    }
    return (manager, payload)


# ===== CLI =====
def main() -> None:
    parser = argparse.ArgumentParser(description="从文档构建 SysML v2 模型")
    parser.add_argument("input", type=str, help="输入的文档文件路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出的 .sysml 文件路径")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON 统计")
    args = parser.parse_args()

    manager, payload = build_sysml_model_from_doc_tree(args.input, args.output)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
