#!/usr/bin/env python3
"""
Headless QA script — uses ONLY SysML MCP tools (sysml_retrieve) for retrieval.
No RAG vector/regex search allowed.
Usage:
  python scripts/run_qa.py --db AIOPS_New --question "..."
  python scripts/run_qa.py  # uses defaults
"""

from __future__ import annotations

import sys
import asyncio
import json
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

DEFAULT_DB_NAME = "AIOPS_New"
DEFAULT_QUESTION = (
    "天河高性能计算机系统的mn0节点上只有命令行可用，"
    "需要怎么做才能获取机柜R1P3上所有节点的加电信息？"
)
DEFAULT_REFERENCE = """根据文档中的信息，您可以通过以下步骤获取R1P3机柜所有节点的加电信息：
1.通过SMU转发命令到CMU
  在mn0节点上执行以下命令（需针对R1P3的4个机柜分别查询）：
  ```
  smu_tranfer_cmd r1.p03a.m yhst
  smu_tranfer_cmd r1.p03b.m yhst
  smu_tranfer_cmd r1.p03c.m yhst
  smu_tranfer_cmd r1.p03d.m yhst
  ```
  - r1.p03a.m：R1列、P3机柜、a机柜的CMU（a/b/c/d代表4个机柜）
  - yhst：CMU上用于查看所有节点加电状态的命令（文档节明确说明）
2.关键说明
  - 文档6.3节指出：yhst命令可"查看所有结点的加电信息"
  - R1P3机柜对应文档中的"R1P0-R1P3号柜"（1.4节系统布局），其机框编号为a/b/c/d
  - 无需直接登录CMU（因CMU IP不固定），通过SMU的smu_tranfer_cmd可安全转发命令
    注：若需更详细信息，可结合ncid命令转换节点号（6.2节），但基础加电信息通过yhst即可获取。"""

OUTPUT_DIR = ROOT_DIR / "agentIdeas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_qa")

QA_SYSTEM_PROMPT = """You are a QA (Quality Assurance) agent for the Huchao supercomputer.
Answer questions using SysML knowledge graph data retrieved via sysml_retrieve.

## Workflow
1. The system will provide SysML entity/relation data retrieved from the knowledge graph.
2. Analyse the entities, their properties, members, and relationship graph.
3. Give a clear answer based on what the KG contains.
4. If the KG data is insufficient, state clearly what's missing.

## Important
- Use ONLY the provided SysML data. Do not fabricate information.
- Pay attention to entity types: Definitions (PartDef, AttributeDef) describe types; Usages (PartUsage, AttributeUsage) describe instances.
- Live values (instant variables) have source/selector fields indicating CLI paths.
- Operations have source fields indicating executable scripts/commands.

## Output format
- First give the answer in Chinese.
- Then add "## QA 反馈" listing issues found in the KG data.
用中文回答。"""


def extract_entity_names(message: str) -> list[str]:
    """Extract candidate entity names from user message."""
    candidates: list[str] = []
    msg = message.strip()

    quoted = re.findall(r'["\'""\u300c\u300d]([^"\'""\u300c\u300d]+)["\'""\u300c\u300d]', msg)
    candidates.extend(quoted)

    patterns = [
        r'([\u4e00-\u9fff]{2,8}(?:模块|系统|节点|引擎|服务|仪表盘|数据库|传感器|计算机|设备|组件|接口|控制器|处理器|存储器|网络|总线|通道|机柜))',
        r'([A-Za-z_][A-Za-z0-9_]*)',
    ]
    for pat in patterns:
        found = re.findall(pat, msg)
        candidates.extend(found)

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
    return result[:8]


def build_context_from_retrieve(result: dict) -> str:
    """Build readable context from sysml_retrieve result."""
    if not result.get("ok"):
        return ""
    entity = result.get("entity", {})
    graph = result.get("relationship_graph", {})

    parts = [f"Entity: {entity.get('name','?')} ({entity.get('type', entity.get('class','?'))})"]

    meta = entity.get("metadata", {})
    if meta.get("description"):
        parts.append(f"  Description: {meta['description']}")
    if meta.get("properties"):
        parts.append(f"  Properties: {json.dumps(meta['properties'], ensure_ascii=False)}")
    if entity.get("aliases"):
        parts.append(f"  Aliases: {', '.join(entity['aliases'])}")

    for extra in ["supertypes", "direction", "multiplicity", "type_refs", "value", "derived", "constant", "reference"]:
        v = entity.get(extra)
        if v is not None:
            parts.append(f"  {extra}: {v}")

    members = entity.get("members", [])
    if members:
        lines = [f"  Members ({len(members)}):"]
        for m in members[:30]:
            line = f"    - {m.get('type','?')} {m.get('name','')}"
            if m.get("value"):
                line += f" = {m['value']}"
            lines.append(line)
        parts.append("\n".join(lines))

    total_nodes = graph.get("total_nodes", 0)
    total_edges = graph.get("total_edges", 0)
    if total_nodes > 1 or total_edges > 0:
        parts.append(f"\n  Graph: depth={graph.get('depth',0)}, nodes={total_nodes}, edges={total_edges}")
        nodes = graph.get("nodes", {})
        node_summaries = []
        for n, v in nodes.items():
            nt = v.get("type", v.get("class", "?"))
            node_summaries.append(f"{n}({nt})")
        parts.append(f"  Nodes: {', '.join(node_summaries)}")
        for e in graph.get("edges", []):
            rn = e.get("relation", {}).get("name", "?")
            parts.append(f"    {e['from']} --[{rn}]--> {e['to']}")

    return "\n".join(parts)


async def run_qa(
    db_name: str,
    question: str,
    reference: str | None = None,
):
    """Run the QA pipeline: KG load → sysml_retrieve → LLM synthesis."""

    # ── 1. Load SysML KG via MCP tools
    from scripts.sysml_rag_mcp_server import (
        sysml_load_model,
        sysml_retrieve,
        sysml_model_summary,
        _get_manager,
        _global_manager,
        _loaded_files,
    )

    _global_manager = None
    _loaded_files.clear()

    kg_path = ROOT_DIR / "database" / db_name / "knowledge_graph.sysml"
    if not kg_path.exists():
        logger.error("KG file not found: %s", kg_path)
        print(f"ERROR: knowledge_graph.sysml not found in database/{db_name}/")
        return

    logger.info("Loading KG: %s", kg_path)
    load_result = sysml_load_model(str(kg_path))
    if not load_result.get("ok"):
        logger.error("Failed to load KG: %s", load_result)
        return

    summary = sysml_model_summary()
    logger.info("KG loaded: %d entities, %d relations", summary["total_entities"], summary["total_relations"])

    # ── 2. Extract entity names and run sysml_retrieve
    candidates = extract_entity_names(question)
    logger.info("Candidate entity names: %s", candidates)

    context_parts = []
    retrieved = set()
    not_found = []

    for name in candidates:
        for k_val in [1, 2, 0]:
            r = sysml_retrieve(name, k=k_val)
            if r.get("ok"):
                ename = r.get("entity", {}).get("name", "")
                if ename in retrieved:
                    continue
                retrieved.add(ename)
                ctx = build_context_from_retrieve(r)
                if ctx:
                    context_parts.append(f"--- {name} (k={k_val}) ---\n{ctx}")
                break
            else:
                if k_val == 0:  # only record once per candidate
                    not_found.append(name)
                break

    # Also try Live value lookups for power-related HVs
    hv_candidates = ["power_state", "power_status", "power_supply", "power_module"]
    for hv_id in hv_candidates:
        if hv_id not in retrieved:
            r = sysml_retrieve(hv_id, k=1)
            if r.get("ok"):
                retrieved.add(r.get("entity", {}).get("name", ""))
                ctx = build_context_from_retrieve(r)
                if ctx:
                    context_parts.append(f"--- {hv_id} (k=1, HV) ---\n{ctx}")

    context = "\n\n".join(context_parts) if context_parts else "No SysML entities matched."

    logger.info("Retrieved: %d entities, Not found: %d", len(retrieved), len(not_found))
    logger.info("Context length: %d chars", len(context))

    # ── 3. Build messages
    from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import SecretStr

    ref_block = f"\n\n## 参考回答:\n{reference}" if reference else ""

    user_prompt = f"""## SysML 知识图谱检索结果 (仅 sysml_retrieve)

KG summary: {summary['total_entities']} entities, {summary['total_relations']} relations
Candidate names extracted from question: {json.dumps(candidates, ensure_ascii=False)}
Names not found in KG: {json.dumps(not_found, ensure_ascii=False)}

Retrieved entities/relations:
{context[:12000]}
{ref_block}

## 用户问题
{question}

请根据 SysML KG 数据回答。如果 KG 数据不足，请说明缺失了什么。在答案末尾添加 "## QA 反馈"。"""

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # ── 4. Call LLM
    llm = ChatOpenAIWithReasoning(
        model=config.settings.LLM_MODEL or "qwen3:8b",
        temperature=0.1,
        api_key=SecretStr(config.settings.OPENAI_API_KEY or "ollama"),
        base_url=config.settings.OPENAI_API_URL or "http://localhost:11434/v1",
        timeout=180,
        streaming=False,
    )
    try:
        logger.info("Calling LLM (%s)...", config.settings.LLM_MODEL)
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=240)
        answer = str(response.content)
    except asyncio.TimeoutError:
        answer = "LLM 请求超时（240s）。请使用更小的模型或精简上下文。"
    except Exception as e:
        answer = f"LLM 调用失败: {type(e).__name__}: {e}"

    # ── 5. Write output files
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    qa_file = OUTPUT_DIR / "qa_output.txt"
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    qa_file.write_text(answer, encoding="utf-8")

    fb_file = OUTPUT_DIR / "feedback.txt"
    fb_file.parent.mkdir(parents=True, exist_ok=True)
    with fb_file.open("a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] QA session (sysml-only): db={db_name}, "
                f"entities_retrieved={len(retrieved)}, not_found={len(not_found)}, "
                f"model={config.settings.LLM_MODEL}\n"
                f"  question={question[:80]}\n"
                f"  answer_len={len(answer)}\n")

    bug_file = OUTPUT_DIR / "bug.txt"
    bug_file.parent.mkdir(parents=True, exist_ok=True)
    bugs = []
    if summary["total_entities"] == 0:
        bugs.append("[CRITICAL] SysML KG 为空 — 无实体可检索")
    if not retrieved:
        bugs.append("[HIGH] sysml_retrieve 未命中任何实体")
    if not_found:
        bugs.append(f"[MEDIUM] 候选实体未找到: {not_found}")
    if retrieved and len(context_parts) < 3:
        bugs.append("[MEDIUM] 检索到的实体数量不足（需要更多 KG 数据）")

    with bug_file.open("a", encoding="utf-8") as f:
        for b in bugs:
            f.write(f"[{timestamp}] {b}\n")

    # ── 6. Summary
    print("=" * 60)
    print("QA COMPLETE (sysml_retrieve only)")
    print(f"  DB/KG:           {db_name}")
    print(f"  KG entities:     {summary['total_entities']}")
    print(f"  KG relations:    {summary['total_relations']}")
    print(f"  Retrieved:       {len(retrieved)} entities")
    print(f"  Not found:       {len(not_found)} names")
    print(f"  Context:         {len(context)} chars")
    print(f"  Answer:          {len(answer)} chars")
    print(f"  Output:          {qa_file}")
    print(f"  Feedback:        {fb_file}")
    for b in bugs:
        print(f"  Bug: {b}")
    print("=" * 60)
    print()
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless QA script (SysML KG only)")
    parser.add_argument("--db", default=DEFAULT_DB_NAME, help="RAG DB name (default: %(default)s)")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Reference answer (optional)")
    parser.add_argument("--model", default=None, help="LLM model override (e.g. qwen3:8b)")
    args = parser.parse_args()

    if args.model:
        config.settings = config.settings.update(LLM_MODEL=args.model)

    asyncio.run(run_qa(
        db_name=args.db,
        question=args.question,
        reference=args.reference,
    ))
