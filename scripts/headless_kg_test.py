#!/usr/bin/env python3
"""
Headless KG Build + QA Test
============================
Builds SysML knowledge graphs from documents, then answers a question.
Captures all LLM output, reasoning, and tool calls in sequence.

Usage:
  python scripts/headless_kg_test.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from io import StringIO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

# ── Configuration ────────────────────────────────────────────

DB_NAME = "AIOPS_New"
DOC_PATHS = [
    "/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc",
    "/home/ritanlisa/文档/初步验收与试运行分册-6-硬件维护手册 - 1227.doc",
    "/home/ritanlisa/文档/浪潮虚拟化InCloud Sphere 6.5.1运维手册.pdf",
]
QUESTION = (
    "天河高性能计算机系统的mn0节点上只有命令行可用，"
    "需要怎么做才能获取机柜R1P3上所有节点的加电信息？"
)
REFERENCE = """根据文档中的信息，您可以通过以下步骤获取R1P3机柜所有节点的加电信息：
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
  - 无需直接登录CMU（因CMU IP不固定），通过SMU的smu_tranfer_cmd可安全转发命令"""

OUTPUT_DIR = ROOT_DIR / "tmp" / "headless_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"kg_build_qa_{RUN_TIMESTAMP}.log"

# ── Logging Setup ────────────────────────────────────────────

class TeeLogger:
    """Tees output to both file and stderr for real-time monitoring."""
    def __init__(self, filepath: Path):
        self.file = open(str(filepath), "w", encoding="utf-8", buffering=1)
    def write(self, msg: str):
        self.file.write(msg)
        sys.__stderr__.write(msg)
    def flush(self):
        self.file.flush()
        sys.__stderr__.flush()
    def close(self):
        self.file.close()

tee = TeeLogger(LOG_FILE)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=tee,
)

# Suppress noisy libraries
for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
            "llama_index", "urllib3", "asyncio", "faiss", "PIL",
            "mcp.client.stdio"]:        # MCP notifications/initialized parse error (non-fatal)
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("headless_test")


# ── Config overrides ─────────────────────────────────────────

KG_MODEL = "gemma4:31b"              # 强模型 — KG 提取和富化
QA_MODEL = "qwen3:8b"              # 小模型 — QA 合成是简单文本任务

config.settings = config.settings.update(
    RAG_DB_NAME=DB_NAME,
    RAG_DB_NAMES=[DB_NAME],
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL=KG_MODEL,
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=300,
    KG_EXTRACTION_MAX_ITERATIONS=200,
    LLM_MODEL=QA_MODEL,
    AGENT_VERBOSE=True,
    AGENT_INVOKE_TIMEOUT=1200,
    OCR_MODEL=None,
    LLM_REQUEST_TIMEOUT=600,
)

# ── GPU contention check ────────────────────────────────────

def wait_for_gpu_free(poll_interval: float = 10.0, max_wait: float = 120.0):
    """Wait until no other compute-intensive process is using the GPU.
    Ollama is the model server and is expected to always run."""
    import subprocess
    waited = 0.0
    while waited < max_wait:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            our_pid = str(os.getpid())
            # Filter: ignore ollama (model server) and our own process
            heavy_procs = []
            for l in lines:
                if our_pid in l:
                    continue
                if "ollama" in l.lower():
                    continue
                heavy_procs.append(l)
            if not heavy_procs:
                logger.info("GPU available — proceeding")
                return True
            logger.info("GPU busy (%d other heavy process(es)), waiting %.1fs...",
                        len(heavy_procs), poll_interval)
            time.sleep(poll_interval)
            waited += poll_interval
        except Exception as e:
            logger.warning("nvidia-smi check failed: %s", e)
            return True
    logger.warning("GPU still busy after %.0fs, proceeding anyway", max_wait)
    return True

# GPU check disabled — user explicitly requested to run regardless
# wait_for_gpu_free()
logger.info("GPU check skipped — running regardless of GPU contention")

logger.info("=" * 70)
logger.info("HEADLESS KG BUILD + QA TEST")
logger.info(f"Timestamp: {RUN_TIMESTAMP}")
logger.info(f"DB: {DB_NAME}")
logger.info(f"Model: {config.settings.KG_EXTRACTION_MODEL}")
logger.info(f"Agent verbose: {config.settings.AGENT_VERBOSE}")
logger.info(f"Log file: {LOG_FILE}")
logger.info("=" * 70)


# ── Step 1: Load Documents ──────────────────────────────────

async def step1_load_documents():
    """Load all documents via the document tree system."""
    logger.info("─" * 60)
    logger.info("[STEP 1] Loading documents...")
    logger.info("─" * 60)

    from rag.documents import load_rag_documents_from_paths
    from rag.engine import SUPPORTED_RAG_EXTENSIONS

    t0 = time.time()

    rag_docs = load_rag_documents_from_paths(
        DOC_PATHS,
        SUPPORTED_RAG_EXTENSIONS,
    )

    elapsed = time.time() - t0
    logger.info("Loaded %d documents in %.1fs", len(rag_docs), elapsed)

    for doc in rag_docs:
        name = getattr(doc, "doc_name", "?")
        pages = getattr(doc, "page_count", 0)
        mono = doc.get_mono_pages() if hasattr(doc, "get_mono_pages") else []
        content_pages = [p for p in mono
                         if getattr(p, "category", "") not in ("cover", "catalogue")]
        logger.info("  Doc: %s | %d pages | %d content pages", name, pages, len(content_pages))

    return rag_docs


# ── Step 2: Build KG ────────────────────────────────────────

class AgentTraceCapture:
    """Captures agent intermediate steps for logging."""
    def __init__(self):
        self.steps = []
        self.step_count = 0

    def on_agent_step(self, step_data: dict):
        self.step_count += 1
        self.steps.append(step_data)

        msg_type = step_data.get("type", "?")
        content_preview = ""

        if msg_type == "ai":
            content = step_data.get("content", "")
            tool_calls = step_data.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    args_str = json.dumps(args, ensure_ascii=False)[:300]
                    logger.info(f"  [Step {self.step_count}] 🤖 TOOL CALL: {name}({args_str})")
            elif content:
                content_preview = str(content)[:200]
                logger.info(f"  [Step {self.step_count}] 🤖 AI: {content_preview}")

        elif msg_type == "tool":
            name = step_data.get("name", "?")
            result = step_data.get("content", "")
            result_preview = str(result)[:200]
            logger.info(f"  [Step {self.step_count}] 🔧 TOOL RESULT: {name} → {result_preview}")

    def save_trace(self, filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Agent trace saved to %s (%d steps)", filepath, self.step_count)


async def step2_build_kg(rag_docs):
    """Build knowledge graphs from each document."""
    logger.info("─" * 60)
    logger.info("[STEP 2] Building KG from documents...")
    logger.info(f"  Documents to process: {len(rag_docs)}")
    logger.info("─" * 60)

    from agent.kg_build_agent import KGBuildAgent

    all_stats = []
    agent = KGBuildAgent(db_name=DB_NAME)

    # Override timeout for the global agent session
    agent.timeout = config.settings.KG_EXTRACTION_TIMEOUT
    agent.max_iterations = config.settings.KG_EXTRACTION_MAX_ITERATIONS

    for idx, doc in enumerate(rag_docs):
        doc_name = getattr(doc, "doc_name", f"doc_{idx}")
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Building KG for document {idx+1}/{len(rag_docs)}: {doc_name}")
        logger.info(f"{'─' * 50}")

        t0 = time.time()

        try:
            # Save KG checkpoint between documents
            stats = await agent.build_kg_from_document(doc)
            elapsed = time.time() - t0
            all_stats.append({"doc": doc_name, "stats": stats, "elapsed": elapsed})
            logger.info("Document '%s' complete in %.1fs: entities %d→%d, relations %d→%d",
                         doc_name, elapsed,
                         stats.get("entity_count_before", 0), stats.get("entity_count_after", 0),
                         stats.get("relation_count_before", 0), stats.get("relation_count_after", 0))
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("FAILED building KG for '%s' after %.1fs: %s\n%s",
                          doc_name, elapsed, e, traceback.format_exc())

    await agent.close()
    return all_stats


# ── Step 3: QA / Retrieval ──────────────────────────────────

def extract_entity_names(message: str) -> list[str]:
    """Extract candidate entity names from user message."""
    candidates: list[str] = []
    msg = message.strip()

    quoted = re.findall(r'["\'""\u300c\u300d]([^"\'""\u300c\u300d]+)["\'""\u300c\u300d]', msg)
    candidates.extend(quoted)

    patterns = [
        r'([\u4e00-\u9fff]{2,8}(?:模块|系统|节点|引擎|服务|仪表盘|数据库|传感器|计算机|设备|组件|接口|控制器|处理器|存储器|网络|总线|通道|机柜|命令|结点|主机))',
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
    return result[:10]


def build_context_from_retrieve(result: dict) -> str:
    """Build readable context from sysml_retrieve result."""
    if not result.get("ok"):
        return ""
    entity = result.get("entity", {})
    graph = result.get("relationship_graph", {})

    parts = [f"Entity: {entity.get('name','?')} ({entity.get('type', entity.get('class','?'))})"]

    meta = entity.get("metadata", {})
    if meta.get("description"):
        parts.append(f"  Description: {meta['description'][:500]}")
    if meta.get("properties"):
        parts.append(f"  Properties: {json.dumps(meta['properties'], ensure_ascii=False)}")
    if entity.get("aliases"):
        parts.append(f"  Aliases: {', '.join(entity['aliases'])}")

    members = entity.get("members", [])
    if members:
        lines = [f"  Members ({len(members)}):"]
        for m in members[:40]:
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


async def step3_qa():
    """Run QA using sysml_retrieve on the built KG."""
    logger.info("─" * 60)
    logger.info("[STEP 3] QA / Retrieval")
    logger.info(f"  Question: {QUESTION[:80]}...")
    logger.info("─" * 60)

    # Import MCP server functions
    from scripts.sysml_rag_mcp_server import (
        sysml_load_model, sysml_retrieve, sysml_model_summary,
        _get_manager, _global_manager, _loaded_files,
    )

    # Reset MCP state
    import scripts.sysml_rag_mcp_server as mcp_mod
    mcp_mod._global_manager = None
    mcp_mod._loaded_files.clear()

    kg_path = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.sysml"
    if not kg_path.exists():
        logger.error("KG file not found: %s", kg_path)
        return None

    logger.info("Loading KG: %s", kg_path)
    load_result = sysml_load_model(str(kg_path))
    if not load_result.get("ok"):
        logger.error("Failed to load KG: %s", load_result)
        return None

    summary = sysml_model_summary()
    logger.info("KG loaded: %d entities, %d relations",
                 summary["total_entities"], summary["total_relations"])

    # Extract candidates and retrieve
    candidates = extract_entity_names(QUESTION)
    logger.info("Candidate entity names: %s", candidates)

    # Add specific candidates relevant to the question
    extra_candidates = ["R1P3", "yhst", "smu_tranfer_cmd", "mn0", "SMU", "CMU",
                        "加电", "power", "机柜", "R1P0", "R1P1", "R1P2",
                        "ncid", "smu", "cmu", "r1", "p03a", "p03b", "p03c", "p03d"]
    for ec in extra_candidates:
        if ec not in candidates:
            candidates.append(ec)

    context_parts = []
    retrieved = set()
    not_found = []

    for name in candidates:
        for k_val in [1, 2, 0]:
            try:
                r = sysml_retrieve(name, k=k_val)
                logger.debug("  sysml_retrieve('%s', k=%d): ok=%s", name, k_val, r.get("ok"))
                if r.get("ok"):
                    ename = r.get("entity", {}).get("name", "")
                    if ename in retrieved:
                        continue
                    retrieved.add(ename)
                    ctx = build_context_from_retrieve(r)
                    if ctx:
                        context_parts.append(f"--- {name} (k={k_val}) ---\n{ctx}")
                        logger.info("  ✅ Retrieved: %s (matched_by: %s)", ename, r.get("matched_by", "?"))
                    break
            except Exception as e:
                logger.warning("  sysml_retrieve('%s') error: %s", name, e)
                break
        else:
            if name not in retrieved:
                not_found.append(name)
                logger.info("  ❌ Not found: %s", name)

    context = "\n\n".join(context_parts) if context_parts else "(No SysML entities matched)"

    logger.info("Retrieved: %d entities | Not found: %d | Context: %d chars",
                 len(retrieved), len(not_found), len(context))

    # ── LLM Synthesis ──
    from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import SecretStr

    QA_SYSTEM_PROMPT = """You are a QA agent trained on SysML knowledge graph data.
Answer questions using the provided SysML entity/relation data only.
Do not fabricate information. State clearly if data is insufficient.
Answer in Chinese. End with "## QA 反馈" section."""

    user_prompt = f"""## SysML KG 检索结果

KG: {summary['total_entities']} entities, {summary['total_relations']} relations
Candidates: {json.dumps(candidates, ensure_ascii=False)}
Not found: {json.dumps(not_found, ensure_ascii=False)}

{context[:14000]}

## 参考回答:
{REFERENCE}

## 用户问题
{QUESTION}

请根据 SysML KG 数据回答。如果 KG 数据不足，请说明缺失了什么。在答案末尾添加 "## QA 反馈"。"""

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Calling LLM for QA synthesis...")
    llm = ChatOpenAIWithReasoning(
        model=config.settings.LLM_MODEL or "qwen3:8b",
        temperature=0.1,
        api_key=SecretStr(config.settings.OPENAI_API_KEY or "ollama"),
        base_url=config.settings.OPENAI_API_URL or "http://localhost:11434/v1",
        timeout=300,
        streaming=False,
    )

    t0 = time.time()
    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=600)
        answer = str(response.content)
        elapsed = time.time() - t0
        logger.info("QA LLM response in %.1fs (%d chars)", elapsed, len(answer))
    except asyncio.TimeoutError:
        answer = "LLM 请求超时（600s）。"
        logger.error("QA LLM timeout")
    except Exception as e:
        answer = f"LLM 调用失败: {type(e).__name__}: {e}"
        logger.error("QA LLM error: %s", e)

    return {
        "question": QUESTION,
        "answer": answer,
        "retrieved": list(retrieved),
        "not_found": not_found,
        "kg_summary": summary,
        "candidates": candidates,
    }


# ── Main ─────────────────────────────────────────────────────

async def main():
    total_start = time.time()

    try:
        # ── Check existing build state ──
        kg_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.sysml"
        build_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.build.json"
        if build_file.exists():
            try:
                with open(build_file, 'r', encoding='utf-8') as f:
                    prev_state = json.load(f)
                logger.info("Found previous build state: %s", json.dumps(prev_state, ensure_ascii=False))
            except Exception:
                pass
        if kg_file.exists():
            logger.info("Resuming from existing KG: %s", kg_file)
        else:
            logger.info("Starting fresh KG build")

        # Step 1: Load documents
        rag_docs = await step1_load_documents()

        if not rag_docs:
            logger.error("No documents loaded. Aborting.")
            return 1

        # Step 2: Build KG — 只处理湖超 (最相关的文档，141 叶子页)
        logger.info("\n\n")
        logger.info("=" * 70)
        logger.info("NOTE: Building KG from 湖超 only (141 leaf pages). Model: %s", KG_MODEL)
        logger.info("=" * 70)
        logger.info("\n")

        # Filter to only 湖超 doc
        huchao_docs = [d for d in rag_docs if "湖超" in getattr(d, "doc_name", "")]
        if not huchao_docs:
            huchao_docs = rag_docs[:1]  # fallback to first doc
        logger.info("Selected document: %s", getattr(huchao_docs[0], "doc_name", "?"))
        build_stats = await step2_build_kg(huchao_docs)

        # Step 3: QA
        qa_result = await step3_qa()

        # ── Final Summary ──
        total_elapsed = time.time() - total_start
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("HEADLESS TEST COMPLETE")
        logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
        logger.info("=" * 70)

        # Build stats
        logger.info("\n## Build Statistics")
        for bs in build_stats:
            s = bs["stats"]
            logger.info(f"  {bs['doc']}: {bs['elapsed']:.0f}s | "
                         f"entities {s.get('entity_count_before',0)}→{s.get('entity_count_after',0)} | "
                         f"relations {s.get('relation_count_before',0)}→{s.get('relation_count_after',0)}")

        if qa_result:
            logger.info(f"\n## QA Results")
            logger.info(f"  KG: {qa_result['kg_summary']['total_entities']}e / {qa_result['kg_summary']['total_relations']}r")
            logger.info(f"  Retrieved: {len(qa_result['retrieved'])} entities")
            logger.info(f"  Not found: {len(qa_result['not_found'])} names")
            logger.info(f"\n## Answer")
            logger.info(qa_result['answer'])

            # Write answer to file
            answer_file = OUTPUT_DIR / f"answer_{RUN_TIMESTAMP}.txt"
            answer_file.write_text(qa_result['answer'], encoding="utf-8")

        # Write full results JSON
        results = {
            "timestamp": RUN_TIMESTAMP,
            "db": DB_NAME,
            "model": config.settings.KG_EXTRACTION_MODEL,
            "total_time_s": round(total_elapsed, 1),
            "build_stats": [
                {"doc": bs["doc"], "elapsed_s": round(bs["elapsed"], 1),
                 "stats": bs["stats"], "errors": bs.get("errors", [])}
                for bs in build_stats
            ],
            "qa": qa_result,
            "build_state_file": str(build_file) if build_file.exists() else None,
        }
        results_file = OUTPUT_DIR / f"results_{RUN_TIMESTAMP}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"\nFull results saved to: {results_file}")

        print(f"\nLog: {LOG_FILE}")
        print(f"Results: {results_file}")
        print(f"Answer: {OUTPUT_DIR}/answer_{RUN_TIMESTAMP}.txt")

    except Exception as e:
        logger.error("FATAL: %s\n%s", e, traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
