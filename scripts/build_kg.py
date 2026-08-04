#!/usr/bin/env python3
"""
Unified KG build entry point (merged from three legacy scripts).
================================================================
Merges the three near-duplicate KG build scripts into one parameterized
entry point. All three original behaviors remain reachable via arguments:

  - build_kg_comprehensive.py  →  --engine direct   (direct LLM extraction
                                                      + SysMLManager serialization)
  - build_kg_huchao_qwen3.py   →  --engine agent --strict --log-to-file
                                                      (KGBuildAgent full pipeline,
                                                       strict mode, tee log file)
  - build_kg_湖超.py           →  --engine agent --print-kg
                                                      (KGBuildAgent full pipeline,
                                                       dumps the whole KG file)

Usage (run from the repo root; requires Ollama serving the configured model):

  # direct LLM extraction (was build_kg_comprehensive.py)
  python scripts/build_kg.py --engine direct

  # KGBuildAgent full pipeline, strict + file log (was build_kg_huchao_qwen3.py)
  python scripts/build_kg.py --engine agent \
      --doc-path /home/ritanlisa/文档/湖超-硬件维护手册20231225.doc \
      --db-name AIOPS_New --strict --log-to-file

  # KGBuildAgent full pipeline, full KG dump (was build_kg_湖超.py)
  python scripts/build_kg.py --engine agent --print-kg

The extraction logic itself is unchanged from the original scripts — this is
a pure merge of entry points, not a rewrite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_DOC = "database/AIOPS/docs/湖超-硬件维护手册20231225.doc"
DEFAULT_DB = "AIOPS"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_PACKAGE = "湖超-硬件维护手册"


# ── strict mode / tee helpers (was build_kg_huchao_qwen3.py) ────────────

class FatalWarnHandler(logging.Handler):
    """Convert ERROR/CRITICAL logs to exceptions, abort immediately."""

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise RuntimeError(f"[FATAL] {record.levelname}: {record.getMessage()}")


class Tee:
    """Duplicate writes to multiple streams."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def fileno(self):
        return self.files[0].fileno()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SysML knowledge graph from a technical document",
    )
    parser.add_argument(
        "--engine", choices=["direct", "agent"], default="direct",
        help="direct: single LLM call per chunk + SysMLManager serialization "
             "(was build_kg_comprehensive.py); agent: KGBuildAgent full pipeline "
             "(was build_kg_huchao_qwen3.py / build_kg_湖超.py)",
    )
    parser.add_argument("--doc-path", default=DEFAULT_DOC,
                        help="path to the source document (default: %(default)s)")
    parser.add_argument("--db-name", default=DEFAULT_DB,
                        help="target knowledge-graph database name (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model to use (default: %(default)s)")
    parser.add_argument("--chunk-size", type=int, default=6000,
                        help="chunk size in chars for the direct engine (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="LLM timeout in seconds for the agent engine (default: %(default)s)")
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="max iterations for the agent engine (default: %(default)s)")
    parser.add_argument("--package-name", default=DEFAULT_PACKAGE,
                        help="root SysML package name for the direct engine (default: %(default)s)")
    parser.add_argument("--strict", action="store_true",
                        help="abort on any ERROR/CRITICAL log (was build_kg_huchao_qwen3.py behavior)")
    parser.add_argument("--log-to-file", action="store_true",
                        help="tee stdout/stderr to tmp/kg_builds/<db>.log "
                             "(was build_kg_huchao_qwen3.py behavior)")
    parser.add_argument("--print-kg", action="store_true",
                        help="print the whole generated KG file (was build_kg_湖超.py behavior)")
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════
# Direct engine — was build_kg_comprehensive.py
# ════════════════════════════════════════════════════════════════════════

COMPREHENSIVE_EXTRACTION_PROMPT = """你是SysML v2知识图谱构建专家。请从技术文档中**全面且详细**地提取系统架构信息。

## 实体提取（必须覆盖每一段内容）：
输出格式: {"type":"PartDef"|"AttributeDef"|"PortDef"|"ItemDef"|"RequirementDef","name":"...","description":"...","properties":{},"value":"数值","unit":"单位"}

## 关系提取（必须连接所有相关实体）：
输出格式: {"type":"Connection"|"Allocation"|"Interface","source":"源实体名","target":"目标实体名","description":"关系描述"}

## 要求：
- 技术规格中的**每一个参数、数值、指标都要提取**为AttributeDef
- 每个**连接关系、数据流、信号都要提取**为Connection
- 实体名称使用原文术语，描述保留原文关键信息
- 关系必须覆盖：物理连接、数据流、电源连接、信号传输、监控链路、冷却管道

每行输出一个JSON对象。不要遗漏！"""


async def extract_comprehensive(text: str, client, model: str) -> list[dict]:
    """Single LLM call extracting comprehensive entities/relations (unchanged)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMPREHENSIVE_EXTRACTION_PROMPT},
            {"role": "user", "content": text[:12000]},
        ],
        temperature=0.1,
        max_tokens=3000,
        timeout=180,
    )
    content = resp.choices[0].message.content or ""

    results = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "type" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try to extract JSON from the line
            m = re.search(r'\{.*\}', line)
            if m:
                try:
                    results.append(json.loads(m.group()))
                except Exception:
                    pass
    return results


async def run_direct(args: argparse.Namespace) -> int:
    """Comprehensive KG build via direct LLM calls (was build_kg_comprehensive.py)."""
    from openai import OpenAI
    from tqdm import tqdm

    from rag.documents import load_rag_documents_from_paths
    from rag.engine import SUPPORTED_RAG_EXTENSIONS

    config.settings = config.settings.update(
        RAG_DB_NAME=args.db_name, RAG_DB_NAMES=[args.db_name], OCR_MODEL=None,
    )

    client = OpenAI(
        api_key=config.settings.OPENAI_API_KEY,
        base_url=config.settings.OPENAI_API_URL,
        timeout=300,
    )

    print("=" * 60)
    print(f"Comprehensive KG Build: {Path(args.doc_path).name}")
    print(f"Model: {args.model}")
    print("=" * 60)
    t0 = time.time()

    # Load document
    print("\n[1] Loading document...")
    rag_docs = load_rag_documents_from_paths([args.doc_path], SUPPORTED_RAG_EXTENSIONS)
    if not rag_docs:
        print("  ERROR: Failed to load document!")
        return 1
    doc = rag_docs[0]
    print(f"  Pages: {doc.page_count}")

    # Get all text
    pages = doc.get_mono_pages()
    all_text = "\n\n".join(
        getattr(p, "markdown_text", "") or ""
        for p in pages
        if (getattr(p, "markdown_text", "") or "").strip()
        and getattr(p, "category", "") not in ("cover", "catalogue")
    )
    print(f"  Total content text: {len(all_text)} chars")

    # Chunk into args.chunk_size-char segments
    chunk_size = args.chunk_size
    chunks = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]
    print(f"  Chunks: {len(chunks)}")

    # Extract from each chunk
    all_results = []
    with tqdm(total=len(chunks), desc="  Extracting", unit="chunk") as bar:
        for chunk in chunks:
            results = await extract_comprehensive(chunk, client, args.model)
            all_results.extend(results)
            bar.set_postfix_str(f"{len(results)} items")
            bar.update(1)

    print(f"\n  Total extracted items: {len(all_results)}")

    # Categorize
    entities = [r for r in all_results
                if r["type"] in ("PartDef", "AttributeDef", "PortDef", "ItemDef", "RequirementDef")]
    relations = [r for r in all_results
                 if r["type"] in ("Connection", "Allocation", "Interface")]

    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")

    # Build SysML
    from sysml.sysml_manager import SysMLManager
    mgr = SysMLManager()

    # Create main document package (in memory, will be serialized)
    doc_pkg = args.package_name
    mgr.add_entity_with_metadata("Package", doc_pkg)

    # Group entities by chapter and create sub-packages
    packages = {}
    for e in entities:
        source = e.get("source_section", "") or ""
        chapter_match = re.search(r'(第[一二三四五六七八九十\d]+章|[A-Za-z]+ \d+)', source)
        pkg = chapter_match.group(1) if chapter_match else "通用组件"
        packages.setdefault(pkg, []).append(e)

    # Create sub-packages inside the doc package
    pkg_map = {}  # pkg_name → entity object
    for pkg_name in sorted(packages):
        elem = mgr.add_entity_with_metadata(
            "Package", pkg_name, parent_package=doc_pkg,
        )
        pkg_map[pkg_name] = elem.qualified_name if elem else doc_pkg

    # Add entities to their respective sub-packages
    for pkg_name, items in packages.items():
        parent_qn = pkg_map.get(pkg_name, doc_pkg)
        for item in items:
            etype = item["type"]
            name = item.get("name", "").strip()
            desc = item.get("description", "")
            props = item.get("properties", {})

            mgr.add_entity_with_metadata(
                entity_type=etype,
                name=name,
                description=desc,
                properties=props,
                source_sections=[item.get("source_section", "")],
                parent_package=parent_qn,
            )

    # Add relations to the doc package
    for r in relations:
        rtype = r["type"].lower()
        src = r.get("source", "")
        tgt = r.get("target", "")
        desc = r.get("description", "")
        if src and tgt:
            mgr.add_relation(rtype, src, tgt, description=desc, parent_package=doc_pkg)

    # Save
    kg_path = Path("database") / args.db_name / "knowledge_graph.sysml"
    mgr.save_to_file(str(kg_path))

    elapsed = time.time() - t0
    print(f"\n[Result] Saved to {kg_path}")
    print(f"  Total items: {len(all_results)}")
    print(f"  Time: {elapsed:.0f}s")

    # Show sample
    with open(kg_path) as f:
        content = f.read()
    print(f"\n  File size: {len(content)} chars, {len(content.splitlines())} lines")
    print(f"  First 500 chars:\n{content[:500]}")
    return 0


# ════════════════════════════════════════════════════════════════════════
# Agent engine — was build_kg_huchao_qwen3.py / build_kg_湖超.py
# ════════════════════════════════════════════════════════════════════════

async def run_agent(args: argparse.Namespace) -> int:
    """KG build via the unified KGBuildAgent (was build_kg_huchao_qwen3.py / build_kg_湖超.py)."""
    from rag.documents import load_rag_documents_from_paths
    from rag.engine import SUPPORTED_RAG_EXTENSIONS
    from agent.kg_build_agent import KGBuildAgent

    config.settings = config.settings.update(
        KG_EXTRACTION_ENABLED=True,
        KG_EXTRACTION_MODEL=args.model,
        KG_LIGHT_MODEL=args.model,
        KG_EXTRACTION_TEMPERATURE=0.1,
        KG_EXTRACTION_TIMEOUT=args.timeout,
        KG_EXTRACTION_MAX_ITERATIONS=args.max_iterations,
        BATCH_CONCURRENCY=5,
        KG_KEEP_ALIVE="600s",
        RAG_DB_NAME=args.db_name,
        RAG_DB_NAMES=[args.db_name],
        OCR_MODEL=None,
        OCR_API_URL=None,
        AGENT_VERBOSE=False,
    )

    t0 = time.time()
    print("=" * 60)
    print(f"KG Build: {Path(args.doc_path).name}")
    print(f"Model: {args.model}")
    print(f"Database: {args.db_name}")
    print("=" * 60)

    # ── Load document ──
    print("\n[1/3] Loading & building document tree...")
    rag_docs = load_rag_documents_from_paths([args.doc_path], SUPPORTED_RAG_EXTENSIONS)
    if not rag_docs:
        print("  ERROR: No documents loaded!")
        return 1

    doc = rag_docs[0]
    doc_name = getattr(doc, "doc_name", Path(args.doc_path).stem)
    pages = getattr(doc, "page_count", 0)
    mono = doc.get_mono_pages() or []
    content_pages = [p for p in mono
                     if getattr(p, "category", "") not in ("cover", "catalogue")]
    print(f"  Document: {doc_name} | {pages} pages | {len(content_pages)} content pages")
    print(f"  Chunks: {len(getattr(doc, 'chunk_documents', []) or [])}")
    t1 = time.time()
    print(f"  Time: {t1 - t0:.1f}s")

    # ── Build KG via unified agent ──
    est_minutes = max(1, len(content_pages) // 3)
    print(f"\n[2/3] KG extraction (unified agent, ~{len(content_pages)} pages, "
          f"est ~{est_minutes}min, model: {args.model})...")
    print("  Agent will autonomously navigate the document tree, extract "
          "entities+relations, and check off pages.")

    agent = KGBuildAgent(db_name=args.db_name)
    agent.timeout = args.timeout
    agent.max_iterations = args.max_iterations

    try:
        stats = await agent.build_kg_from_document(doc)
    except Exception as e:
        print(f"BUILD FAILED: {e}\n{traceback.format_exc()}")
        return 1
    finally:
        await agent.close()

    t2 = time.time()
    print(f"  Extraction done in {t2 - t1:.1f}s")

    # ── Show final KG state ──
    print("=" * 60)
    print(f"BUILD COMPLETE in {t2 - t0:.0f}s ({((t2 - t0) / 60):.1f}m)")
    print(f"Entities: {stats.get('entity_count_before', 0)} → {stats.get('entity_count_after', 0)}")
    print(f"Relations: {stats.get('relation_count_before', 0)} → {stats.get('relation_count_after', 0)}")
    print(f"Phase times: P1={stats.get('phase1_time_s', 0):.0f}s "
          f"P2={stats.get('phase2_time_s', 0):.0f}s "
          f"P3={stats.get('phase3_time_s', 0):.0f}s "
          f"P4={stats.get('phase4_time_s', 0):.0f}s")
    print(f"Full stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    print("=" * 60)

    kg_file = ROOT_DIR / "database" / args.db_name / "knowledge_graph.sysml"
    if kg_file.exists():
        size_kb = kg_file.stat().st_size / 1024
        print(f"KG file: {kg_file} ({size_kb:.1f} KB)")

    build_file = ROOT_DIR / "database" / args.db_name / "knowledge_graph.build.json"
    if build_file.exists():
        with open(build_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        print(f"Build state: {json.dumps(state, ensure_ascii=False, indent=2)}")

    # ── Optional full KG dump (was build_kg_湖超.py) ──
    if args.print_kg and kg_file.exists():
        content = kg_file.read_text(encoding="utf-8")
        entity_lines = [l for l in content.splitlines()
                        if any(kw in l for kw in ("def ", "usage ", "part ", "attribute ",
                                                  "port ", "item ", "requirement ", "package ",
                                                  "connect ", "interface ", "allocation "))]
        print(f"\n{'=' * 60}")
        print(f"KG file: {kg_file}")
        print(f"Size: {len(content)} chars, ~{len(entity_lines)} entity/relation entries")
        print(f"Total time: {t2 - t0:.1f}s")
        print(f"{'=' * 60}")
        print()
        print(content)
        print()
        print(f"{'=' * 60}")
        print(f"  Done! KG exported to {kg_file}")
        print(f"{'=' * 60}")
    return 0


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════

async def main() -> int:
    args = parse_args()

    # --strict: abort on ERROR/CRITICAL (was build_kg_huchao_qwen3.py)
    if args.strict:
        warnings.simplefilter("ignore")
        _agent_logger = logging.getLogger("agent.kg_build_agent")
        _agent_logger.addHandler(FatalWarnHandler())
        _agent_logger.setLevel(logging.WARNING)

    # --log-to-file: tee stdout/stderr to tmp/kg_builds/<db>.log (was build_kg_huchao_qwen3.py)
    if args.log_to_file:
        log_dir = ROOT_DIR / "tmp" / "kg_builds"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{args.db_name}.log"
        log_fh = open(log_file, "a", encoding="utf-8", buffering=1)
        log_fh.write(f"\n{'=' * 70}\n")
        log_fh.write(f"BUILD START: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        log_fh.write(f"{'=' * 70}\n")
        log_fh.flush()
        sys.stdout = Tee(sys.__stdout__, log_fh)
        sys.stderr = Tee(sys.__stderr__, log_fh)

    # Suppress noisy libraries
    for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
                "llama_index", "urllib3", "asyncio", "faiss", "PIL",
                "mcp.client.stdio", "rag.ocr", "rag.documents"]:
        logging.getLogger(lib).setLevel(logging.ERROR)

    if args.engine == "direct":
        return await run_direct(args)
    return await run_agent(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
