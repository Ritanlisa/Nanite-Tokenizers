#!/usr/bin/env python3
"""
递归级联 KG 构建: 湖超-硬件维护手册 → AIOPS_New
=================================================
- Phase 0: gemma4:31b 根实体识别
- Phase 1: gemma4:31b 根小节完整提取
- Phase 2: 实体全文传播 → 构建队列
- Phase 3: qwen3:8b 级联小节处理
- 自动断点续跑 (Ctrl+C 安全)
- 详细日志追加到单一日志文件

Usage:
  python3 scripts/build_kg_recursive.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from gephi_streamer import GephiStreamer

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DB_NAME = "Intel_Manual_v2"
DOC_PATH = "/home/hjq/Nanite-Tokenizers-lite/Intel® 64 和 IA-32 架构软件开发者手册合集.pdf"
ROOT_MODEL = "qwen3:8b"
EXTRACT_MODEL = "qwen3:8b"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = ROOT_DIR / "tmp" / "kg_builds"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"{DB_NAME}.log"


# ═══════════════════════════════════════════════════════════════
# Logging: Tee to file + stderr (append mode)
# ═══════════════════════════════════════════════════════════════

class Tee:
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


log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
log_fh.write(f"\n{'='*70}\n")
log_fh.write(f"SESSION: {RUN_TS} | PIPELINE: recursive\n")
log_fh.write(f"  ROOT_MODEL: {ROOT_MODEL}\n")
log_fh.write(f"  EXTRACT_MODEL: {EXTRACT_MODEL}\n")
log_fh.write(f"  DOC: {DOC_PATH}\n")
log_fh.write(f"  DB: {DB_NAME}\n")
log_fh.write(f"{'='*70}\n")
log_fh.flush()

sys.stderr = Tee(sys.__stderr__, log_fh)  # type: ignore

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# Enable full debug for KG build agent (includes LLM prompt/response)
for name in ["agent.kg_build_agent", "kg_build"]:
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    l.propagate = True

# Suppress noisy external libraries
for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
            "llama_index", "urllib3", "asyncio", "faiss", "PIL",
            "mcp.client.stdio", "rag.ocr", "rag.documents"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger("kg_build")
# ── Gephi Streaming ──
gephi = GephiStreamer()
gephi.connect()
if gephi.connected:
    logger.info("Gephi streaming connected")
else:
    logger.info("Gephi not available (streaming disabled)")


# ═══════════════════════════════════════════════════════════════
# MCP call interceptor for detailed logging
# ═══════════════════════════════════════════════════════════════

_original_call_tool = None


async def _logged_call_tool(session, tool_name: str, arguments: dict) -> str:
    """Intercept MCP call_tool to log every operation with full parameters."""
    t0 = time.time()
    try:
        result = await _original_call_tool(session, tool_name, arguments)
        elapsed = time.time() - t0
        result_parsed = None
        try:
            result_parsed = json.loads(result)
            ok = result_parsed.get("ok", True)
        except Exception:
            ok = "?"

        # Log entity/relation operations specially
        if tool_name == "sysml_add_entity":
            logger.info("MCP +ENTITY | type=%s | name=%s | qn=%s | source=%s | ok=%s | %.2fs",
                         arguments.get("entity_type", "?"), arguments.get("name", "?"),
                         result_parsed.get("qualified_name", "?") if result_parsed else "?",
                         arguments.get("source_sections", []), ok, elapsed)
            if ok and gephi.connected:
                qn = result_parsed.get("qualified_name", "") if result_parsed else ""
                if qn:
                    gephi.sync_entity(qn, entity_type=arguments.get("entity_type", "PartDef"))
        elif tool_name == "sysml_update_entity":
            logger.info("MCP UPDATE | qn=%s | append_source=%s | %.2fs",
                         arguments.get("qualified_name", "?"),
                         arguments.get("append_source_sections", []), elapsed)
        elif tool_name == "sysml_add_relation":
            logger.info("MCP +RELATION | type=%s | %s→%s | name=%s | source=%s | ok=%s | %.2fs",
                         arguments.get("relation_type", "?"),
                         arguments.get("source", "?"), arguments.get("target", "?"),
                         arguments.get("name", "?"), arguments.get("source_sections", []),
                         ok, elapsed)
            if ok and gephi.connected:
                src = arguments.get("source", "")
                tgt = arguments.get("target", "")
                rid = arguments.get("name", "") or f"{src}→{tgt}"
                if src and tgt:
                    gephi.sync_relation(rid, src, tgt, arguments.get("relation_type", "ReferenceUsage"), label=arguments.get("name", ""))
        elif tool_name == "sysml_merge_entities":
            logger.info("MCP MERGE | %s → %s | ok=%s | %.2fs",
                         arguments.get("source", "?"), arguments.get("target", "?"),
                         ok, elapsed)
        elif tool_name == "sysml_search_entity":
            total = result_parsed.get("total_matches", 0) if result_parsed else 0
            if total > 0:
                m = result_parsed.get("matches", [{}])[0]
                logger.debug("MCP SEARCH | q=%s → %d matches | top: %s (%.2f) | %.2fs",
                              arguments.get("query", "?")[:60], total,
                              m.get("qualified_name", "?"), m.get("confidence", 0), elapsed)
        elif tool_name == "sysml_add_alias":
            logger.debug("MCP +ALIAS | entity=%s | alias=%s | %.2fs",
                          arguments.get("qualified_name", "?"), arguments.get("alias", "?"), elapsed)
        elif tool_name in ("sysml_save_model", "sysml_load_model"):
            logger.info("MCP %s | file=%s | %.2fs",
                         tool_name, arguments.get("file_path", "?"), elapsed)
        elif tool_name == "sysml_model_summary":
            logger.debug("MCP SUMMARY | entities=%d relations=%d | %.2fs",
                          result_parsed.get("total_entities", 0) if result_parsed else 0,
                          result_parsed.get("total_relations", 0) if result_parsed else 0,
                          elapsed)
        else:
            logger.debug("MCP %s | params=%s | %.2fs",
                          tool_name,
                          json.dumps(arguments, ensure_ascii=False)[:200],
                          elapsed)
        return result
    except Exception as e:
        logger.error("MCP FAILED | %s | %s | %.2fs",
                      tool_name, e, time.time() - t0)
        raise


def _patch_mcp_session(session):
    """Patch MCPSession.call_tool to log all operations."""
    global _original_call_tool
    if _original_call_tool is None:
        _original_call_tool = type(session).call_tool

    import types
    async def patched_call_tool(self, tool_name, arguments=None):
        return await _logged_call_tool(self, tool_name, arguments or {})
    session.call_tool = types.MethodType(patched_call_tool, session)


# ═══════════════════════════════════════════════════════════════
# Config Override
# ═══════════════════════════════════════════════════════════════

config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL=ROOT_MODEL,
    KG_LIGHT_MODEL=EXTRACT_MODEL,
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=300,
    KG_EXTRACTION_MAX_ITERATIONS=500,
    BATCH_CONCURRENCY=8,
    KG_KEEP_ALIVE="3600s",
    OCR_MODEL=None,
    OCR_API_URL=None,
)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    logger.info("=" * 70)
    logger.info("RECURSIVE KG BUILD: 湖超-硬件维护手册 → %s", DB_NAME)
    logger.info("  Root model: %s (Phase 0/1)", ROOT_MODEL)
    logger.info("  Extract model: %s (Phase 3)", EXTRACT_MODEL)
    logger.info("  Log file: %s", LOG_FILE)
    logger.info("=" * 70)

    # Initialize MCP early (before heavy document loading)
    try:
        from mcp_client.mcp_session import create_sysml_mcp_session
        _early_mcp = create_sysml_mcp_session()
        await asyncio.wait_for(_early_mcp.initialize(), timeout=30)
        logger.info("MCP session established early")
    except Exception as e:
        logger.warning("Early MCP init failed: %s (will retry later)", e)
        _early_mcp = None

    # Load document
    from rag.documents import load_rag_documents_from_paths
    from rag.engine import SUPPORTED_RAG_EXTENSIONS

    t0 = time.time()
    logger.info("Loading document: %s", DOC_PATH)
    rag_docs = load_rag_documents_from_paths([DOC_PATH], SUPPORTED_RAG_EXTENSIONS)
    if not rag_docs:
        logger.error("No documents loaded!")
        return 1

    doc = rag_docs[0]
    doc_name = getattr(doc, "doc_name", "湖超")
    pages = getattr(doc, "page_count", 0)
    mono = doc.get_mono_pages() or []
    content_pages = [p for p in mono
                     if getattr(p, "category", "") not in ("cover", "catalogue")]
    logger.info("Document: %s | %d total pages | %d content pages",
                 doc_name, pages, len(content_pages))

    # Show section overview
    from agent.kg_build_agent import DocumentTreeState
    tree = DocumentTreeState(doc)
    tree_str = json.dumps(tree.get_tree_structure(), ensure_ascii=False, indent=2)
    logger.info("Document tree: %d leaf pages\n%s",
                 tree.total_pages, tree_str[:2000])

    # Create agent
    from agent.kg_build_agent import KGBuildAgent

    agent = KGBuildAgent(db_name=DB_NAME, model=ROOT_MODEL,
                         light_model=EXTRACT_MODEL)
    # Pass pre-initialized MCP session if available
    if _early_mcp is not None:
        agent._mcp_session = _early_mcp
    agent.timeout = 300
    agent.max_iterations = 500

    # Patch MCP session for detailed logging
    await agent.initialize()
    if agent._mcp_session:
        _patch_mcp_session(agent._mcp_session)
        logger.info("MCP session patched for detailed operation logging")

    # Build
    try:
        logger.info("\nStarting recursive KG build...")
        stats = await agent.build_kg_recursive(doc)
        elapsed = time.time() - t0

        logger.info("=" * 70)
        logger.info("BUILD COMPLETE in %.0fs (%.1fm)",
                     elapsed, elapsed / 60)
        logger.info("Entities: %d → %d",
                     stats.get("entity_count_before", 0),
                     stats.get("entity_count_after", 0))
        logger.info("Relations: %d → %d",
                     stats.get("relation_count_before", 0),
                     stats.get("relation_count_after", 0))
        logger.info("Full stats: %s",
                     json.dumps(stats, ensure_ascii=False, indent=2))
        logger.info("=" * 70)

        kg_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.sysml"
        if kg_file.exists():
            size_kb = kg_file.stat().st_size / 1024
            logger.info("KG file: %s (%.1f KB)", kg_file, size_kb)

        build_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.build.json"
        if build_file.exists():
            with open(build_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info("Build state: phase=%s pipeline=%s",
                         state.get("documents", {}).get(doc_name, {}).get("phase", "?"),
                         state.get("documents", {}).get(doc_name, {}).get("pipeline", "?"))

    except KeyboardInterrupt:
        logger.warning("BUILD INTERRUPTED (Ctrl+C), state saved for resume")
        return 1
    except Exception as e:
        logger.error("BUILD FAILED: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        await agent.close()
        if gephi.connected:
            gephi.disconnect()
            logger.info("Gephi streaming disconnected")

    logger.info("Log saved to: %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
