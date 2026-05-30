#!/usr/bin/env python3
"""
KG Build: 湖超-硬件维护手册 → AIOPS_New
========================================
使用 qwen3:8b 全流程构建知识图谱。
输出同时打印到终端和日志文件。
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

# ── Strict mode: abort on any error ───────────────────────────
import warnings
warnings.simplefilter("ignore")  # dont die on Pillow/numpy deprecation warnings

_kg_logger = logging.getLogger("agent.kg_build_agent")
_kg_logger.setLevel(logging.DEBUG)  # detailed LLM calls + MCP operations
_kg_logger.propagate = True

class FatalWarnHandler(logging.Handler):
    """Convert ERROR/CRITICAL logs to exceptions, abort immediately."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise RuntimeError(f"[FATAL] {record.levelname}: {record.getMessage()}")

_kg_logger = logging.getLogger("agent.kg_build_agent")
_kg_logger.addHandler(FatalWarnHandler())
_kg_logger.setLevel(logging.WARNING)

# Suppress noisy libraries
for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
            "llama_index", "urllib3", "asyncio", "faiss", "PIL",
            "mcp.client.stdio", "rag.ocr", "rag.documents"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────────
DB_NAME = "AIOPS_New"
DOC_PATH = "/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = ROOT_DIR / "tmp" / "kg_builds"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"{DB_NAME}.log"  # 每个数据库一个日志文件，追加写入


# ── Logging: stdout + file ────────────────────────────────────
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
log_fh.write(f"BUILD START: {RUN_TS}\n")
log_fh.write(f"{'='*70}\n")
log_fh.flush()
sys.stderr = Tee(sys.__stderr__, log_fh)  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger("kg_build")

# ── Config override ───────────────────────────────────────────
config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL="qwen3:8b",
    KG_LIGHT_MODEL="qwen3:8b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=120,
    KG_EXTRACTION_MAX_ITERATIONS=200,
    BATCH_CONCURRENCY=5,
    KG_KEEP_ALIVE="600s",
    OCR_MODEL=None,
    OCR_API_URL=None,
)


async def main():
    logger.info("=" * 70)
    logger.info("KG BUILD: 湖超-硬件维护手册 → AIOPS_New")
    logger.info(f"Model: {config.settings.KG_EXTRACTION_MODEL}")
    logger.info(f"Concurrency: {config.settings.BATCH_CONCURRENCY}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 70)

    # ── Load document ──
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
    logger.info("Document: %s | %d pages | %d content pages", doc_name, pages, len(content_pages))

    # ── Build KG ──
    from agent.kg_build_agent import KGBuildAgent

    agent = KGBuildAgent(db_name=DB_NAME)
    agent.timeout = 120
    agent.max_iterations = 200

    try:
        logger.info("\nBuilding KG from document...")
        stats = await agent.build_kg_from_document(doc)
        elapsed = time.time() - t0

        logger.info("=" * 70)
        logger.info("BUILD COMPLETE in %.0fs (%.1fm)", elapsed, elapsed / 60)
        logger.info("Entities: %d → %d", stats.get("entity_count_before", 0), stats.get("entity_count_after", 0))
        logger.info("Relations: %d → %d", stats.get("relation_count_before", 0), stats.get("relation_count_after", 0))
        logger.info("Phase times: P1=%.0fs P2=%.0fs P3=%.0fs P4=%.0fs",
                     stats.get("phase1_time_s", 0),
                     stats.get("phase2_time_s", 0),
                     stats.get("phase3_time_s", 0),
                     stats.get("phase4_time_s", 0))
        logger.info("Full stats: %s", json.dumps(stats, ensure_ascii=False, indent=2))
        logger.info("=" * 70)

        # ── Show final KG state ──
        kg_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.sysml"
        if kg_file.exists():
            size_kb = kg_file.stat().st_size / 1024
            logger.info("KG file: %s (%.1f KB)", kg_file, size_kb)

        build_file = ROOT_DIR / "database" / DB_NAME / "knowledge_graph.build.json"
        if build_file.exists():
            with open(build_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info("Build state: %s", json.dumps(state, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error("BUILD FAILED: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        await agent.close()

    logger.info(f"\nLog saved to: {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
