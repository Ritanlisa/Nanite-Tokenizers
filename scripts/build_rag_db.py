#!/usr/bin/env python3
"""
Headless RAG DB build from document files.
Usage:
  python scripts/build_rag_db.py --db AIOPS_New --docs doc1.pdf doc2.doc doc3.pdf
  python scripts/build_rag_db.py  # uses defaults
"""

from __future__ import annotations

import sys
import os
import asyncio
import shutil
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_DB_NAME = "AIOPS_New"
DEFAULT_DOCS = [
    Path("/home/ritanlisa/文档/初步验收与试运行分册-6-硬件维护手册 - 1227.doc"),
    Path("/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc"),
    Path("/home/ritanlisa/文档/浪潮虚拟化InCloud Sphere 6.5.1运维手册.pdf"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_rag_db")


def build_rag_db(db_name: str, doc_paths: list[Path]) -> int:
    """Create/replace a RAG DB with the given documents."""

    # ── 1. Validate input
    doc_paths = [p.resolve() for p in doc_paths]
    missing = [str(p) for p in doc_paths if not p.exists()]
    if missing:
        logger.error("Documents not found: %s", missing)
        return 1

    # ── 2. Configure DB
    persist_dir = Path(config.settings.PERSIST_DIR).resolve()
    db_dir = persist_dir / db_name
    docs_dir = db_dir / "docs"

    logger.info("DB dir:  %s", db_dir)
    logger.info("Docs dir: %s", docs_dir)
    logger.info("Documents (%d):", len(doc_paths))
    for p in doc_paths:
        logger.info("  %s (%s)", p.name, p.suffix)

    # ── 3. Set config
    config.settings = config.settings.update(
        RAG_DB_NAME=db_name,
        RAG_DB_NAMES=[db_name],
        ENABLE_RAG=True,
    )

    # ── 4. Prepare directories
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous docs
    for f in docs_dir.iterdir():
        if f.is_file():
            f.unlink()

    # Copy source documents into docs directory
    copied = []
    for src in doc_paths:
        dst = docs_dir / src.name
        shutil.copy2(str(src), str(dst))
        copied.append(str(dst))
        logger.info("Copied: %s", dst)

    # ── 5. Rebuild index
    logger.info("Building RAG index (this may take several minutes)...")

    from rag.engine import RAGEngine

    engine = RAGEngine()
    engine.invalidate_runtime_state(clear_chroma_cache=True)
    engine.clear_query_cache()

    added = engine.rebuild_index_from_paths(
        paths=copied,
        progress_callback=lambda stage, info: logger.info(
            "[%s] %s", stage.upper(), info
        ),
    )

    logger.info("Done! Indexed %d chunks into database '%s'.", added, db_name)
    logger.info("Persist dir: %s", db_dir)

    # ── 6. Extract Knowledge Graph from documents
    if config.settings.KG_EXTRACTION_ENABLED:
        logger.info("===== KG EXTRACTION =====")
        try:
            _run_kg_extraction(db_name, persist_dir)
        except Exception as e:
            logger.warning("KG extraction failed (continuing): %s", e)

    return 0


def _run_kg_extraction(db_name: str, persist_dir: Path) -> None:
    """Extract SysML KG entities and relations from RAG documents."""
    from rag.documents import load_rag_documents_from_persist_dir
    from rag.engine import SUPPORTED_RAG_EXTENSIONS
    from agent.kg_build_agent import KGBuildAgent, SectionInfo
    from web_server import _extract_sections_from_rag_doc

    persist_str = str(persist_dir)
    logger.info("Scanning for docs in: %s", persist_str)
    rag_docs = load_rag_documents_from_persist_dir(
        persist_str, SUPPORTED_RAG_EXTENSIONS,
    )
    if not rag_docs:
        logger.warning("No RAG documents found for KG extraction")
        return

    agent = KGBuildAgent(db_name=db_name)

    async def extract():
        await agent.initialize()
        for rag_doc in rag_docs:
            sections = _extract_sections_from_rag_doc(rag_doc)
            if not sections:
                logger.info("No sections in doc: %s", getattr(rag_doc, "doc_name", "?"))
                continue
            logger.info("Extracting KG from %d sections in %s", len(sections),
                        getattr(rag_doc, "doc_name", "?"))
            try:
                await agent.build_kg_for_sections(
                    getattr(rag_doc, "doc_name", db_name), sections,
                )
            except Exception as exc:
                logger.warning("KG build failed for doc '%s': %s",
                               getattr(rag_doc, "doc_name", db_name), exc)
        await agent.close()

    asyncio.run(extract())
    logger.info("KG extraction complete. Saved to %s/knowledge_graph.sysml", persist_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless RAG DB builder")
    parser.add_argument("--db", default=DEFAULT_DB_NAME, help="RAG DB name (default: %(default)s)")
    parser.add_argument("--docs", nargs="*", default=None, help="Document paths (defaults to 3 preset files)")
    args = parser.parse_args()

    if args.docs:
        doc_paths = [Path(p) for p in args.docs]
    else:
        doc_paths = DEFAULT_DOCS

    sys.exit(build_rag_db(args.db, doc_paths))
