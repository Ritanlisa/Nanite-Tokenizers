#!/usr/bin/env python3
"""
断点续传脚本 —— 从已有 KG 继续 Phase 3 富化 + Phase 4 聚合。
跳过 Phase 1 (提取) 和 Phase 2 (去重)，直接运行后续阶段。
"""
from __future__ import annotations
import asyncio, logging, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import config

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] resume: %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("resume")

DB_NAME = config.settings.RAG_DB_NAME or "AIOPS_New"
KG_FILE = Path(config.settings.PERSIST_DIR) / DB_NAME / "knowledge_graph.sysml"

async def main():
    from agent.kg_build_agent import KGBuildAgent

    if not KG_FILE.exists():
        logger.error("No existing KG at %s — run headless_kg_test.py first", KG_FILE)
        return

    agent = KGBuildAgent(db_name=DB_NAME)
    await agent.initialize()

    summary = await agent._summary()
    e = summary.get("total_entities", 0)
    r = summary.get("total_relations", 0)
    logger.info("Resuming from KG: %d entities, %d relations", e, r)

    if e == 0:
        logger.error("KG is empty — run headless_kg_test.py first")
        await agent.close()
        return

    doc_name = "resume"  # placeholder

    t0 = time.time()

    # Phase 3: enrichment
    logger.info("Phase 3: enrichment...")
    await agent._enrich_entities(doc_name)
    logger.info("Phase 3 done in %.0fs", time.time() - t0)

    # Phase 4: aggregation  
    t4 = time.time()
    logger.info("Phase 4: graph aggregation...")
    await agent._aggregate_graph(doc_name)
    logger.info("Phase 4 done in %.0fs", time.time() - t4)

    # Save
    await agent._save_knowledge_graph()
    await agent.close()

    summary = await agent._summary()
    logger.info("Complete: %d entities, %d relations, %.0fs total",
                 summary.get("total_entities", 0),
                 summary.get("total_relations", 0),
                 time.time() - t0)

if __name__ == "__main__":
    asyncio.run(main())
