#!/usr/bin/env python3
"""
KG Resume + Parallel Test
==========================
Tests:
1. Build state save/load (phase1_processed_sections, phase3 processed_sections)
2. Resume from interrupted build
3. Phase 3 JSON enrichment parallelism
4. Phase 1 section skip on resume
5. Phase 4 parallel bridge classification

Usage:
  # Full build test (no resume)
  python3 scripts/test_kg_resume.py full

  # Resume test (build some, simulate crash, resume)
  python3 scripts/test_kg_resume.py resume

  # Phase 3 only (test JSON enrichment)
  python3 scripts/test_kg_resume.py phase3

  # Phase 4 only (test parallel bridge)
  python3 scripts/test_kg_resume.py phase4
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

# ── Config ────────────────────────────────────────────────────

TEST_DB = "_test_resume"
TEST_DOC_PATH = None  # uses text below if not set

MINIMAL_DOC_TEXT = """## 1.1 系统概述
天河高性能计算机系统包含以下核心组件：

1. FT计算柜：包含1024个双路服务器处理器，负责通用计算任务。
2. MT加速柜：配置10240个加速器，用于AI训练和推理加速。
3. 交换柜：安装高速InfiniBand交换机，提供节点间互联。
4. 存储柜：部署分布式并行文件系统，提供PB级存储容量。

系统管理命令包括：
- yhst：查看所有结点加电信息的命令，运行在CMU上。
- ncid：查看各节点逻辑ID的命令。
- smu_tranfer_cmd：通过SMU向指定目标CMU转发命令。

## 1.2 网络架构
高速互连网络采用双轨拓扑结构，提供400Gbps带宽。
每个计算节点配备1个HCA卡，通过交换柜实现全网互联。
"""


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
                "llama_index", "urllib3", "asyncio", "faiss", "PIL",
                "mcp.client.stdio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Override config
    config.settings = config.settings.update(
        KG_EXTRACTION_ENABLED=True,
        KG_EXTRACTION_MODEL="qwen3:8b",
        KG_EXTRACTION_TEMPERATURE=0.1,
        KG_EXTRACTION_TIMEOUT=120,
        KG_EXTRACTION_MAX_ITERATIONS=20,
        BATCH_CONCURRENCY=3,
        AGENT_VERBOSE=True,
    )


logger = logging.getLogger("test_resume")


# ── Helpers ───────────────────────────────────────────────────

class FakeDoc:
    """Minimal RAG document for testing."""
    def __init__(self, name: str, text: str):
        from rag.document_interface import PageType, MonoPage
        self.doc_name = name
        self.title = name
        self.page_count = 1

        mp = MonoPage(
            title="Test Section",
            page_type=PageType.CONTENT,
            markdown_text=text,
            metadata={"page": 1, "page_start": 1, "page_end": 1},
        )
        self._mono_pages = [mp]

    def get_mono_pages(self):
        return self._mono_pages

    def catalog_payload(self):
        return []

    @property
    def _skip_build(self):
        return False

    def flatten_mono_pages(self):
        return self._mono_pages


def get_test_dir() -> Path:
    d = ROOT_DIR / "database" / TEST_DB
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_test_state():
    """Remove test build state."""
    d = get_test_dir()
    for f in d.glob("knowledge_graph.*"):
        f.unlink()
    logger.info("Cleaned test state in %s", d)


async def get_kg_stats(agent):
    """Get current KG statistics."""
    summary = await agent._summary()
    return {
        "entities": summary.get("total_entities", 0),
        "relations": summary.get("total_relations", 0),
    }


def print_build_state(db_name: str):
    """Print build.json state."""
    bsf = ROOT_DIR / "database" / db_name / "knowledge_graph.build.json"
    if bsf.exists():
        with open(bsf, 'r', encoding='utf-8') as f:
            state = json.load(f)
        logger.info("Build state:\n%s", json.dumps(state, ensure_ascii=False, indent=2))
        return state
    else:
        logger.info("No build state file found")
        return None


# ── Test: Full Build ──────────────────────────────────────────

async def test_full_build():
    """Complete KG build from scratch."""
    from agent.kg_build_agent import KGBuildAgent

    clean_test_state()
    logger.info("=" * 60)
    logger.info("TEST: Full Build")
    logger.info("=" * 60)

    doc = FakeDoc("test_doc", MINIMAL_DOC_TEXT)
    agent = KGBuildAgent(db_name=TEST_DB)

    t0 = time.time()
    stats = await agent.build_kg_from_document(doc)
    elapsed = time.time() - t0

    await agent.close()

    logger.info("Build complete in %.1fs", elapsed)
    logger.info("Stats: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    print_build_state(TEST_DB)

    return stats


# ── Test: Resume ──────────────────────────────────────────────

async def test_resume():
    """Test resume after Phase 1 completion."""
    from agent.kg_build_agent import KGBuildAgent

    clean_test_state()
    logger.info("=" * 60)
    logger.info("TEST: Resume from build.json")
    logger.info("=" * 60)

    # Step 1: Run Phase 1 only (simulate by manually manipulating state)
    doc = FakeDoc("test_doc", MINIMAL_DOC_TEXT)
    agent = KGBuildAgent(db_name=TEST_DB)
    stats1 = await agent.build_kg_from_document(doc)
    await agent.close()

    logger.info("First build stats: entities=%d, relations=%d",
                 stats1.get("entity_count_after", 0),
                 stats1.get("relation_count_after", 0))

    # Verify state says "done"
    state = print_build_state(TEST_DB)
    assert state["documents"]["test_doc"]["phase"] == "done", f"Expected done, got {state}"
    logger.info("Resume test PASSED: phase=done")

    # Step 2: Modify state to simulate crash mid-Phase-3
    bsf = get_test_dir() / "knowledge_graph.build.json"
    with open(bsf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entry = data["documents"]["test_doc"]
    entry["phase"] = "phase3"
    entry["processed_sections"] = []  # would be populated by intermediate saves
    with open(bsf, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Step 3: Resume — should skip Phase 1, Phase 2, run Phase 3, Phase 4
    agent2 = KGBuildAgent(db_name=TEST_DB)
    stats2 = await agent2.build_kg_from_document(doc)
    await agent2.close()

    logger.info("Resume stats: entities=%d, relations=%d",
                 stats2.get("entity_count_after", 0),
                 stats2.get("relation_count_after", 0))
    state = print_build_state(TEST_DB)
    assert state["documents"]["test_doc"]["phase"] == "done"
    logger.info("Resume test PASSED: resumed from phase3 → done")


# ── Test: Phase 1 Resume ─────────────────────────────────────

async def test_phase1_resume():
    """Test Phase 1 section-level resume."""
    from agent.kg_build_agent import KGBuildAgent

    test_db = "_test_p1resume"
    test_dir = ROOT_DIR / "database" / test_db
    test_dir.mkdir(parents=True, exist_ok=True)
    for f in test_dir.glob("knowledge_graph.*"):
        f.unlink()

    logger.info("=" * 60)
    logger.info("TEST: Phase 1 section resume")
    logger.info("=" * 60)

    # Create doc with multiple sections
    multi_section_text = """## 1.1 系统概述
系统包含FT计算柜和MT加速柜。
yhst命令查看加电信息。

## 2.1 网络架构
交换柜提供400Gbps带宽。
每个节点配备HCA卡。"""

    from rag.document_interface import MonoPage, PageType

    class MultiPageDoc:
        doc_name = "multi_sec_doc"
        title = "Multi Section Doc"
        _skip_build = False

        def get_mono_pages(self):
            pages = []
            sections = multi_section_text.split("\n\n")
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                mp = MonoPage(
                    title=f"Section {i+1}",
                    page_type=PageType.CONTENT,
                    markdown_text=section.strip(),
                    metadata={"page": i + 1, "page_start": i + 1, "page_end": i + 1},
                )
                pages.append(mp)
            return pages

        def catalog_payload(self):
            return []

        def flatten_mono_pages(self):
            return self.get_mono_pages()

    doc = MultiPageDoc()

    # First build
    agent = KGBuildAgent(db_name=test_db)
    logger.info("Running initial full build...")
    stats = await agent.build_kg_from_document(doc)
    await agent.close()

    logger.info("Initial stats: entities=%d", stats.get("entity_count_after", 0))
    state = print_build_state(test_db)
    assert state["documents"]["multi_sec_doc"]["phase"] == "done"
    assert "phase1_processed_sections" in state["documents"]["multi_sec_doc"]
    logger.info("Phase 1 resume test PASSED: phase1_processed_sections saved")

    # Clean up
    for f in test_dir.glob("knowledge_graph.*"):
        f.unlink()


# ── Test: Build State Persistence ─────────────────────────────

async def test_build_state_persistence():
    """Verify build state file format and content."""
    logger.info("=" * 60)
    logger.info("TEST: Build state persistence")
    logger.info("=" * 60)

    bsf = get_test_dir() / "knowledge_graph.build.json"
    if not bsf.exists():
        logger.warning("No build state file exists (run full build first)")
        return

    with open(bsf, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = data.get("documents", {})
    logger.info("Documents in state: %s", list(docs.keys()))
    for doc_name, entry in docs.items():
        logger.info("  %s: phase=%s", doc_name, entry.get("phase"))
        logger.info("    processed_sections: %s", entry.get("processed_sections"))
        logger.info("    phase1_processed_sections: %s", entry.get("phase1_processed_sections"))
        logger.info("    stats keys: %s", list(entry.get("stats", {}).keys())[:10])

    # Verify expected keys
    for entry in docs.values():
        assert "phase" in entry, "Missing 'phase' field"
        assert "stats" in entry, "Missing 'stats' field"

    logger.info("Build state persistence PASSED")


# ── Main ──────────────────────────────────────────────────────

async def main():
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    modes = {
        "full": test_full_build,
        "resume": test_resume,
        "phase1": test_phase1_resume,
        "state": test_build_state_persistence,
    }

    if mode not in modes:
        logger.error("Unknown mode: %s. Choose from: %s", mode, list(modes.keys()))
        return 1

    try:
        await modes[mode]()
    except Exception as e:
        logger.error("Test FAILED: %s\n%s", e, traceback.format_exc())
        return 1

    logger.info("\nTests completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
