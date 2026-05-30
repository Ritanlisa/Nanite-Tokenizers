#!/usr/bin/env python3
"""
快速递归KG管线验证 (不跑全流程)
===============================
测试:
1. Phase 0: 根实体识别 (gemma4:31b)
2. Phase 1: 根小节实体提取
3. Phase 2: 实体传播 (文本搜索+队列构建)
4. Phase 3: 小节处理 (处理1-2个队列项)

Usage:
  uv run python scripts/test_recursive_kg.py

预期运行时间: 1-2分钟
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

for name in ["agent.kg_build_agent", "kg_test"]:
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    l.propagate = True

for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
            "llama_index", "urllib3", "asyncio", "faiss", "PIL",
            "mcp.client.stdio"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger("kg_test")

# ═══════════════════════════════════════════════════════════════
# Minimal Test Document
# ═══════════════════════════════════════════════════════════════

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

mn0节点只有命令行可用，通过SMU转发命令到CMU来执行运维操作。
例如：smu_tranfer_cmd r1.p03a.m yhst 获取R1P3a机柜的加电信息。

## 1.3 运维命令

yhst命令运行在CMU上，用于查看所有结点加电信息。
ncid命令用于查看各节点逻辑ID。
smu_tranfer_cmd用于通过SMU向CMU转发命令。

R1P3机柜由4个机柜组成：r1.p03a.m、r1.p03b.m、r1.p03c.m、r1.p03d.m。
"""


class _FakeDoc:
    def __init__(self):
        self.doc_name = "天河测试文档"
        self.title = "天河高性能计算机系统简介"
        self._skip_build = True
        self._mono_pages = []
        self._build_mono_pages()

    def _build_mono_pages(self):
        from rag.document_interface import MonoPage, PageType
        lines = MINIMAL_DOC_TEXT.strip().split("\n")
        current_section = ""
        page_num = 1
        for line in lines:
            if line.startswith("## "):
                current_section = line[3:].strip()
                continue
            if line.strip():
                meta = {"page": page_num, "section_title": current_section}
                mp = MonoPage(
                    metadata=meta,
                    markdown_text=line.strip(),
                    category=PageType.CONTENT,
                )
                mp.page_num = page_num
                self._mono_pages.append(mp)
                page_num += 1

    def get_mono_pages(self):
        return self._mono_pages

    @property
    def page_count(self):
        return len(self._mono_pages)


# ═══════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════

config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL="gemma4:31b",
    KG_LIGHT_MODEL="qwen3:8b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=180,
    KG_EXTRACTION_MAX_ITERATIONS=200,
    BATCH_CONCURRENCY=1,
    KG_KEEP_ALIVE="600s",
    OCR_MODEL=None,
    OCR_API_URL=None,
)


async def test_phase0_root_identification(agent, doc, sections):
    """Phase 0: 根实体识别"""
    logger.info("=" * 60)
    logger.info("TEST: Phase 0 - Root Identification")
    logger.info("=" * 60)

    from agent.kg_build_agent import DocumentTreeState
    tree = DocumentTreeState(doc)
    logger.info("Tree structure:\n%s", tree.get_tree_structure()[:1000])

    root, starts = await agent._identify_root(doc, tree, sections, "test_doc")
    logger.info("Root entity: %s", json.dumps(root, ensure_ascii=False))
    logger.info("Start sections: %s", starts)

    assert root.get("name"), "Root entity must have a name"
    assert starts, "Must have at least one start section"
    logger.info("Phase 0: PASSED")


async def test_phase1_root_extraction(agent, doc, sections, root, starts):
    """Phase 1: 根小节实体提取"""
    logger.info("=" * 60)
    logger.info("TEST: Phase 1 - Root Section Extraction")
    logger.info("=" * 60)

    from agent.kg_build_agent import DocumentTreeState
    tree = DocumentTreeState(doc)

    new_names = await agent._process_root_sections(
        tree, sections, root, starts, "test_doc"
    )
    logger.info("New entities discovered: %s", new_names)

    assert len(new_names) > 0, "Must find at least one entity"
    logger.info("Phase 1: PASSED (%d entities)", len(new_names))


async def test_phase2_propagation(agent, doc, sections):
    """Phase 2: 实体传播"""
    logger.info("=" * 60)
    logger.info("TEST: Phase 2 - Entity Propagation")
    logger.info("=" * 60)

    from agent.kg_build_agent import DocumentTreeState
    from collections import deque
    tree = DocumentTreeState(doc)

    # Get existing entity names from the KG
    entities_raw = await agent._mcp_session.call_tool(
        "sysml_list_entities", {"include_details": False}
    )
    entities_data = json.loads(entities_raw) if isinstance(entities_raw, str) else entities_raw
    entity_names = []
    if isinstance(entities_data, list):
        entity_names = [e.get("name", "") for e in entities_data if e.get("name")]
    elif isinstance(entities_data, dict):
        for v in entities_data.values():
            if isinstance(v, list):
                for e in v:
                    if isinstance(e, dict) and e.get("name"):
                        entity_names.append(e["name"])

    logger.info("Existing entities: %s", entity_names[:10])

    build_queue = deque()
    processed_pairs = set()
    processed_sections = set()

    build_queue = await agent._propagate_entities(
        tree, sections, entity_names,
        processed_sections, processed_pairs, build_queue, "test_doc"
    )

    logger.info("Build queue after propagation: %d items", len(build_queue))
    for item in list(build_queue)[:10]:
        logger.info("  Queue: %s → %s", item[0], item[1])

    logger.info("Phase 2: PASSED")


async def test_phase3_cascading(agent, doc, sections):
    """Phase 3: 处理1-2个队列项"""
    logger.info("=" * 60)
    logger.info("TEST: Phase 3 - Cascading Section Processing")
    logger.info("=" * 60)

    from agent.kg_build_agent import DocumentTreeState
    from collections import deque
    tree = DocumentTreeState(doc)

    # Get entities again
    entities_raw = await agent._mcp_session.call_tool(
        "sysml_list_entities", {"include_details": False}
    )
    entities_data = json.loads(entities_raw) if isinstance(entities_raw, str) else entities_raw
    entity_names = []
    if isinstance(entities_data, list):
        entity_names = [e.get("name", "") for e in entities_data if e.get("name")]
    elif isinstance(entities_data, dict):
        for v in entities_data.values():
            if isinstance(v, list):
                for e in v:
                    if isinstance(e, dict) and e.get("name"):
                        entity_names.append(e["name"])

    build_queue = deque()
    processed_pairs = set()
    processed_sections = set()

    build_queue = await agent._propagate_entities(
        tree, sections, entity_names,
        processed_sections, processed_pairs, build_queue, "test_doc"
    )

    processed_count = 0
    while build_queue and processed_count < 2:
        entity_name, section_id = build_queue.popleft()
        logger.info("Processing queue item: (%s, %s)", entity_name, section_id)

        section_nodes = sections.get(section_id, [])
        if not section_nodes:
            processed_sections.add(section_id)
            continue

        new_ents = await agent._process_queued_section(
            section_nodes, entity_name, section_id, "test_doc"
        )
        logger.info("  New entities: %s", new_ents)
        processed_sections.add(section_id)
        processed_pairs.add((entity_name, section_id))
        processed_count += 1

        for nd in section_nodes:
            tree.mark_processed(nd.node_id)

    logger.info("Phase 3: PASSED (processed %d queue items)", processed_count)


async def main():
    from agent.kg_build_agent import KGBuildAgent, DocumentTreeState

    doc = _FakeDoc()
    logger.info("Test document: %d pages", doc.page_count)

    agent = KGBuildAgent(
        db_name="_test_recursive",
        model="gemma4:31b",
        light_model="qwen3:8b",
        persist_dir=str(ROOT_DIR / "tmp" / "_test_recursive"),
    )
    agent.timeout = 180
    agent.max_iterations = 200

    try:
        await agent.initialize()

        # Build section map
        tree = DocumentTreeState(doc)
        sections = agent._build_section_map(tree)
        logger.info("Sections: %d", len(sections))
        for sid, nodes in sections.items():
            titles = [n.title[:30] for n in nodes]
            logger.info("  %s: %d pages (%s)", sid, len(nodes), titles)

        # Run tests
        await test_phase0_root_identification(agent, doc, sections)

        root, starts = await agent._identify_root(doc, tree, sections, "test_doc")
        await test_phase1_root_extraction(agent, doc, sections, root, starts)

        await test_phase2_propagation(agent, doc, sections)
        await test_phase3_cascading(agent, doc, sections)

        # Final summary
        summary_raw = await agent._mcp_session.call_tool("sysml_model_summary", {})
        summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED")
        logger.info("KG Summary: entities=%d, relations=%d",
                     summary.get("total_entities", 0),
                     summary.get("total_relations", 0))
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error("TEST FAILED: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        await agent.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
