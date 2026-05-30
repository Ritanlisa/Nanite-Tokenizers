#!/usr/bin/env python3
"""
快速递归KG管线验证 (不跑全流程)
===============================
测试: Phase 0/1/2/3 各阶段基本逻辑

Usage:
  uv run python scripts/test_recursive_kg.py

预期: ~5-8分钟 (gemma4:31b × 2-3 calls)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

for name in ["agent.kg_build_agent"]:
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    l.propagate = True

for lib in ["openai", "httpx", "httpcore", "chromadb", "sentence_transformers",
            "llama_index", "urllib3", "asyncio", "faiss", "PIL",
            "mcp.client.stdio"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger("kg_test")

# ═══════════════════════════════════════════════════════════════
# Minimal test doc — using text + DocumentTreeState builder
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
    TEXT = MINIMAL_DOC_TEXT

    def __init__(self):
        self.doc_name = "天河测试文档"
        self.title = "天河高性能计算机系统简介"

    def get_mono_pages(self):
        from rag.document_interface import MonoPage, PageType
        sections = {}
        current_section = "概述"
        for line in self.TEXT.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
                continue
            if not line:
                continue
            sections.setdefault(current_section, []).append(line)

        result = []
        page_num = 1
        for sec_title, lines in sections.items():
            text = " ".join(lines)
            mp = MonoPage(
                metadata={"page": page_num, "section_title": sec_title},
                markdown_text=text,
                title=sec_title,
                page_type=PageType.CONTENT,
            )
            result.append(mp)
            page_num += 1
        return result

    @property
    def page_count(self):
        return len(self.get_mono_pages())


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL="qwen3:8b",
    KG_LIGHT_MODEL="qwen3:8b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=300,
    KG_EXTRACTION_MAX_ITERATIONS=50,
    KG_EXTRACTION_MAX_TOKENS=4096,
    BATCH_CONCURRENCY=1,
    KG_KEEP_ALIVE="600s",
)


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

async def main():
    from agent.kg_build_agent import KGBuildAgent, DocumentTreeState

    doc = _FakeDoc()
    pages = doc.get_mono_pages()
    logger.info("Test document: %d mono pages", len(pages))
    for p in pages:
        logger.info("  p%d: %s (%d chars)",
                     p.metadata.get("page", 0), p.title, len(p.markdown_text or ""))

    agent = KGBuildAgent(
        db_name="_test_recursive",
        model="qwen3:8b",
        light_model="qwen3:8b",
        persist_dir=str(ROOT_DIR / "tmp" / "_test_recursive"),
    )
    agent.timeout = 600
    agent.max_iterations = 50

    try:
        await agent.initialize()

        # Build tree from fake doc (uses _build_tree from DocumentTreeState)
        tree_state = DocumentTreeState(doc)
        logger.info("Tree: %d leaf pages", tree_state.total_pages)

        # Build section map using the same logic as _build_section_map
        sections: Dict[str, List] = {}
        for nid in tree_state._all_page_ids:
            if nid not in tree_state.nodes:
                continue
            node = tree_state.nodes[nid]
            parent_id = node.parent_id or "__root__"
            sections.setdefault(parent_id, []).append(node)
        logger.info("Section groups: %d", len(sections))
        for pid, nodes in sections.items():
            logger.info("  %s: %d pages (%s)", pid, len(nodes),
                         [n.title[:30] for n in nodes])

        # ── Test 1: Phase 0 - Root identification ──
        logger.info("=" * 60)
        logger.info("TEST 1: Phase 0 - Root Identification")
        logger.info("=" * 60)

        root, starts = await agent._identify_root(doc, tree_state, sections, "test_doc")
        logger.info("Root entity: %s", json.dumps(root, ensure_ascii=False))
        logger.info("Start sections: %s", starts)
        assert root.get("name"), "Root entity must have name"
        assert starts, "Must have start sections"
        logger.info("TEST 1 PASSED")

        # ── Test 2: Phase 1 - Root section extraction ──
        logger.info("=" * 60)
        logger.info("TEST 2: Phase 1 - Root Section Extraction")
        logger.info("=" * 60)

        new_names = await agent._process_root_sections(
            tree_state, sections, root, starts, "test_doc"
        )
        logger.info("New entities from root section: %d: %s", len(new_names), new_names)
        assert len(new_names) > 0, f"Must find at least one entity, got: {new_names}"
        logger.info("TEST 2 PASSED")

        # ── Test 3: Phase 2 - Entity propagation ──
        logger.info("=" * 60)
        logger.info("TEST 3: Phase 2 - Entity Propagation")
        logger.info("=" * 60)

        build_queue = deque()
        processed_pairs = set()
        processed_sections = set(starts)

        build_queue = await agent._propagate_entities(
            tree_state, sections, new_names,
            processed_sections, processed_pairs, build_queue, "test_doc"
        )
        logger.info("Build queue after propagation: %d items", len(build_queue))
        for item in list(build_queue)[:10]:
            logger.info("  Queue: entity=%s section=%s", item[0], item[1])
        logger.info("TEST 3 PASSED")

        # ── Test 4: Phase 3 - Process 1-2 queue items ──
        processed_count = 0
        while build_queue and processed_count < 2:
            entity_name, section_id = build_queue.popleft()
            pair = (entity_name, section_id)
            if pair in processed_pairs or section_id in processed_sections:
                continue

            logger.info("=" * 60)
            logger.info("TEST 4.%d: Phase 3 - Process (%s, %s)",
                         processed_count + 1, entity_name, section_id)
            logger.info("=" * 60)

            section_nodes = sections.get(section_id, [])
            if not section_nodes:
                processed_sections.add(section_id)
                continue

            new_ents = await agent._process_queued_section(
                section_nodes, entity_name, section_id, "test_doc"
            )
            logger.info("  New entities from cascade: %s", new_ents)
            processed_sections.add(section_id)
            processed_pairs.add(pair)
            processed_count += 1

        logger.info("TEST 4 PASSED")

        # ── Final Summary ──
        summary_raw = await agent._mcp_session.call_tool("sysml_model_summary", {})
        summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED")
        logger.info("  Entities: %d", summary.get("total_entities", 0))
        logger.info("  Relations: %d", summary.get("total_relations", 0))
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error("TEST FAILED: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        await agent.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
