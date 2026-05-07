"""Build KG for 湖超-硬件维护手册20231225.doc (with tqdm progress bars + OCR disabled)"""
import asyncio, json, os, sys, time, logging
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noisy OCR retry logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import config
config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL="qwen3:8b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=300,    # Longer timeout per section
    RAG_DB_NAME="AIOPS",
    RAG_DB_NAMES=["AIOPS"],
    OCR_MODEL=None,
    AGENT_VERBOSE=False,
)

from rag.documents import load_rag_documents_from_paths
from rag.engine import SUPPORTED_RAG_EXTENSIONS
from agent.kg_build_agent import KGBuildAgent, SectionInfo
from web_server import _extract_sections_from_rag_doc

DOC_PATH = "database/AIOPS/docs/湖超-硬件维护手册20231225.doc"
KG_PATH = Path("database/AIOPS/knowledge_graph.sysml")

async def main():
    print("=" * 60)
    print("KG Build: 湖超-硬件维护手册20231225.doc")
    print(f"Model: {config.settings.KG_EXTRACTION_MODEL}")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Load & build document tree ──
    print("\n[1/4] Loading & building document tree...")
    with tqdm(total=1, desc="  Loading doc", unit="doc") as pbar:
        rag_docs = load_rag_documents_from_paths(
            [DOC_PATH], SUPPORTED_RAG_EXTENSIONS,
            progress_callback=lambda stage, payload: pbar.set_postfix_str(f"{stage}") or None,
        )
        pbar.update(1)

    if not rag_docs:
        print("  ERROR: Failed to load document!")
        return
    doc = rag_docs[0]
    print(f"  Document: {doc.doc_name}")
    print(f"  Pages: {doc.page_count}")
    print(f"  Chunks: {len(getattr(doc, 'chunk_documents', []) or [])}")
    t1 = time.time()
    print(f"  Time: {t1 - t0:.1f}s")

    # ── Step 2: Extract sections ──
    print("\n[2/4] Extracting sections from document tree...")
    sections = _extract_sections_from_rag_doc(doc)
    print(f"  Sections (de-dup, >100c, skip cover/TOC): {len(sections)}")
    for i, s in enumerate(sections[:5]):
        print(f"    [{i}] p{s.page} {s.title[:40]} ({len(s.text)}c)")
    if len(sections) > 5:
        print(f"    ... and {len(sections) - 5} more")
    print(f"  Total text: {sum(len(s.text) for s in sections)} chars")

    if not sections:
        print("  WARNING: No content sections found")
        return

    # ── Step 3: KG extraction via LLM ──
    est_time = len(sections) * 90  # ~90s per section
    print(f"\n[3/4] KG extraction ({len(sections)} sections, est ~{est_time//60}min, model: {config.settings.KG_EXTRACTION_MODEL})...")
    agent = KGBuildAgent(db_name="AIOPS")
    agent.timeout = 300

    # Progress bar per section
    with tqdm(total=len(sections), desc="  Phase 1: Entities", unit="sec", position=0) as bar:
        async def extract_entities(section):
            try:
                await agent._extract_entities_from_section(section, doc.doc_name)
            except Exception as e:
                bar.set_postfix_str(f"err: {str(e)[:40]}")
            bar.update(1)

        await agent.initialize()
        for sec in sections:
            await extract_entities(sec)

    t2 = time.time()
    # Save checkpoint after Phase 1
    await agent._save_knowledge_graph()
    print(f"  Entity extraction done in {t2 - t1:.1f}s (KG saved)")
    summary = await agent._summary()
    print(f"  Entities: {summary.get('total_entities', 0)}")

    # Cross-section dedup
    print("  Phase 1.5: Cross-section deduplication...")
    try:
        await agent._deduplicate_entities()
    except Exception as e:
        print(f"  Dedup skipped ({e})")

    # Phase 2: Relations
    print(f"  Phase 2: Relations...")
    with tqdm(total=len(sections), desc="  Phase 2: Relations", unit="sec", position=0) as bar:
        async def extract_relations(section):
            try:
                await agent._extract_relations_from_section(section, doc.doc_name)
            except Exception as e:
                bar.set_postfix_str(f"err: {str(e)[:30]}")
            bar.update(1)

        for sec in sections:
            await extract_relations(sec)

    t3 = time.time()
    print(f"  Relation extraction done in {t3 - t2:.1f}s")

    # ── Step 4: Save & display results ──
    print("\n[4/4] Saving results...")
    await agent._save_knowledge_graph()
    await agent.close()

    t4 = time.time()

    # Display KG
    if KG_PATH.exists():
        content = KG_PATH.read_text(encoding="utf-8")
        entities_raw = content.count("def") + content.count("usage") + content.count("part")
        print(f"\n{'=' * 60}")
        print(f"KG file: {KG_PATH}")
        print(f"Size: {len(content)} chars, ~{entities_raw} entity/relation entries")
        print(f"Total time: {t4 - t0:.1f}s")
        print(f"{'=' * 60}")
        print()
        print(content)
        print()
        print(f"{'=' * 60}")
        print(f"  Done! KG exported to {KG_PATH}")
        print(f"{'=' * 60}")
    else:
        print(f"  WARNING: KG file not found at {KG_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
