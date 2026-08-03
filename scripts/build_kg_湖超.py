"""Build KG for 湖超-硬件维护手册20231225.doc (uses unified agent with document tree navigation)"""
import asyncio, sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noisy OCR retry logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import config
config.settings = config.settings.update(
    KG_EXTRACTION_ENABLED=True,
    KG_EXTRACTION_MODEL="qwen3:8b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=300,
    RAG_DB_NAME="AIOPS",
    RAG_DB_NAMES=["AIOPS"],
    OCR_MODEL=None,
    AGENT_VERBOSE=False,
)

from rag.documents import load_rag_documents_from_paths
from rag.engine import SUPPORTED_RAG_EXTENSIONS
from agent.kg_build_agent import KGBuildAgent

DOC_PATH = "database/AIOPS/docs/湖超-硬件维护手册20231225.doc"
KG_PATH = Path("database/AIOPS/knowledge_graph.sysml")


async def main():
    print("=" * 60)
    print("KG Build: 湖超-硬件维护手册20231225.doc")
    print(f"Model: {config.settings.KG_EXTRACTION_MODEL}")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Load & build document tree ──
    print("\n[1/3] Loading & building document tree...")
    rag_docs = load_rag_documents_from_paths(
        [DOC_PATH], SUPPORTED_RAG_EXTENSIONS,
    )
    if not rag_docs:
        print("  ERROR: Failed to load document!")
        return

    doc = rag_docs[0]
    print(f"  Document: {doc.doc_name}")
    print(f"  Pages: {doc.page_count}")
    print(f"  Chunks: {len(getattr(doc, 'chunk_documents', []) or [])}")

    mono_pages = doc.get_mono_pages()
    content_pages = [p for p in mono_pages
                     if getattr(p, "category", "") not in ("cover", "catalogue")]
    print(f"  Content pages: {len(content_pages)}")
    t1 = time.time()
    print(f"  Time: {t1 - t0:.1f}s")

    # ── Step 2: KG extraction via unified agent ──
    est_minutes = max(1, len(content_pages) // 3)
    print(f"\n[2/3] KG extraction (unified agent, ~{len(content_pages)} pages, est ~{est_minutes}min, model: {config.settings.KG_EXTRACTION_MODEL})...")
    print("  Agent will autonomously navigate the document tree, extract entities+relations, and check off pages.")

    agent = KGBuildAgent(db_name="AIOPS")
    agent.timeout = 300

    stats = await agent.build_kg_from_document(doc)

    t2 = time.time()
    print(f"  Extraction done in {t2 - t1:.1f}s")
    print(f"  Entities: {stats.get('entity_count_before', 0)} → {stats.get('entity_count_after', 0)}")
    print(f"  Relations: {stats.get('relation_count_before', 0)} → {stats.get('relation_count_after', 0)}")

    await agent.close()

    # ── Step 3: Display results ──
    print(f"\n[3/3] Results...")
    t3 = time.time()

    if KG_PATH.exists():
        content = KG_PATH.read_text(encoding="utf-8")
        entity_lines = [l for l in content.splitlines()
                        if any(kw in l for kw in ("def ", "usage ", "part ", "attribute ",
                                                    "port ", "item ", "requirement ", "package ",
                                                    "connect ", "interface ", "allocation "))]
        print(f"\n{'=' * 60}")
        print(f"KG file: {KG_PATH}")
        print(f"Size: {len(content)} chars, ~{len(entity_lines)} entity/relation entries")
        print(f"Total time: {t3 - t0:.1f}s")
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
