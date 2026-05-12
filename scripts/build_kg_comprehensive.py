"""
Enhanced KG extraction: direct LLM call with comprehensive prompt.
Processes full document text in large chunks for thorough entity/relation extraction.
"""
import asyncio
import json
import re
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import config
config.settings = config.settings.update(
    RAG_DB_NAME="AIOPS", RAG_DB_NAMES=["AIOPS"], OCR_MODEL=None,
)

from rag.documents import load_rag_documents_from_paths
from rag.engine import SUPPORTED_RAG_EXTENSIONS
from web_server import _extract_sections_from_rag_doc


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


async def extract_comprehensive(text: str, client: OpenAI, model: str) -> list[dict]:
    """单次LLM调用提取全面的实体/关系"""
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
                except:
                    pass
    return results


async def build_kg_comprehensive(client: OpenAI, model: str):
    """全面构建知识图谱"""
    print("=" * 60)
    print(f"Comprehensive KG Build: 湖超-硬件维护手册")
    print(f"Model: {model}")
    print("=" * 60)
    t0 = time.time()

    # Load document
    print("\n[1] Loading document...")
    rag_docs = load_rag_documents_from_paths(
        ["database/AIOPS/docs/湖超-硬件维护手册20231225.doc"],
        SUPPORTED_RAG_EXTENSIONS,
    )
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

    # Chunk into 6000-char segments
    chunk_size = 6000
    chunks = []
    for i in range(0, len(all_text), chunk_size):
        chunks.append(all_text[i:i + chunk_size])
    print(f"  Chunks: {len(chunks)}")

    # Extract from each chunk
    all_results = []
    with tqdm(total=len(chunks), desc="  Extracting", unit="chunk") as bar:
        for chunk in chunks:
            results = await extract_comprehensive(chunk, client, model)
            all_results.extend(results)
            bar.set_postfix_str(f"{len(results)} items")
            bar.update(1)

    print(f"\n  Total extracted items: {len(all_results)}")

    # Categorize
    entities = [r for r in all_results if r["type"] in ("PartDef", "AttributeDef", "PortDef", "ItemDef", "RequirementDef")]
    relations = [r for r in all_results if r["type"] in ("Connection", "Allocation", "Interface")]
    
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")

    # Build SysML
    from sysml.sysml_manager import SysMLManager
    mgr = SysMLManager()
    
    # Create main document package (in memory, will be serialized)
    doc_pkg = "湖超-硬件维护手册"
    mgr.add_entity_with_metadata("Package", doc_pkg)
    
    # Group entities by chapter and create sub-packages
    # First, create all sub-packages
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

    # Add relations to the doc package (connections go at same level or in sub-packages)
    for r in relations:
        rtype = r["type"].lower()
        src = r.get("source", "")
        tgt = r.get("target", "")
        desc = r.get("description", "")
        if src and tgt:
            mgr.add_relation(rtype, src, tgt, description=desc, parent_package=doc_pkg)

    # Save
    kg_path = Path("database/AIOPS/knowledge_graph.sysml")
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


async def main():
    client = OpenAI(
        api_key=config.settings.OPENAI_API_KEY,
        base_url=config.settings.OPENAI_API_URL,
        timeout=300,
    )
    # Use qwen3:8b (only one that fits in memory)
    await build_kg_comprehensive(client, "qwen3:8b")

if __name__ == "__main__":
    asyncio.run(main())
