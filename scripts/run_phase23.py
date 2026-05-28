#!/usr/bin/env python3
"""Run Phase 2 (dedup) + Phase 3 (enrichment) + QA on existing AIOPS_New KG."""
import asyncio, json, sys, logging, time
from pathlib import Path
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for lib in ['openai','httpx','httpcore','mcp.client.stdio','sse_starlette']:
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("phase23")

import config
config.settings = config.settings.update(
    KG_EXTRACTION_MODEL="qwen3-vl:32b",
    KG_EXTRACTION_TEMPERATURE=0.1,
    KG_EXTRACTION_TIMEOUT=600,
    KG_EXTRACTION_MAX_ITERATIONS=100,
    AGENT_VERBOSE=False,
    AGENT_INVOKE_TIMEOUT=1200,
    LLM_REQUEST_TIMEOUT=600,
)

async def heartbeat(duration=0):
    """Show . every 3s to indicate activity."""
    import sys
    while True:
        sys.stderr.write(".")
        sys.stderr.flush()
        await asyncio.sleep(3)

async def main():
    t0 = time.time()
    db_name = "AIOPS_New"

    from mcp_client.mcp_session import create_sysml_mcp_session
    session = create_sysml_mcp_session()
    await session.initialize()

    kg_path = ROOT / "database" / db_name / "knowledge_graph.sysml"
    logger.info("Loading KG: %s", kg_path)
    r = json.loads(await session.call_tool("sysml_load_model", {"file_path": str(kg_path)}))
    if not r.get("ok"):
        logger.error("Failed to load KG")
        return
    logger.info("Loaded OK")

    s0 = json.loads(await session.call_tool("sysml_model_summary", {}))
    logger.info("Initial: %d entities, %d relations", s0["total_entities"], s0["total_relations"])

    # ═══════════════════════════════════════════════════════
    # Phase 2: Dedup
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 50)
    logger.info("PHASE 2: Cross-section entity deduplication")
    logger.info("=" * 50)
    t2 = time.time()
    hb2 = asyncio.create_task(heartbeat())

    suggest = json.loads(await session.call_tool("sysml_suggest_merge", {"threshold": 0.5}))
    sc = len(suggest.get("suggestions", []))
    logger.info("Merge suggestions: %d pairs", sc)

    merged_count = 0
    if sc > 0:
        # Take top 30 most confident
        items = sorted(suggest["suggestions"], key=lambda x: -x["confidence"])[:40]
        logger.info("Processing top %d suggestions", len(items))
        
        for i, item in enumerate(items):
            a_qn = item["entity_a"]["qualified_name"]
            b_qn = item["entity_b"]["qualified_name"]
            a_name = item["entity_a"]["name"]
            b_name = item["entity_b"]["name"]
            conf = item["confidence"]
            reasons = ", ".join(item.get("reasons", []))
            
            logger.info("  [%d/%d] %s ←→ %s (%.2f: %s)", i+1, len(items), a_name, b_name, conf, reasons)
            
            # Direct merge (skip LLM review for speed)
            try:
                result = json.loads(await session.call_tool(
                    "sysml_merge_entities",
                    {"source": a_qn, "target": b_qn},
                ))
                if result.get("ok"):
                    merged_count += 1
                    logger.info("    ✅ Merged %s → %s", a_qn, b_qn)
            except Exception as e:
                logger.warning("    ❌ Merge error: %s", e)

    hb2.cancel()
    try: await hb2
    except asyncio.CancelledError: pass

    t2e = time.time() - t2
    s2 = json.loads(await session.call_tool("sysml_model_summary", {}))
    logger.info("PHASE 2 complete: %.0fs | merged: %d pairs | entities: %d→%d",
                 t2e, merged_count, s0["total_entities"], s2["total_entities"])

    # Save checkpoint
    await session.call_tool("sysml_save_model", {"file_path": str(kg_path)})
    logger.info("Phase 2 KG saved")

    # ═══════════════════════════════════════════════════════
    # Phase 3: Enrichment
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 50)
    logger.info("PHASE 3: Entity enrichment (aliases, properties, relations)")
    logger.info("=" * 50)
    t3 = time.time()
    hb3 = asyncio.create_task(heartbeat())

    entity_list = json.loads(await session.call_tool("sysml_list_entities", {"include_details": False}))
    entity_names = [e.get("name","") for e in entity_list.get("entities",[])][:60]
    logger.info("Enrichment targets: %d entities (top 60 of %d)", len(entity_names), len(entity_list.get("entities",[])))

    from mcp_client.tool_wrapper import build_mcp_tools
    enrich_tools = await build_mcp_tools(session, prefix="mcp", tool_filter=[
        "sysml_search_entity","sysml_add_alias","sysml_add_relation",
        "sysml_update_entity","sysml_list_entities","sysml_model_summary",
    ])

    from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAIWithReasoning(
        model="qwen3-vl:32b", temperature=0.2,
        api_key=SecretStr(config.settings.OPENAI_API_KEY),
        base_url=config.settings.OPENAI_API_URL,
        timeout=600, streaming=True,
    )

    agent = create_agent(
        model=llm, tools=enrich_tools,
        system_prompt="""你是SysML v2知识图谱专家。审查实体列表，补充别名、属性、关系。

## 可用工具
- mcp__sysml_search_entity: 搜索实体
- mcp__sysml_add_alias: 添加别名 (qualified_name, alias)
- mcp__sysml_add_relation: 创建关系 (relation_type, source, target, name, parent_package, role_source, role_target)
- mcp__sysml_update_entity: 更新实体 (需要: update_properties, new_name, supertypes, type_refs)

## 关系类型
- connection: 物理连接/数据流
- interface: 接口实现  
- allocation: 包含/分配

## 策略
1. 用mcp__sysml_search_entity了解各实体
2. 添加中英文别名
3. 识别实体间关系并创建
4. 关系参数: parent_package填"系统架构"等分类, role_source/role_target填"节点/设备"等角色""",
        debug=False, name="kg_enricher",
    )

    prompt = f"审查以下{len(entity_names)}个实体，补充别名和关系:\n{json.dumps(entity_names, ensure_ascii=False, indent=2)}"

    try:
        await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 60},
            ),
            timeout=3600,
        )
    except asyncio.TimeoutError:
        logger.warning("Enrichment timeout (3600s)")
    except Exception as e:
        logger.warning("Enrichment error: %s", e)

    hb3.cancel()
    try: await hb3
    except asyncio.CancelledError: pass

    t3e = time.time() - t3
    s3 = json.loads(await session.call_tool("sysml_model_summary", {}))
    logger.info("PHASE 3 complete: %.0fs | entities: %d | relations: %d→%d",
                 t3e, s3["total_entities"], s2["total_relations"], s3["total_relations"])

    # Save final
    await session.call_tool("sysml_save_model", {"file_path": str(kg_path)})

    # ═══════════════════════════════════════════════════════
    # QA Retrieval
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 50)
    logger.info("QA: Retrieval test")
    logger.info("=" * 50)

    import re
    question = "天河高性能计算机系统的mn0节点上只有命令行可用，需要怎么做才能获取机柜R1P3上所有节点的加电信息？"
    candidates = []
    for m in re.findall(r'[\u4e00-\u9fff]{2,8}(?:模块|系统|节点|机柜|命令|设备|组件|结点)', question):
        candidates.append(m)
    for m in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', question):
        candidates.append(m)
    candidates.extend(['yhst','smu_tranfer_cmd','SMU','CMU','ncid','加电','R1P3','mn0','机柜','管理结点','Commands'])

    from scripts.sysml_rag_mcp_server import sysml_retrieve
    retrieved = set()
    not_found = []
    for name in candidates:
        for k in [1, 2, 0]:
            r = sysml_retrieve(name, k=k)
            if r.get("ok"):
                ename = r["entity"]["name"]
                if ename not in retrieved:
                    retrieved.add(ename)
                    logger.info("  ✅ %s → %s (%s)", name, ename, r.get("matched_by","?"))
                break
        else:
            not_found.append(name)
            logger.info("  ❌ %s", name)

    total = time.time() - t0
    logger.info("=" * 50)
    logger.info("ALL PHASES COMPLETE in %.0fs (%.1fm)", total, total/60)
    logger.info("Initial:       %d entities, %d relations", s0["total_entities"], s0["total_relations"])
    logger.info("Phase 2 (dedup): %.0fs | merged %d pairs → %d entities, %d relations", 
                 t2e, merged_count, s2["total_entities"], s2["total_relations"])
    logger.info("Phase 3 (enrich):%.0fs | %d entities, %d relations", 
                 t3e, s3["total_entities"], s3["total_relations"])
    logger.info("QA: %d retrieved, %d not found", len(retrieved), len(not_found))

    await session.close()

asyncio.run(main())
