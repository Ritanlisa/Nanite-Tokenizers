#!/usr/bin/env python3
"""Continue enrichment + QA on existing KG."""
import asyncio, json, sys, logging, time
from pathlib import Path
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for lib in ['openai','httpx','httpcore','mcp.client.stdio','sse_starlette']:
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("enrich")

import config

DB = "AIOPS_New"

async def heartbeat():
    import sys as _sys
    while True:
        _sys.stderr.write(".")
        _sys.stderr.flush()
        await asyncio.sleep(5)

async def main():
    t0 = time.time()
    
    from mcp_client.mcp_session import create_sysml_mcp_session
    session = create_sysml_mcp_session()
    await session.initialize()
    
    kg_path = ROOT / "database" / DB / "knowledge_graph.sysml"
    r = json.loads(await session.call_tool("sysml_load_model", {"file_path": str(kg_path)}))
    logger.info("Loaded KG: %s", r.get("ok"))
    
    s0 = json.loads(await session.call_tool("sysml_model_summary", {}))
    logger.info("Start: %d entities, %d relations", s0["total_entities"], s0["total_relations"])
    
    # Phase 3: Enrichment
    logger.info("=" * 50)
    logger.info("PHASE 3: Enrichment (all entities)")
    logger.info("=" * 50)
    t3 = time.time()
    hb = asyncio.create_task(heartbeat())
    
    el = json.loads(await session.call_tool("sysml_list_entities", {"include_details": False}))
    names = [e['name'] for e in el['entities']]
    logger.info("Enrichment targets: %d entities", len(names))
    
    from mcp_client.tool_wrapper import build_mcp_tools
    tools = await build_mcp_tools(session, prefix="mcp", tool_filter=[
        "sysml_search_entity","sysml_add_alias","sysml_add_relation",
        "sysml_update_entity","sysml_list_entities","sysml_get_entity","sysml_model_summary",
    ])
    
    from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from agent.kg_build_agent import ENRICHMENT_PROMPT
    
    llm = ChatOpenAIWithReasoning(
        model="gemma4:31b", temperature=0.2,
        api_key=SecretStr(config.settings.OPENAI_API_KEY),
        base_url=config.settings.OPENAI_API_URL,
        timeout=600, streaming=True,
    )
    
    agent = create_agent(
        model=llm, tools=tools,
        system_prompt=ENRICHMENT_PROMPT,
        debug=False, name="kg_enricher",
    )
    
    # Process in batches of 15 to avoid context overflow
    batch_size = 15
    for bi in range(0, len(names), batch_size):
        batch = names[bi:bi+batch_size]
        prompt = f"审查以下{batch_size}个实体，补充别名和关系:\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
        
        try:
            await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": 40},
                ),
                timeout=600,
            )
            logger.info("  Batch %d/%d complete (%d entities)", 
                        bi//batch_size + 1, (len(names)+batch_size-1)//batch_size, len(batch))
        except asyncio.TimeoutError:
            logger.warning("  Batch %d timeout", bi//batch_size + 1)
        except Exception as e:
            logger.warning("  Batch %d error: %s", bi//batch_size + 1, e)
    
    hb.cancel()
    try: await hb
    except asyncio.CancelledError: pass
    
    t3e = time.time() - t3
    s3 = json.loads(await session.call_tool("sysml_model_summary", {}))
    logger.info("PHASE 3 complete: %.0fs | entities: %d | relations: %d->%d",
                 t3e, s3["total_entities"], s0["total_relations"], s3["total_relations"])
    
    await session.call_tool("sysml_save_model", {"file_path": str(kg_path)})
    
    # QA
    logger.info("=" * 50)
    logger.info("QA: Retrieval test")
    logger.info("=" * 50)
    
    # QA — need to load KG in main process (not just MCP subprocess)
    from scripts.sysml_rag_mcp_server import sysml_retrieve, sysml_load_model
    import scripts.sysml_rag_mcp_server as mcp_mod
    mcp_mod._global_manager = None
    mcp_mod._loaded_files.clear()
    sysml_load_model(str(kg_path))
    logger.info("KG loaded in main process for QA")
    import re
    
    question = "天河高性能计算机系统的mn0节点上只有命令行可用，需要怎么做才能获取机柜R1P3上所有节点的加电信息？"
    cands = list(set(
        re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[\\u4e00-\u9fff]{2,8}(?:模块|系统|节点|机柜|命令|设备|组件|结点)', question)
    ))
    cands.extend(['yhst','smu_tranfer_cmd','SMU','CMU','ncid','加电','R1P3','mn0','机柜','管理结点','Commands','Nvidia','A100'])
    cands = list(dict.fromkeys(cands))
    
    ret = set()
    nf = []
    for name in cands:
        for k in [1,2,0]:
            r = sysml_retrieve(name, k=k)
            if r.get("ok"):
                en = r["entity"]["name"]
                if en not in ret:
                    ret.add(en)
                    logger.info("  ✅ %s → %s (%s)", name, en, r.get("entity",{}).get("type","?"))
                break
        else:
            nf.append(name)
            logger.info("  ❌ %s", name)
    
    total = time.time() - t0
    logger.info("=" * 50)
    logger.info("ALL DONE in %.0fs (%.1fm)", total, total/60)
    logger.info("Phase 3: %.0fs", t3e)
    logger.info("Entities: %d→%d  Relations: %d→%d", 
                 s0["total_entities"], s3["total_entities"],
                 s0["total_relations"], s3["total_relations"])
    logger.info("QA: %d retrieved, %d not found", len(ret), len(nf))
    logger.info("Retrieved: %s", list(ret)[:10])
    
    await session.close()

asyncio.run(main())
