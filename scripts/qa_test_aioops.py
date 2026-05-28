#!/usr/bin/env python3
"""Custom QA script for AIOPS KG - detailed retrieval logging."""
import asyncio, json, sys, logging, time
from pathlib import Path
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for lib in ['openai', 'httpx', 'httpcore', 'faiss']:
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("qa_test")

import config
from scripts.sysml_rag_mcp_server import sysml_load_model, sysml_retrieve, sysml_model_summary
import scripts.sysml_rag_mcp_server as mcp_mod
from agent.chatOpenAIWithReasoning import ChatOpenAIWithReasoning
from langchain_core.messages import HumanMessage, SystemMessage

import re

QUESTION = "天河高性能计算机系统的mn0节点上只有命令行可用，需要怎么做才能获取机柜R1P3上所有节点的加电信息？"

def extract_names(msg):
    candidates = []
    for m in re.findall(r'[\u4e00-\u9fff]{2,8}(?:模块|系统|节点|机柜|命令|设备|组件|结点)', msg):
        candidates.append(m)
    for m in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', msg):
        candidates.append(m)
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:15]

async def main():
    # Load KG
    mcp_mod._global_manager = None
    mcp_mod._loaded_files.clear()
    r = sysml_load_model(str(ROOT / 'database/AIOPS/knowledge_graph.sysml'))
    if not r.get('ok'):
        logger.error("Failed to load KG")
        return

    summary = sysml_model_summary()
    logger.info("KG: %d entities, %d relations", summary['total_entities'], summary['total_relations'])
    logger.info("Types: %s", summary['type_distribution'])

    # Extract candidates
    candidates = extract_names(QUESTION)
    candidates.extend(['yhst', 'smu_tranfer_cmd', 'SMU', 'CMU', 'ncid',
                        '加电', 'power', 'smu', 'cmu', 'R1P3', 'mn0', '机柜',
                        'R1P0', 'ManagementNode', 'Commands'])
    logger.info("Candidates: %s", candidates)

    # Retrieve
    context_parts = []
    retrieved = set()
    not_found = []
    t0 = time.time()

    for name in candidates:
        for k_val in [1, 2, 0]:
            try:
                r = sysml_retrieve(name, k=k_val)
                if r.get('ok'):
                    ename = r.get('entity', {}).get('name', '')
                    if ename in retrieved:
                        continue
                    retrieved.add(ename)
                    entity = r.get('entity', {})
                    graph = r.get('relationship_graph', {})

                    parts = [f"Entity: {entity.get('name')} ({entity.get('type', entity.get('class'))})"]
                    meta = entity.get('metadata', {})
                    if meta.get('description'):
                        parts.append(f"  Desc: {meta['description']}")
                    if meta.get('properties'):
                        parts.append(f"  Props: {json.dumps(meta['properties'], ensure_ascii=False)}")
                    if entity.get('aliases'):
                        parts.append(f"  Aliases: {entity.get('aliases')}")

                    members = entity.get('members', [])
                    if members:
                        lines = [f"  Members ({len(members)}):"]
                        for m in members[:40]:
                            line = f"    - {m.get('type')} {m.get('name')}"
                            if m.get('value'):
                                line += f" = {m['value']}"
                            lines.append(line)
                        parts.append("\n".join(lines))

                    total_nodes = graph.get('total_nodes', 0)
                    total_edges = graph.get('total_edges', 0)
                    if total_nodes > 1 or total_edges > 0:
                        parts.append(f"  Graph: depth={graph.get('depth')}, nodes={total_nodes}, edges={total_edges}")
                        for e in graph.get('edges', []):
                            rn = e.get('relation', {}).get('name', '?')
                            parts.append(f"    {e['from']} --[{rn}]--> {e['to']}")

                    context_parts.append(f"--- {name} (k={k_val}) ---\n" + "\n".join(parts))
                    logger.info("  ✅ %s → %s", name, ename)
                    break
            except Exception as e:
                logger.warning("  ⚠️ %s error: %s", name, e)
                break
            if k_val == 0:
                not_found.append(name)
                logger.info("  ❌ %s", name)
            break

    elapsed = time.time() - t0
    context = "\n\n".join(context_parts) if context_parts else "(none)"
    logger.info("Retrieval done in %.1fs: %d entities, %d not found, %d chars context",
                 elapsed, len(retrieved), len(not_found), len(context))

    # LLM synthesis
    prompt = f"""## SysML KG 检索结果
KG: {summary['total_entities']} entities, {summary['total_relations']} relations

{context}

## 用户问题
{QUESTION}

请根据知识图谱数据用中文回答，给出具体步骤和可执行命令。"""

    llm = ChatOpenAIWithReasoning(
        model='qwen3:8b',
        temperature=0.1,
        api_key=SecretStr(config.settings.OPENAI_API_KEY or 'ollama'),
        base_url=config.settings.OPENAI_API_URL or 'http://localhost:11434/v1',
        timeout=900,
        streaming=False,
    )

    logger.info("Calling LLM (model: qwen3:8b, timeout: 900s)...")
    t1 = time.time()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="你是系统运维QA专家。根据知识图谱数据用中文回答问题，给出具体步骤和可执行命令。"),
                HumanMessage(content=prompt),
            ]),
            timeout=900,
        )
        answer = str(response.content)
        llm_elapsed = time.time() - t1
        logger.info("LLM response in %.1fs (%d chars)", llm_elapsed, len(answer))
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)

        # Save
        out_dir = ROOT / "tmp" / "headless_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "question": QUESTION,
            "candidates": candidates,
            "retrieved": list(retrieved),
            "not_found": not_found,
            "context_chars": len(context),
            "context": context,
            "answer": answer,
            "retrieval_time_s": round(elapsed, 1),
            "llm_time_s": round(llm_elapsed, 1),
            "kg_summary": summary,
        }
        out_file = out_dir / "qa_full_trace.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Full trace saved to: %s", out_file)
        ans_file = out_dir / "qa_answer.txt"
        ans_file.write_text(answer, encoding="utf-8")
        logger.info("Answer saved to: %s", ans_file)

    except asyncio.TimeoutError:
        logger.error("LLM timeout (900s)")
    except Exception as e:
        logger.error("LLM error: %s", e)

asyncio.run(main())
