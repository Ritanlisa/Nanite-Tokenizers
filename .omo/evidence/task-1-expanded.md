# Task 1 — Nanite-Tokenizers Performance Evidence

> Extracted from source code: `agent/kg_build_agent.py`, `config.py`, `AGENTS.md`

---

## 1. Chunk Strategy

**Per-section semantic boundary, no hard character limit like other RAG systems.**

- Phase 1a groups pages by parent section (paragraph/chapter grouping) — `agent/kg_build_agent.py:782-795` (`_run_phase1`)
- However, within a single page, if text > 4000 chars, a sliding window is applied: `window=3000`, `stride=2500` (500-char overlap) → `agent/kg_build_agent.py:948-963` (`_extract_page_candidates`)
- Overlapping chunks are merged post-extraction: same-name entities keep the richest description, duplicates by (source, target, type) key — `agent/kg_build_agent.py:1012-1037`
- Phase 3 enrichment batches entities into groups of `MAX_ENTITIES_PER_BATCH = 15` per LLM call — `agent/kg_build_agent.py:1337-1359`

**Result:** Extraction respects document section structure (semantic boundaries) but uses a 4000-char sliding window as a safeguard for extremely long single pages.

---

## 2. Context Window Handling

**Per-section isolated sessions via prompt construction, not persistent chat threads.**

- Phase 1a: Each section/page gets a fresh `SystemMessage` + `HumanMessage` pair — `agent/kg_build_agent.py:979-982`. No message history accumulation.
- Phase 3: Each entity batch is a fresh call — `agent/kg_build_agent.py:1371-1374`. No conversation state between calls.
- MCP state (SysML model) persists independently via `MCPSession` stdio — `agent/kg_build_agent.py:600-601`. LLM sessions are stateless; entity knowledge accumulates in MCP-side memory.
- The system prompt (`UNIFIED_EXTRACTION_SYSTEM_PROMPT`, ~1.5KB) and extraction prompt (`EXTRACTION_CANDIDATES_PROMPT`, ~3KB) are the primary context overhead per call.

**Practical section size limit:** `KG_EXTRACTION_MAX_TOKENS: 4096` — `config.py:144`, used as `max_tokens` for both `light_llm` and `llm` — `agent/kg_build_agent.py:565-566,582-583`. This is the **output** token limit for the LLM response; the input context is gated by the 4000-char sliding window (roughly ~1000 tokens for Chinese technical text, ~2000 tokens for English).

---

## 3. VRAM / Memory Management

**Explicit, multi-layered VRAM management:**

| Mechanism | Detail | Source |
|-----------|--------|--------|
| `KG_KEEP_ALIVE: "30s"` | Light model auto-unloads after 30s of idle | `config.py:142` |
| `keep_alive=0` explicit unload | After Phase 2, sends Ollama `/api/generate` with `keep_alive=0` to force-unload the light model | `agent/kg_build_agent.py:1722-1754` (`_unload_light_model`) |
| Two-model architecture | Phase 1a/3: `light_llm` (`qwen3:8b`). Phase 3 enrichment uses `llm` (strong model, also `qwen3:8b` by default). Only one model loaded at a time in VRAM. | `config.py:140-141`, `agent/kg_build_agent.py:539-540` |
| Phase 2 unload checkpoint | Light model unloaded before Phase 3 to free VRAM for strong model — `AGENTS.md:28` | `AGENTS.md:28-29` |
| `num_predict: 1` in keep_alive=0 call | Minimizes token generation during unload to conserve VRAM/power | `agent/kg_build_agent.py:1744` |

**VRAM flow:**
```
Start    → Light model loaded
Phase 1a → Light model active (5 concurrent calls)
Phase 1b → Light model idle (MCP operations)
Phase 2  → No LLM needed (pure MCP dedup, <1s)
         → _unload_light_model() called → Light model unloaded
Phase 3  → Strong model loaded for enrichment
```

No Ollama VRAM check (`/api/ps`) in current code that gates model loading, but the `_unload_light_model` method does query `/api/ps` to confirm unload status — `agent/kg_build_agent.py:1730-1736`.

---

## 4. Token Cost Model

**Cost scales with number of entities found, not raw document text length.**

- Phase 1a: Light model (`qwen3:8b`, local Ollama) — **zero API cost**. Runs in parallel on all sections.
- Phase 1b: No LLM calls — pure MCP search+create operations.
- Phase 2: No LLM — pure algorithmic dedup (merge suggestions from MCP).
- Phase 3: Light model again (JSON enrichment). Calls per section = `ceil(entity_count / 15)` — `agent/kg_build_agent.py:1359`.
- Phase 4 (optional): Light model bridge classification, limited to 20 candidates — `agent/kg_build_agent.py:1689`.

**Cost formula (approximate):**
```
LLM calls = NumSections (Phase 1a) + sum(ceil(section_entities / 15)) (Phase 3) + bridges (Phase 4, ≤20)
```
All via local Ollama → **zero API cost**. If an external API were used, cost would scale with the number of entities extracted, not the document page count.

---

## 5. Throughput / Parallelism

| Parameter | Value | Source |
|-----------|-------|--------|
| `BATCH_CONCURRENCY` | 5 (default) | `config.py:128` |
| Semaphore gating | `asyncio.Semaphore(BATCH_CONCURRENCY)` | `agent/kg_build_agent.py:800, 1335, 1632` |
| Phase 1a parallelism | `asyncio.gather(*tasks)` — all section LLM calls submitted simultaneously, gated by Semaphore(5) | `agent/kg_build_agent.py:821-822` |
| Phase 1b | Sequential MCP processing (search+create per entity — no parallelism) | `agent/kg_build_agent.py:1125-1278` (`_process_candidates`) |
| Phase 3 parallelism | Same Semaphore(5) gating for enrichment calls | `agent/kg_build_agent.py:1334-1335` |
| Phase 4 parallelism | Semaphore(5) for bridge classification calls via `httpx.AsyncClient` (raw HTTP, not LangChain) | `agent/kg_build_agent.py:1631-1690` |
| Light LLM timeout | 180s (Phase 1a), 600s (Phase 3) | `agent/kg_build_agent.py:983-984, 1376` |
| Light LLM streaming | Phase 1a: `streaming=False`, Phase 3: inherited `light_llm` (streaming=False) | `agent/kg_build_agent.py:590` |
| Strong LLM streaming | `streaming=True` for LangChain agent | `agent/kg_build_agent.py:573` |
| MCP calls | Sequential within each section (no parallelism for MCP) | `agent/kg_build_agent.py:1125-1278` |

**Async I/O pattern:** All LLM calls are submitted first (Phase 1a: `asyncio.gather`), then MCP processing proceeds sequentially — `AGENTS.md:26`.

---

## 6. Practical Maximum Document Size

**Single-document, section-limited by LLM context (4096 output tokens), not page count.**

- No hard page count limit. A 500-page document would work: Phase 1a would process all sections in parallel batches of 5.
- The practical bottleneck is **per-section text length**: the 4000-char sliding window (`agent/kg_build_agent.py:948-963`) limits input per LLM call. For Chinese technical text (~2 chars/token), this is ~2000 tokens input. For dense English text (~4 chars/token), ~1000 tokens.
- `KG_EXTRACTION_MAX_TOKENS: 4096` limits the **output** (extracted entities/relations JSON array).
- `KG_EXTRACTION_MAX_ITERATIONS: 10` (default) — `config.py:146`. The old LangChain agent interface uses this as `recursion_limit` — `agent/kg_build_agent.py:1516`; the new `build_kg_from_document` pipeline does NOT use this limit (it processes all sections deterministically).
- `KG_EXTRACTION_TIMEOUT: 120` (default) — `config.py:145`. Used as LLM HTTP timeout; Phase 1a uses per-call timeout of 180s — `agent/kg_build_agent.py:984`.

**Maximum tested:** No explicit limit found in source. The recursive pipeline (`build_kg_recursive`) has `max_iterations = 500` — `agent/kg_build_agent.py:1978`, suggesting documents with 500+ section-entity pairs are supported.

---

## Summary Table

| Dimension | Value | Mechanism |
|-----------|-------|-----------|
| Chunk boundary | Per-section (parent_id grouping) | `_run_phase1` groups pages by `parent_id` |
| Page-level safeguard | 4000-char sliding window, 500-char overlap | `_extract_page_candidates` |
| Context window per call | Fresh SystemMessage+HumanMessage, no history accumulation | Stateless prompt construction |
| Output token limit | 4096 (`KG_EXTRACTION_MAX_TOKENS`) | `config.py:144` |
| VRAM auto-unload | 30s idle → auto-unload (`KG_KEEP_ALIVE`) | `config.py:142` |
| VRAM explicit unload | `keep_alive=0` after Phase 2 | `_unload_light_model` |
| Model architecture | Phase 1a/3: light, Phase 3 strong | Two-model: `qwen3:8b` (both by default) |
| Concurrency | 5 (`BATCH_CONCURRENCY`) | `asyncio.Semaphore(5)` |
| Parallel phases | Phase 1a, Phase 3, Phase 4 (LLM calls) | `asyncio.gather` |
| Sequential phases | Phase 1b (MCP ops per-section), Phase 2 (dedup) | Deterministic ordering |
| Cost driver | Number of entities extracted (not page count) | Zero-cost with local Ollama |
| MCP state persistence | `MCPSession` stdio, independent of LLM sessions | In-memory SysML model |
| Iteration cap (recursive) | 500 | `build_kg_recursive:1978` |
| Default LLM timeout | 120s (config), 180s (Phase 1a per-call), 600s (Phase 3) | Multiple levels |
