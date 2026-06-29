# KG Pipeline Performance Optimization

## TL;DR

> **Quick Summary**: 通过批量 MCP 操作、Phase 1 并发、倒排索引、状态保存频率降低、提示词精简和模型卸载六个实现级优化，将 AIOPS_New（57页）的 KG 构建时间从 ~45 分钟降至 ~15-18 分钟，**零精度损失**。
>
> **Deliverables**:
> - MCP 服务器新增 `sysml_batch` 批量工具（减少 75% stdio 往返）
> - 递归管线 Phase 1 根小节并发化（LLM 并行 + MCP 串行）
> - Phase 2 SQLite FTS5 倒排索引（大型文档加速 Phase 2 传播）
> - 状态保存频率从每 10 轮降至每 50 轮
> - `EXTRACTION_CANDIDATES_PROMPT` 精简（移除代码示例块）
> - 模型卸载函数接入 Phase 过渡
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 → Task 7 → Task 8 → Task 9 → FINAL

---

## Context

### Original Request
用户经过 ograg2 与 Nanite 深度对比分析后，要求在不损失提取精度的前提下，实施六个实现级优化以加速 KG 构建管线。

### Interview Summary
**Key Discussions**:
- 批量 MCP：MCP 服务器单线程同步，天然安全。服务器端 ~30 行，客户端 `_process_candidates` 需重构为分批模式
- Phase 1 并发：LLM 调用可并行（无状态 HTTP），但 MCP 必须串行（单一 stdio 子进程 + 共享 SysMLManager 状态）
- Phase 2 倒排索引：技术可行但 Phase 2 并非瓶颈（Phase 3 LLM 调用占主导）。对小型文档收益 < 1 秒，对大型文档（>1000节）才显著
- 状态保存频率：简单常量修改（10→50），崩溃恢复粒度降低但整体安全
- 提示词精简：可行但收益有限（每调用节省 ~200 tokens），需保持提取质量
- 模型卸载：`_unload_light_model()` 现有但从未被调用，`KG_KEEP_ALIVE` 配置未使用

**Research Findings**:
- MCP 调用精确计数：典型节（3实体+2关系）= 12-18 次 stdio 往返
- AIOPS_New: Phase 2 <1s, Phase 3 队列 19-68, 整体 45 分钟
- Intel: Phase 2 秒级, Phase 3 队列峰值 11651, 整体估计 32 小时
- `DocTreeNode.text` 构造后永不变 → 索引安全
- `_unload_light_model` 完整实现但死代码，`KG_KEEP_ALIVE` 配置键未在任何代码中读取

### Metis Review
> **SKIPPED** — yunwu/claude-opus-4-8 模型 API 额度耗尽，三次重试均失败。计划生成中通过加强自审弥补。

---

## Work Objectives

### Core Objective
在保持完全相同 KG 提取精度的前提下，通过六个实现级优化将 AIOPS_New KG 构建时间从 ~45 分钟降低至 ~15-18 分钟。

### Concrete Deliverables
- `scripts/sysml_rag_mcp_server.py` — 新增 `sysml_batch` 工具定义和处理函数
- `agent/kg_build_agent.py` — 重构 `_process_candidates` 为批量模式；重构 `_process_root_sections` 为并行 LLM + 串行 MCP；新增 SQLite FTS5 索引辅助函数；修改状态保存间隔；修改模型初始化添加 `model_kwargs`
- `agent/chatOpenAIWithReasoning.py` — 无需修改（已有 `model_kwargs` 支持）
- 回归测试验证 KG 质量不变

### Definition of Done
- [ ] AIOPS_New 完整构建成功，实体/关系数量与优化前一致
- [ ] 构建时间从 ~45min 降至 ≤20min
- [ ] `sysml_batch` 工具在 MCP 服务器中可用
- [ ] Phase 1 根小节 LLM 调用并发执行
- [ ] Phase 2 传播使用倒排索引（如文档 >500 节则启用）
- [ ] 状态保存频率为每 50 轮
- [ ] `model_kwargs` 正确传入 ChatOpenAIWithReasoning 构造函数
- [ ] 所有现有测试通过

### Must Have
- 六项优化的实现代码
- 批量 MCP 操作的 `sysml_batch` 工具
- Phase 1 LLM 调用并发化
- 回归测试：AIoPS_New 构建成功，KG 质量不变
- 向后兼容：旧管线（`build_kg_from_document`）不受影响

### Must NOT Have (Guardrails)
- **禁止修改提示词语义** — 只精简表达，不改变提取指令
- **禁止引入并发 MCP 调用** — MCP 操作必须保持串行
- **禁止移除任何现有的去重/验证逻辑**
- **禁止改变 BFS 级联算法或实体传播逻辑**
- **禁止修改 SysML 模型定义或序列化格式**
- **禁止添加新的外部依赖**（SQLite 标准库自带，不算新依赖）

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (项目无自动化测试框架)
- **Automated tests**: None
- **Agent-Executed QA**: ALL verification via Bash（构建脚本运行 + 日志分析） + interactive_bash（服务器启动验证）

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **API/Backend**: Use Bash (python/bun REPL) — 启动构建，检查日志输出，对比前后 KG 统计
- **MCP Server**: Use Bash — 直接调用 MCP 服务器验证 `sysml_batch` 工具可用性
- **Performance**: Use Bash — 记录构建时间，对比优化前后

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation + infrastructure, MAX PARALLEL):
├── Task 1: MCP 批量工具服务器端实现 [quick]
├── Task 2: Phase 1 根小节并发化 [quick]
├── Task 3: 状态保存频率降低 [quick]
├── Task 4: 模型卸载接入 [quick]
├── Task 5: 提示词精简 [quick]
└── Task 6: Phase 2 倒排索引基础设施 [quick]

Wave 2 (After Wave 1 — client integration, depends on Task 1):
└── Task 7: _process_candidates 批量模式重构 [deep]

Wave 3 (After Wave 2 — integration & verification):
├── Task 8: 集成回归测试 [deep]
└── Task 9: 性能基准测试 [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 7 → Task 8 → Task 9 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 6 (Wave 1)
```

### Dependency Matrix
- **1-6**: — (独立，可立即开始)
- **7**: 1 — 8, 2
- **8**: 2, 7 — 9, 3
- **9**: 8 — F1-F4, 3

### Agent Dispatch Summary
- **1**: **6** — T1-T6 → `quick`
- **2**: **1** — T7 → `deep`
- **3**: **2** — T8 → `deep`, T9 → `quick`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. MCP 批量工具 `sysml_batch` 服务器端实现

  **What to do**:
  - 在 `scripts/sysml_rag_mcp_server.py` 的 `TOOL_DEFINITIONS` 字典中新增 `"sysml_batch"` 条目
  - 新增 `sysml_batch(operations: List[Dict], continue_on_error: bool = True) -> Dict` 处理函数
  - 函数内部：遍历 `operations`，对每个 `{"tool": name, "arguments": {...}}` 调用 `_run_tool(tool, args)`
  - 返回格式：`{"ok": bool, "total": int, "succeeded": int, "failed": int, "results": [{...}]}`
  - 在 `_mcp_serve` 的 `tools/list` 响应中自动包含新工具（无需额外代码，`TOOL_DEFINITIONS` 迭代处理）
  - 测试：通过 stdin 发送 `tools/call` JSON-RPC 请求，验证批量处理返回正确 results 数组

  **Must NOT do**:
  - 不要修改 `SysMLManager` 或任何现有工具函数
  - 不要修改 `_run_tool` 的调度逻辑
  - 不要引入线程/锁（服务器保持单线程同步）
  - 不要改变现有工具的 JSON Schema

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯服务器端新增，~30 行代码，无需修改客户端
  - **Skills**: None
    - Reason: 单一文件修改，纯 Python，无需外部依赖

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6)
  - **Blocks**: Task 7 (batch-mode _process_candidates)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `scripts/sysml_rag_mcp_server.py:2120-2199` — TOOL_DEFINITIONS 结构模板（参数 JSON Schema 定义格式）
  - `scripts/sysml_rag_mcp_server.py:2495-2506` — `_run_tool()` 调度逻辑（工具名→函数→参数分发）
  - `scripts/sysml_rag_mcp_server.py:2510-2591` — `_mcp_serve()` stdio 循环（tools/list 和 tools/call 路由）
  - `scripts/sysml_rag_mcp_server.py:995-1047` — `sysml_add_entity` 现有内部批处理模式（`_expand_bracket_name`）

  **Acceptance Criteria**:
  - [ ] `TOOL_DEFINITIONS` 包含 `"sysml_batch"` 键
  - [ ] `sysml_batch()` 函数存在且可被 `_run_tool` 调度
  - [ ] `tools/list` 响应包含 `sysml_batch` 工具

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 批量搜索实体
    Tool: Bash (curl/echo to MCP server)
    Preconditions: MCP server running via `python scripts/sysml_rag_mcp_server.py serve`, manager initialized
    Steps:
      1. Send JSON-RPC tools/list → verify sysml_batch in response
      2. Send JSON-RPC tools/call with sysml_batch: {"operations":[{"tool":"sysml_list_entities","arguments":{}},{"tool":"sysml_model_summary","arguments":{}}]}
      3. Assert response.ok = true, response.succeeded = 2, response.failed = 0
      4. Assert response.results[0].result matches sysml_list_entities output format
      5. Assert response.results[1].result contains "total_entities" key
    Expected Result: Batch returns all individual tool results correctly
    Failure Indicators: response.ok = false, any tool fails unexpectedly, response structure doesn't match
    Evidence: .omo/evidence/task-1-batch-search.json

  Scenario: Failure/edge case — 部分操作失败不阻断
    Tool: Bash (echo to MCP server)
    Preconditions: MCP server running
    Steps:
      1. Send sysml_batch with: [{"tool":"sysml_list_entities","arguments":{}},{"tool":"nonexistent_tool","arguments":{}},{"tool":"sysml_model_summary","arguments":{}}]
      2. Assert response.ok = false (中间操作失败)
      3. Assert response.succeeded = 2, response.failed = 1
      4. Assert response.results[0].ok = true, response.results[1].ok = false, response.results[2].ok = true
    Expected Result: Failed tool doesn't block subsequent operations
    Evidence: .omo/evidence/task-1-batch-partial-failure.json
  ```

  **Evidence to Capture**:
  - [ ] task-1-batch-search.json — 成功批量操作响应
  - [ ] task-1-batch-partial-failure.json — 部分失败响应

  **Commit**: YES (groups with Task 7)
  - Message: `feat(mcp): add sysml_batch tool for bulk entity operations`
  - Files: `scripts/sysml_rag_mcp_server.py`

- [x] 2. Phase 1 根小节 LLM 调用并发化

  **What to do**:
  - 在 `agent/kg_build_agent.py` 中重构 `_process_root_sections()` 方法（第 2185-2286 行）
  - 将当前串行 `for sid in start_section_ids` 循环拆分为两阶段：
    - **Phase 1a**（并行 LLM）：用 `asyncio.gather` + `asyncio.Semaphore(batch_concurrency)` 并发提交所有根小节的 LLM 调用
    - **Phase 1b**（串行 MCP）：LLM 结果收集后，顺序调用 `_process_candidates` 处理 MCP 操作
  - 根实体创建保持先于 LLM 调用（第 2200-2207 行，不变）
  - `all_new_entity_names` 收集保持跨节累加
  - 错误处理：单个根小节的 LLM 失败不阻断其他小节（`return_exceptions=True`）

  **Must NOT do**:
  - 不要并行化 MCP 调用（`_process_candidates` 必须串行）
  - 不要修改 `tree_state.mark_processed` 的调用时机
  - 不要改变根实体创建逻辑
  - 不要引入新的共享状态

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 模式已存在于旧管线 `_run_phase1`（第 749-881 行），直接复用
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5, 6)
  - **Blocks**: Task 8 (integration test)
  - **Blocked By**: None

  **References**:
  - `agent/kg_build_agent.py:745-881` — 旧管线 `_run_phase1` 的正确并发模式（`asyncio.gather` + `Semaphore` + 串行 MCP）
  - `agent/kg_build_agent.py:2185-2286` — 当前 `_process_root_sections`（需重构）
  - `agent/kg_build_agent.py:784` — `asyncio.Semaphore(batch_concurrency)` 用法

  **Acceptance Criteria**:
  - [ ] `_process_root_sections` 使用 `asyncio.gather` 并发提交 LLM 调用
  - [ ] MCP 调用保持串行顺序
  - [ ] 单个根小节 LLM 失败不阻断其他小节

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 3 个根小节并发 LLM 提取
    Tool: Bash (python script)
    Preconditions: build_kg_recursive.py SELECTED_OPTION="AIOPS_New", Ollama running with qwen3:8b
    Steps:
      1. Run `timeout 120 python -c "
import asyncio, time
t0 = time.time()
# Call _process_root_sections with 3 start sections
asyncio.run(test_root_sections_parallel())
print(f'ELAPSED: {time.time()-t0:.1f}s')
" 2>&1`
      2. Assert ELAPSED < 100s (3×60s 并发应 < 2×60s)
      3. Check log for "Phase 1 done: N entities discovered" with N > 0
      4. Verify all 3 sections marked processed
    Expected Result: Elapsed time less than sum of individual LLM times
    Failure Indicators: ELAPSED > 120s (串行行为), exceptions from any section
    Evidence: .omo/evidence/task-2-parallel-phase1.txt

  Scenario: Failure/edge case — 一个根小节 LLM 超时
    Tool: Bash (python script)
    Preconditions: 模拟或使用真实文档
    Steps:
      1. Process root sections where one section triggers asyncio.TimeoutError
      2. Assert other sections complete successfully
      3. Assert all_new_entity_names contains entities from successful sections only
      4. Check log contains timeout warning for failed section
    Expected Result: Partial success — timeout doesn't cascade
    Evidence: .omo/evidence/task-2-timeout-handling.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-parallel-phase1.txt — 并发耗时日志
  - [ ] task-2-timeout-handling.txt — 超时处理日志

  **Commit**: YES
  - Message: `perf(agent): parallel Phase 1 root section LLM calls`
  - Files: `agent/kg_build_agent.py`

- [x] 3. 状态保存频率降低

  **What to do**:
  - 在 `agent/kg_build_agent.py` 第 2054 行，将 `iteration % 10 == 0` 改为 `iteration % 50 == 0`
  - 同时改 `REMINDER_MSG` 常量（如果有）中的间隔说明
  - 添加注释注明：崩溃恢复最大丢失进度从 ~10 轮变为 ~50 轮

  **Must NOT do**:
  - 不要移除 Ctrl+C 信号处理器中的最终保存逻辑
  - 不要改变 `_save_recursive_state` 的序列化内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 两字符修改（10→50），单行代码
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5, 6)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `agent/kg_build_agent.py:2054` — `if iteration % 10 == 0:` 当前间隔
  - `agent/kg_build_agent.py:2435-2487` — `_save_recursive_state` 方法

  **Acceptance Criteria**:
  - [ ] 第 2054 行 `iteration % 10` 改为 `iteration % 50`
  - [ ] 注释说明最大丢失进度

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 验证保存间隔生效
    Tool: Bash (grep)
    Preconditions: 代码已修改
    Steps:
      1. grep "iteration % 50" agent/kg_build_agent.py → found 1 match
      2. grep "iteration % 10" agent/kg_build_agent.py → found 0 matches in build_kg_recursive context
    Expected Result: Single match at correct line
    Evidence: .omo/evidence/task-3-grep-result.txt

  Scenario: Edge case — Ctrl+C 最终保存仍触发
    Tool: Bash (grep)
    Steps:
      1. grep "KeyboardInterrupt" agent/kg_build_agent.py → found in build_kg_recursive
      2. Verify the except block still calls _save_recursive_state
    Expected Result: KeyboardInterrupt handler unchanged
    Evidence: .omo/evidence/task-3-ctrl-c-save.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-grep-result.txt — grep 验证结果
  - [ ] task-3-ctrl-c-save.txt — Ctrl+C 处理器验证

  **Commit**: YES
  - Message: `perf(agent): reduce state save interval 10→50 iterations`
  - Files: `agent/kg_build_agent.py`

- [x] 4. 模型卸载接入

  **What to do**:
  - 在 `agent/kg_build_agent.py` 的 `llm` 和 `light_llm` 属性（第 562-593 行）中，向 `ChatOpenAIWithReasoning` 构造函数添加 `model_kwargs={"keep_alive": str(settings.KG_KEEP_ALIVE)}`
  - 在 `build_kg_recursive` 方法的 Phase 1 结束后、Phase 2 传播前，调用 `await self._unload_light_model()`
  - 确保 `_unload_light_model()` 在卸载前检查 `self.light_model` 是否已加载（通过 Ollama `/api/ps`）
  - 验证 Phase 3 第一个 LLM 调用能自动触发模型热加载（Ollama 冷启动延迟 2-5 秒可接受）

  **Must NOT do**:
  - 不要移除 `_unload_light_model` 的实现
  - 不要在 Phase 0 前卸载模型
  - 不要改变模型的 Ollama API base URL 提取逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 添加 `model_kwargs` 参数 + 一行调用语句
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5, 6)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `agent/kg_build_agent.py:562-593` — `llm` 和 `light_llm` 属性（需添加 model_kwargs）
  - `agent/kg_build_agent.py:1706-1738` — `_unload_light_model()` 现有实现
  - `agent/kg_build_agent.py:1917-2008` — `build_kg_recursive` Phase 0→1→2 过渡（插入卸载调用点）
  - `agent/chatOpenAIWithReasoning.py:70-79` — `_get_chat_model_kwargs` 确认 model_kwargs 透传机制

  **Acceptance Criteria**:
  - [ ] `ChatOpenAIWithReasoning` 构造函数包含 `model_kwargs={"keep_alive": ...}`
  - [ ] `_unload_light_model()` 在 Phase 1 结束后被调用
  - [ ] Phase 3 第一个 LLM 调用成功（模型自动重新加载）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — Phase 2 期间模型卸载
    Tool: Bash (python script + curl)
    Preconditions: Ollama running, qwen3:8b loaded
    Steps:
      1. Start build_kg_recursive for AIOPS_New
      2. During Phase 2 (watch log for "Phase 2: queue size="), run: curl localhost:11434/api/ps | python -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin)['models']]; print('qwen3:8b loaded:', 'qwen3:8b' in models)"
      3. Assert model is NOT loaded during Phase 2 (keep_alive=0 took effect)
      4. During Phase 3, verify model reloads and extraction continues
    Expected Result: Model unloaded during Phase 2, auto-reloads for Phase 3
    Failure Indicators: Model stays loaded during Phase 2, Phase 3 fails with connection error
    Evidence: .omo/evidence/task-4-model-unload.log

  Scenario: Failure/edge case — 模型未加载时继续工作
    Tool: Bash (curl)
    Steps:
      1. Run _unload_light_model when model already unloaded
      2. Check log: "Light model 'qwen3:8b' already unloaded"
      3. Phase 3 LLM call succeeds (Ollama auto-loads)
    Expected Result: Graceful no-op when already unloaded
    Evidence: .omo/evidence/task-4-already-unloaded.log
  ```

  **Evidence to Capture**:
  - [ ] task-4-model-unload.log — 模型卸载和重载日志
  - [ ] task-4-already-unloaded.log — 重复卸载无错误

  **Commit**: YES
  - Message: `fix(agent): wire model_kwargs and _unload_light_model into pipeline`
  - Files: `agent/kg_build_agent.py`

- [x] 5. 提示词精简

  **What to do**:
  - 在 `agent/kg_build_agent.py` 中精简 `EXTRACTION_CANDIDATES_PROMPT`（第 91-150 行）
  - 保留：实体类型列表（6 种）、关系类型列表（20 种）
  - 移除：示例代码块（第 129-148 行的输入输出示例）
  - 将移除的示例移至注释中（`# 原示例: ...` 保留作为文档参考）
  - 同时检查 `ENRICHMENT_JSON_PROMPT`（第 155-177 行）是否有可精简的等效块

  **Must NOT do**:
  - 不要删除实体/关系类型列表
  - 不要改变提示词的指令语义
  - 不要改变 JSON 输出格式要求

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯文本删除，保留关键信息
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 6)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `agent/kg_build_agent.py:91-150` — `EXTRACTION_CANDIDATES_PROMPT` 当前内容
  - `agent/kg_build_agent.py:155-177` — `ENRICHMENT_JSON_PROMPT` 参考

  **Acceptance Criteria**:
  - [ ] `EXTRACTION_CANDIDATES_PROMPT` 字符数从 1693 降至 ≤1400
  - [ ] 实体类型列表（6 种）和关系类型列表（20 种）完整保留
  - [ ] 移除的示例在注释中可查

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 提示词精简后提取质量不变
    Tool: Bash (python script)
    Preconditions: 优化前后各运行一次单节提取
    Steps:
      1. 用精简前提示词提取同一页面的实体/关系，记录结果
      2. 用精简后提示词提取同一页面
      3. 比较两组的实体名称集合和关系数量
      4. Assert 实体名称集合相同或超集（精简版不应遗漏）
    Expected Result: 精简后提取结果一致
    Failure Indicators: 精简后缺失实体或关系
    Evidence: .omo/evidence/task-5-prompt-quality.txt

  Scenario: Edge case — 字符数验证
    Tool: Bash (wc)
    Steps:
      1. grep "EXTRACTION_CANDIDATES_PROMPT" agent/kg_build_agent.py -A 60 | wc -c
      2. Assert count ≤ 1400
    Expected Result: 精简版提示词在预算内
    Evidence: .omo/evidence/task-5-char-count.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-prompt-quality.txt — 提取质量对比
  - [ ] task-5-char-count.txt — 字符数验证

  **Commit**: YES
  - Message: `perf(prompt): prune EXTRACTION_CANDIDATES_PROMPT code examples`
  - Files: `agent/kg_build_agent.py`

- [x] 6. Phase 2 倒排索引基础设施

  **What to do**:
  - 在 `agent/kg_build_agent.py` 中新增辅助方法 `_build_inverted_index(sections: Dict)` → 返回 `sqlite3.Connection` (memory)
  - 使用 SQLite FTS5：`CREATE VIRTUAL TABLE section_text_fts USING fts5(section_id UNINDEXED, text, tokenize='unicode61')`
  - 在 Phase 0/1 边界（`_build_section_map` 之后）调用 `_build_inverted_index`
  - 在 `KGBuildAgent` 存储索引连接引用：`self._fts_index: Optional[sqlite3.Connection]`
  - 新增辅助方法 `_fts_find_sections(entity_name: str, aliases: List[str])` → `Set[str]`（返回匹配的 section_ids）
  - **仅在节数 > 500 时启用索引**（通过 `len(sections) > 500` 判断），否则回退到当前子串扫描
  - 索引查询格式：`SELECT DISTINCT section_id FROM section_text_fts WHERE text MATCH ?`（带 `*` 前缀匹配）

  **Must NOT do**:
  - 不要强制小型文档使用索引（<500 节时直接跳过）
  - 不要打破当前子串匹配的语义（注意 `in` vs FTS5 MATCH 的差异）
  - 不要引入外部依赖（`sqlite3` 是 Python 标准库）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 新增辅助方法，纯 Python 标准库，不与现有逻辑耦合
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 5)
  - **Blocks**: None（可选的性能优化，不阻塞任何任务）
  - **Blocked By**: None

  **References**:
  - `agent/kg_build_agent.py:2288-2340` — `_propagate_entities` 当前子串匹配逻辑（需新增索引分支）
  - `agent/kg_build_agent.py:2106-2114` — `_build_section_map`（索引构建点）
  - `agent/kg_build_agent.py:1914-1916` — sections 构建后的调用时机
  - Python docs: `sqlite3` FTS5 `CREATE VIRTUAL TABLE ... USING fts5(...)` 语法

  **Acceptance Criteria**:
  - [ ] `_build_inverted_index` 方法存在，返回 sqlite3 Connection
  - [ ] `_fts_find_sections` 方法存在，返回 Set[str]
  - [ ] 节数 ≤500 时跳过索引构建（回退到子串扫描）
  - [ ] 索引构建不引入异常（try/except 保护）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 大型文档索引查询
    Tool: Bash (python REPL)
    Preconditions: 使用 AIOPS_New 文档（141 节，应启用索引以验证功能）
    Steps:
      1. 构建 sections 字典并调用 _build_inverted_index
      2. 调用 _fts_find_sections("FT计算柜", []) → 返回包含此文本的小节
      3. Assert 返回的 Set 非空（FT计算柜在文档中存在）
      4. 对比手动子串扫描结果，验证匹配小节一致
    Expected Result: FTS 查询返回与子串扫描相同的小节
    Failure Indicators: FTS 遗漏子串扫描找到的小节，或返回多余假阳性
    Evidence: .omo/evidence/task-6-fts-match.txt

  Scenario: Edge case — 小型文档跳过索引
    Tool: Bash (python REPL)
    Steps:
      1. 创建 ≤500 节的 sections 字典
      2. 验证 _build_inverted_index 返回 None 或标记跳过
      3. 验证 _fts_find_sections 回退到子串扫描
    Expected Result: 小型文档不构建索引，回退到原逻辑
    Evidence: .omo/evidence/task-6-small-doc-skip.txt
  ```

  **Evidence to Capture**:
  - [ ] task-6-fts-match.txt — FTS 查询结果
  - [ ] task-6-small-doc-skip.txt — 小型文档跳过验证

  **Commit**: YES
  - Message: `perf(agent): add SQLite FTS5 inverted index for Phase 2 propagation`
  - Files: `agent/kg_build_agent.py`

- [x] 7. `_process_candidates` 批量模式重构

  **What to do**:
  - 在 `agent/kg_build_agent.py` 中重构 `_process_candidates`（第 1096-1262 行），通过 `sysml_batch` 将多次 MCP 调用批量化
  - 新建 `_process_candidates_batch()` 方法，保留旧 `_process_candidates()` 作为回退（通过配置开关控制）
  - 批量策略分为三批：
    - **批 1**：所有实体的 `sysml_search_entity`（并行搜索）
    - **批 2**：分析搜索结果 → `sysml_add_entity`（新实体）+ `sysml_update_entity`（已存在）+ `sysml_add_alias`（别名）
    - **批 3**：所有关系的 `sysml_add_relation`（含端点验证的 search 和 add）
  - `entity_qn_map` 在批 1→2 之间本地维护，批 2 完成后更新
  - 确保 `source_sections` 参数正确传递到每个操作
  - 在 `_process_root_sections` 和 `_process_queued_section` 中切换调用 `_process_candidates_batch`

  **Must NOT do**:
  - 不要移除旧 `_process_candidates`（保留作为回退）
  - 不要改变实体/关系创建的语义和参数
  - 不要并行化 MCP 调用（批 1/2/3 本身是单个 MCP 调用，但三批之间串行）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 核心重构，涉及多批 MCP 调用协调、错误处理、状态管理，复杂度高
  - **Skills**: None
    - Reason: 纯 Python 重构，无外部依赖或特殊工具需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential, depends on Task 1)
  - **Blocks**: Task 8 (integration test)
  - **Blocked By**: Task 1 (`sysml_batch` 服务器端)

  **References**:
  - `agent/kg_build_agent.py:1096-1262` — `_process_candidates` 当前实现（实体搜索→创建/更新→关系创建循环）
  - `agent/kg_build_agent.py:1119-1127` — 单个 `sysml_search_entity` 调用模式
  - `agent/kg_build_agent.py:1149-1163` — 单个 `sysml_add_entity` 调用模式
  - `agent/kg_build_agent.py:1174-1262` — 单条关系创建流程（端点验证 + 创建）
  - `agent/kg_build_agent.py:2342-2416` — `_process_queued_section`（调用 _process_candidates 的入口之一）
  - `agent/kg_build_agent.py:2185-2286` — `_process_root_sections`（调用 _process_candidates 的入口之二）

  **Acceptance Criteria**:
  - [ ] `_process_candidates_batch` 方法存在，实现三批批量策略
  - [ ] 旧 `_process_candidates` 保留不变
  - [ ] `_process_root_sections` 和 `_process_queued_section` 调用 batch 版本
  - [ ] 创建的实体/关系数量和内容与旧方法一致（同一份输入）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 批量创建 3 实体 + 2 关系
    Tool: Bash (python script)
    Preconditions: MCP server running, manager initialized, sysml_batch tool available
    Steps:
      1. Prepare test candidates: 3 entities (2 new, 1 duplicate) + 2 relations
      2. Call _process_candidates_batch(test_candidates, test_relations, "test_doc", 1, "Test Section")
      3. Assert returned dict: created_entities >= 2, created_relations >= 2
      4. Verify via sysml_list_entities: 3 entities exist in model
      5. Verify via sysml_get_connections: 2 relations exist
    Expected Result: Same entities/relations as old _process_candidates would create
    Failure Indicators: Missing entities, wrong entity_qn_map entries, relation endpoint resolution failures
    Evidence: .omo/evidence/task-7-batch-create.json

  Scenario: Failure/edge case — 全部新实体无重复
    Tool: Bash (python script)
    Preconditions: Fresh model (no existing entities)
    Steps:
      1. 5 new entities + 3 relations, all endpoints within the 5 entities
      2. Call _process_candidates_batch
      3. Assert batch 1 returns total_matches=0 for all searches
      4. Assert batch 2 creates all 5 entities
      5. Assert entity_qn_map contains all 5 entries
      6. Assert batch 3 creates all 3 relations
    Expected Result: All batch operations succeed on clean model
    Evidence: .omo/evidence/task-7-clean-model.json

  Scenario: Regression — 批量模式 vs 旧模式输出一致
    Tool: Bash (python script)
    Preconditions: Same input data, fresh models for each run
    Steps:
      1. Run _process_candidates on input data → record entity names + relation count
      2. Reset model (new manager)
      3. Run _process_candidates_batch on same input data → record entity names + relation count
      4. Assert entity name sets identical
      5. Assert relation counts within ±1
    Expected Result: Batch mode produces equivalent results
    Evidence: .omo/evidence/task-7-regression.json
  ```

  **Evidence to Capture**:
  - [ ] task-7-batch-create.json — 批量创建结果
  - [ ] task-7-clean-model.json — 全新模型批量创建
  - [ ] task-7-regression.json — 回归对比结果

  **Commit**: YES
  - Message: `refactor(agent): batch-mode _process_candidates via sysml_batch`
  - Files: `agent/kg_build_agent.py`

- [~] 8. 集成回归测试 ⚠️ 首次超时(35min) — 管线正常但需要更长时间或分批策略

  **What to do**:
  - 运行 AIOPS_New 完整构建（`python scripts/build_kg_recursive.py`，SELECTED_OPTION="AIOPS_New"）
  - 对比优化前后的 KG 输出：
    - 实体总数（PartDef、AttributeDef、PortDef、ItemDef、RequirementDef、CommandDef）
    - 关系总数（所有 20 种关系类型）
    - `.sysml` 文件大小
    - 连通分量数量
  - 允许 ±5% 偏差（LLM 的非确定性输出）
  - 验证无崩溃、无超时、无 MCP 错误

  **Must NOT do**:
  - 不要跳过任何 KG 构建阶段
  - 不要使用缓存的构建结果

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 完整构建约需 15-20 分钟，需要端到端验证和日志分析
  - **Skills**: None
    - Reason: 纯 Python 脚本执行和日志解析

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (depends on all implementation tasks)
  - **Blocks**: Task 9 (performance benchmark)
  - **Blocked By**: Task 2, Task 7

  **References**:
  - `scripts/build_kg_recursive.py` — 构建入口
  - `tmp/kg_builds/AIOPS_New.log` — 优化前日志（基线数据）
  - `database/AIOPS_New/knowledge_graph.sysml` — 输出 KG 文件

  **Acceptance Criteria**:
  - [ ] 构建完成无崩溃（exit code 0）
  - [ ] 实体总数在基线的 95-105% 范围内
  - [ ] 关系总数在基线的 95-105% 范围内
  - [ ] 连通分量数 ≤ 优化前

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 完整 AIOPS_New 构建
    Tool: Bash (timeout + python)
    Preconditions: database/AIOPS_New/ 为空或已清理，Ollama running qwen3:8b
    Steps:
      1. cd /media/ritanlisa/Weights/Nanite-Tokenizers
      2. python -c "
with open('scripts/build_kg_recursive.py','r') as f: c=f.read()
c=c.replace('SELECTED_OPTION = \"Intel\"','SELECTED_OPTION = \"AIOPS_New\"')
with open('scripts/build_kg_recursive.py','w') as f: f.write(c)
"
      3. timeout 1800 python scripts/build_kg_recursive.py 2>&1 | tee /tmp/optimized_build.log
      4. grep "BUILD COMPLETE" /tmp/optimized_build.log → found
      5. grep "Entities:" /tmp/optimized_build.log → 实体数 ≥ 425
      6. grep "Relations:" /tmp/optimized_build.log → 关系数 ≥ 1100
      7. ls database/AIOPS_New/knowledge_graph.sysml → exists, size > 100KB
    Expected Result: 构建成功，实体≥425，关系≥1100
    Failure Indicators: 崩溃、超时、实体/关系数显著低于基线
    Evidence: .omo/evidence/task-8-full-build.log

  Scenario: Regression — 构建无 MCP 错误
    Tool: Bash (grep)
    Steps:
      1. grep "MCP FAILED" /tmp/optimized_build.log → count = 0
      2. grep "Phase 3 timeout" /tmp/optimized_build.log → count = 0
      3. grep "Traceback" /tmp/optimized_build.log → count = 0
    Expected Result: 零 MCP 失败、零超时、零异常
    Evidence: .omo/evidence/task-8-no-errors.txt
  ```

  **Evidence to Capture**:
  - [ ] task-8-full-build.log — 完整构建日志
  - [ ] task-8-no-errors.txt — 错误统计

  **Commit**: YES (if code changes needed for test)
  - Message: `test(kg): integration regression test for optimized pipeline`
  - Files: 无（纯验证任务）

- [~] 9. 性能基准测试 — 阻塞于 Task 8 未完成

  **What to do**:
  - 提取优化前后的构建时间（来自日志的 `BUILD COMPLETE in Xs` 行）
  - 分别记录各阶段耗时：
    - Phase 0 (root identification)
    - Phase 1 (root sections extraction)
    - Phase 2 (propagation, all rounds)
    - Phase 3 (queue processing, all rounds)
    - Dedup + Aggregation
  - 计算加速比：`before_time / after_time`
  - 记录 MCP 调用次数（来自日志的 `MCP +ENTITY/+RELATION` 模式）
  - 生成 JSON 格式的性能报告保存到 `.omo/evidence/task-9-benchmark.json`

  **Must NOT do**:
  - 不要跳过任何阶段的计时
  - 不要使用估算值替代实测值

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯日志分析和基准计算，无代码修改
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (depends on Task 8)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - `tmp/kg_builds/AIOPS_New.log` — 优化前构建日志（基线耗时）
  - Task 8 输出日志 — 优化后构建日志

  **Acceptance Criteria**:
  - [ ] 优化后总构建时间 < 20 分钟
  - [ ] Phase 1 耗时减少 ≥50%（并发 LLM）
  - [ ] MCP 调用次数减少 ≥50%（批量操作）
  - [ ] 性能报告 JSON 包含所有阶段的 before/after 对比

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — 构建时间 < 20 分钟
    Tool: Bash (grep + python)
    Preconditions: Task 8 完成的构建日志
    Steps:
      1. grep "BUILD COMPLETE in" /tmp/optimized_build.log → extract seconds
      2. python -c "
import re, json
with open('/tmp/optimized_build.log') as f:
    log = f.read()
m = re.search(r'BUILD COMPLETE in (\d+)s', log)
t = int(m.group(1)) if m else 9999
print(f'BUILD TIME: {t}s ({t/60:.1f}min)')
print('PASS' if t < 1200 else 'FAIL: >20min')
" 
      3. Assert PASS
    Expected Result: 构建时间 < 1200 秒
    Failure Indicators: 构建时间 > 1200 秒
    Evidence: .omo/evidence/task-9-build-time.txt

  Scenario: Benchmark — 阶段级加速比
    Tool: Bash (python comparison)
    Steps:
      1. Extract Phase 1-3 timings from optimized build log
      2. Compare with baseline log timings
      3. Generate JSON: {"phase1_before": X, "phase1_after": Y, "speedup": X/Y, ...}
      4. Save to .omo/evidence/task-9-benchmark.json
    Expected Result: Phase 1 speedup ≥2x, MCP operations speedup ≥3x
    Evidence: .omo/evidence/task-9-benchmark.json
  ```

  **Evidence to Capture**:
  - [ ] task-9-build-time.txt — 构建时间验证
  - [ ] task-9-benchmark.json — 阶段级性能报告

  **Commit**: NO (纯验证任务，无代码修改)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [~] F1. **Plan Compliance Audit** — 阻塞于 Task 8 未完成
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [~] F2. **Code Quality Review** — 阻塞于 Task 8 未完成
  Run build/lint/test commands. Review all changed files for: type suppression, empty catches, debug logging, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [~] F3. **Real Manual QA** — 阻塞于 Task 8 未完成
  Run AIOPS_New full build with optimizations. Compare entity/relation counts with pre-optimization baseline. Verify KG .sysml file is identical in structure. Execute ALL QA scenarios from every task.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [~] F4. **Scope Fidelity Check** — 阻塞于 Task 8 未完成
  For each task: read "What to do", read actual diff. Verify 1:1 — everything spec'd was built, nothing beyond spec built. Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `feat(mcp): add sysml_batch tool for bulk entity operations` — scripts/sysml_rag_mcp_server.py
- **2**: `perf(agent): parallel Phase 1 root section LLM calls` — agent/kg_build_agent.py
- **3**: `perf(agent): reduce state save interval 10→50 iterations` — agent/kg_build_agent.py
- **4**: `fix(agent): wire model_kwargs and _unload_light_model into pipeline` — agent/kg_build_agent.py
- **5**: `perf(prompt): prune EXTRACTION_CANDIDATES_PROMPT code examples` — agent/kg_build_agent.py
- **6**: `perf(agent): add SQLite FTS5 inverted index for Phase 2 propagation` — agent/kg_build_agent.py
- **7**: `refactor(agent): batch-mode _process_candidates via sysml_batch` — agent/kg_build_agent.py
- **8-9**: `test(kg): integration regression and performance benchmark` — agent/kg_build_agent.py

---

## Success Criteria

### Verification Commands
```bash
# 验证 sysml_batch 工具可用
python scripts/sysml_rag_mcp_server.py serve &
sleep 2
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | head -1 | \
  python -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read())))" | \
  python -c "import sys,json; tools=[t['name'] for t in json.loads(sys.stdin.read())['tools']]; print('sysml_batch' in tools)"

# 运行优化后构建并计时
time python scripts/build_kg_recursive.py

# 验证 KG 质量
python -c "
import json
with open('database/AIOPS_New/knowledge_graph.sysml') as f:
    content = f.read()
print(f'Entities: {content.count(\"part def\") + content.count(\"attribute def\") + content.count(\"port def\")}')
print(f'Relations: {content.count(\"connect\")}')
"
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tasks completed
- [ ] AOIPS_New build time ≤ 20 minutes
- [ ] KG quality unchanged (entity/relation counts within 5% of baseline)
- [ ] All evidence files captured in `.omo/evidence/`
