# KG Build Granularity — 元架构引导生成

## TL;DR

> **Quick Summary**: 在 KG 构建管线中插入"元架构"阶段——用户用自然语言描述粒度，LLM 根据文档结构+内容采样确定 Schema（节点种类、根节点、关系模式），后续 Phase 按元架构规则填充。
>
> **Deliverables**:
> - `agent/kg_granularity.py` — 新模块：元架构确定 Agent + Schema 管理器
> - MCP 工具 `sysml_meta_schema` — 元架构的创建/查询/更新
> - `scripts/build_kg_recursive.py` — 新增 `--granularity` CLI 参数
> - `agent/kg_build_agent.py` — Phase -1 (meta-architecture) + 动态 Prompt 生成
>
> **Estimated Effort**: Large (新模块 + 管线修改 + Prompt 重构)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 → T3 → T4 → T5 → T7 → T8 → FINAL

---

## Context

### Original Request
用户希望在 KG 构建前插入粒度控制机制：自然语言描述 → LLM 确定"元架构"（节点种类、根节点、关系模式）→ 按架构填充式生成。这与现有自底向上的 BFS 级联形成互补——元架构提供"自上而下的约束"。

### Design Decisions
**从之前的访谈中推断，以下采用默认假设（可覆盖）**：

| 问题 | 默认方案 | 理由 |
|------|---------|------|
| 元架构粒度 | **A2: 模式级** — 类型列表 + 结构模式 + 约束 | 类型级太粗（和现有 Prompt 没区别），Schema 级太细（实现爆炸） |
| 输入方式 | **B1+B2: CLI + Config** — `--granularity "只提取顶层架构"` | 符合现有 `SELECTED_OPTION` 配置模式 |
| 采样策略 | **C2: LLM 自主选择** — LLM 看 TOC 后决定读哪些小节 | 最智能，LLM 有 langchain Agent 的导航能力 |
| 融合方式 | **D2: 新 Phase -1** — 在现有 Phase 0 之前插入 | 不破坏现有管线，叠加而非替换 |
| 文档规模 | **E3: 通用** — 支持小/大型文档 | 两种规模的采样策略不同（小型全读，大型抽样） |
| 元架构灵活性 | **F1: 一次性确定** — 构建全程不变 | 简化实现；后续可升级到 F2 |

### Research Context
- 当前 Phase 0 只做根实体识别（`_identify_root`，单次 LLM 调用），元架构是此概念的泛化
- 当前 `EXTRACTION_CANDIDATES_PROMPT` 是静态的（硬编码 20 种关系类型），元架构使其动态生成
- 当前 `_process_candidates` 接受任何类型的实体/关系，元架构可做后验过滤
- MCP `sysml_batch` 已在 Task 1 实现，可直接用于元架构查询

---

## Work Objectives

### Core Objective
在 KG 构建管线中插入"元架构确定"阶段（Phase -1），用户通过自然语言粒度描述（CLI `--granularity`）控制提取的精细程度，LLM 根据文档结构+采样确定 Schema，后续 Phase 按 Schema 约束进行填充，实现**用户可控的 KG 粒度**。

### Concrete Deliverables
- `agent/kg_granularity.py` — 新模块：`MetaArchitecture` 数据类 + `GranularityAgent` 类
- `scripts/sysml_rag_mcp_server.py` — 新增 MCP 工具 `sysml_meta_schema`（元架构 CRUD）
- `agent/kg_build_agent.py` — Phase -1 集成 + 动态 Prompt 生成 + 元架构验证
- `scripts/build_kg_recursive.py` — `--granularity` CLI 参数 + 粒度配置

### Definition of Done
- [ ] `--granularity "只提取顶层架构"` 构建出的 KG 实体数明显少于无粒度约束版本
- [ ] `--granularity "详细到端口和命令级别"` 构建出的 KG 包含 AttributeDef 和 CommandDef
- [ ] 元架构在 Phase -1 确定后，Phase 1-3 自动使用约束
- [ ] 无粒度参数时，行为与当前管线完全一致（向后兼容）
- [ ] 元架构 Schema 持久化到 `build.json`，支持断点续跑

### Must Have
- 自然语言粒度输入 → 元架构生成的完整流程
- 元架构包含：允许的实体类型、根节点、关系模式
- 动态 Prompt 生成（将元架构注入提取 Prompt）
- 无粒度参数时的向后兼容
- 元架构的可观测性（日志/JSON 输出）

### Must NOT Have
- 不破坏现有的无粒度模式（`--granularity` 未指定时完全不变）
- 不改变 BFS 级联或实体传播逻辑
- 不对 LLM 提取质量做硬性假设——元架构是指南，不是硬过滤
- 不引入新模型（复用现有的 `self.llm` 和 `self.light_llm`）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO
- **Agent-Executed QA**: Bash（构建脚本运行 + 日志分析 + 元架构 JSON 验证）

### QA Policy
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

```
Wave 1 (Foundation — data model + MCP tools, MAX PARALLEL):
├── Task 1: MetaArchitecture 数据类 + GranularityAgent [deep]
└── Task 2: sysml_meta_schema MCP 工具 [quick]

Wave 2 (Pipeline integration):
├── Task 3: Phase -1 插入 build_kg_recursive [deep]
└── Task 4: 动态 Prompt 生成器 [quick]

Wave 3 (Extraction constraint):
└── Task 5: _process_candidates 元架构过滤 [quick]

Wave 4 (CLI + Config):
└── Task 6: --granularity CLI 参数 [quick]

Wave 5 (Integration & verification):
├── Task 7: 不同粒度构建对比测试 [deep]
└── Task 8: 回归测试（无粒度参数 = 基线） [deep]

Wave FINAL:
├── F1: Plan compliance audit [oracle]
├── F2: Code quality review [unspecified-high]
├── F3: Real manual QA [unspecified-high]
└── F4: Scope fidelity check [deep]
```

---

## TODOs

- [ ] 1. `MetaArchitecture` 数据类 + `GranularityAgent`

  **What to do**:
  创建 `agent/kg_granularity.py` 新文件。包含两个核心类。

  **`MetaArchitecture` 数据类**：
  ```python
  @dataclass
  class MetaArchitecture:
      entity_types: List[str]  # 允许的实体类型: ["PartDef","AttributeDef","CommandDef"]
      root_nodes: List[Dict]    # [{"name":"系统概览","type":"PartDef","description":"..."}]
      relation_patterns: List[Dict]  # [{"source_type":"PartDef","target_type":"PartDef","relation_type":"allocation","desc":"组成关系"}]
      constraints: List[str]    # 自然语言约束: ["忽略温度参数","不提取命令"]
      granularity_description: str  # 原始用户输入
  ```

  **`GranularityAgent` 类**（三步流程）：

  **Step 1 — TOC 分析**：读取 `DocumentTreeState.get_tree_structure()`，识别章节层次、内容密度（每章页数）、关键章节

  **Step 2 — 小节采样**：LLM 根据 TOC 决定采样 3-5 个小节（LLM 输出 node_id 列表）。Agent 用 MCP `read_section` 读取采样文本

  **Step 3 — 元架构生成**：LLM 综合粒度描述 + TOC + 采样文本 → 输出 `MetaArchitecture` JSON。调用 MCP `sysml_check_section` 验证采样是否足够

  **LLM 调用模式**：Step 1+2 可合并为 1 次 Agent 调用（LangChain Agent，2-3 个工具调用），Step 3 为单独 1 次调用

  **Must NOT do**: 不使用大模型（用 `self.light_llm`/`self.llm`），不创建新的 LangChain Agent（复用现有 `_get_agent` 模式）

  **Recommended Agent Profile**: `deep` — 新模块设计，核心逻辑

  **Acceptance Criteria**:
  - [ ] `MetaArchitecture` 数据类可 JSON 序列化/反序列化
  - [ ] `GranularityAgent.determine_meta_architecture()` 返回完整元架构
  - [ ] TOC 分析输出采样节点建议

- [ ] 2. `sysml_meta_schema` MCP 工具

  **What to do**:
  在 `scripts/sysml_rag_mcp_server.py` 新增 2 个 MCP 工具：

  **`sysml_set_meta_schema`**：将 `MetaArchitecture` JSON 存入 `SysMLManager`（新增 `_meta_schema` 属性）
  ```python
  def sysml_set_meta_schema(schema_json: dict) -> dict
  ```

  **`sysml_get_meta_schema`**：查询当前元架构
  ```python
  def sysml_get_meta_schema() -> dict
  ```

  **`SysMLManager` 修改**：`__init__` 增加 `self._meta_schema: Optional[dict] = None`

  **Must NOT do**: 不修改现有工具函数，不改变序列化格式

  **Recommended Agent Profile**: `quick` — 两个简单 CRUD 工具

  **Acceptance Criteria**:
  - [ ] `tools/list` 包含 `sysml_set_meta_schema` 和 `sysml_get_meta_schema`
  - [ ] `sysml_set_meta_schema` 后 `sysml_get_meta_schema` 返回一致内容

- [ ] 3. Phase -1 插入 `build_kg_recursive`

  **What to do**:
  在 `agent/kg_build_agent.py` 的 `build_kg_recursive()` 中 Phase 0 之前插入 Phase -1：

  ```python
  # ── Phase -1: 元架构确定 ──
  meta_arch = None
  if granularity_description:
      from agent.kg_granularity import GranularityAgent
      ga = GranularityAgent(self, doc_name)
      meta_arch = await ga.determine_meta_architecture(
          granularity_description, tree_state, sections
      )
      await self._mcp_session.call_tool("sysml_set_meta_schema", {
          "schema_json": asdict(meta_arch)
      })
      self._save_recursive_state(doc_name, "recursive_phase0", meta_schema=asdict(meta_arch))
  ```

  **断点续跑支持**：`_load_build_state` 中恢复 `meta_schema`；如果已存在则跳过 Phase -1

  **Must NOT do**: 不改变 Phase 0-3 的逻辑，中间不阻断现有流程

  **Recommended Agent Profile**: `deep` — 管线集成，需要理解完整流程

  **Acceptance Criteria**:
  - [ ] `granularity_description=None` 时 Pipeline 行为不变
  - [ ] 有粒度描述时 Phase -1 先于 Phase 0 执行
  - [ ] 元架构序列化到 `build.json` 中

- [ ] 4. 动态 Prompt 生成器

  **What to do**:
  在 `KGBuildAgent` 中新增 `_build_extraction_prompt()` 方法：

  ```python
  def _build_extraction_prompt(self) -> str:
      """根据元架构动态生成提取 Prompt"""
      meta = await self._mcp_session.call_tool("sysml_get_meta_schema", {})
      schema = json.loads(meta).get("schema", {})
      
      if not schema:
          return EXTRACTION_CANDIDATES_PROMPT  # 回退到静态 Prompt
      
      types = schema.get("entity_types", [])
      relations = schema.get("relation_patterns", [])
      constraints = schema.get("constraints", [])
      roots = schema.get("root_nodes", [])
      
      # 动态构建 Prompt
      prompt = f"""你是技术文档实体提取器。当前粒度要求：{schema.get('granularity_description','')}
  
  ## 允许的实体类型
  {chr(10).join(f'- {t}' for t in types)}
  
  ## 允许的关系模式
  {chr(10).join(f'- {r[\"source_type\"]} → {r[\"relation_type\"]} → {r[\"target_type\"]}: {r[\"desc\"]}' for r in relations)}
  
  ## 约束
  {chr(10).join(f'- {c}' for c in constraints)}
  
  ## 根节点
  {chr(10).join(f'- {r[\"name\"]} ({r[\"type\"]}): {r[\"description\"]}' for r in roots)}
  
  ## 输出格式
  输出一个JSON数组...（保留原格式描述）"""
      return prompt
  ```

  在 `_extract_page_candidates` 和 `_process_queued_section` 中用 `self._build_extraction_prompt()` 替代硬编码的 `EXTRACTION_CANDIDATES_PROMPT`

  **Must NOT do**: 不取代 `EXTRACTION_CANDIDATES_PROMPT`——无元架构时回退

  **Recommended Agent Profile**: `quick` — 字符串拼接逻辑

  **Acceptance Criteria**:
  - [ ] 无元架构时 Prompt = 静态 `EXTRACTION_CANDIDATES_PROMPT`
  - [ ] 有元架构时 Prompt 包含实体类型、关系模式、约束、根节点

- [ ] 5. `_process_candidates` 元架构过滤

  **What to do**:
  在 `_process_candidates_batch()` 中（或新增包装方法），添加元架构一致性检查：

  ```python
  # 在实体创建前检查
  for c in candidates:
      name = c.get("name")
      etype = c.get("type")
      if meta_schema and etype not in meta_schema.get("entity_types", []):
          logger.debug("Skipping entity '%s' type=%s (not in meta-schema)", name, etype)
          continue  # 跳过元架构不允许的类型
  ```

  **注意**：这是软过滤（log + skip），不是硬阻断。LLM 可能输出不在元架构中的类型——跳过而非报错。

  **Must NOT do**: 不要删除实体——只是不在元架构允许列表中的不创建

  **Recommended Agent Profile**: `quick` — 条件检查逻辑

  **Acceptance Criteria**:
  - [ ] 元架构允许的类型全部创建
  - [ ] 元架构禁止的类型被跳过（log 中有 skip 消息）

- [ ] 6. `--granularity` CLI 参数

  **What to do**:
  在 `scripts/build_kg_recursive.py` 添加：
  ```python
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--granularity", type=str, default=None,
                      help="自然语言粒度描述 (如 '只提取顶层架构')")
  args = parser.parse_args()
  ```

  在 `main()` 中传入 `granularity_description=args.granularity` 到 `agent.build_kg_recursive()`

  同时在 `OPTIONS` 配置中添加可选键 `"granularity"`

  **Must NOT do**: 不改变默认行为（无参数时 = 无粒度控制）

  **Recommended Agent Profile**: `quick` — argparse 添加

  **Acceptance Criteria**:
  - [ ] `--granularity "只提取顶层架构"` 被正确解析并传入 Agent
  - [ ] 无 `--granularity` 时行为不变

- [ ] 7. 不同粒度构建对比测试

  **What to do**:
  用三种粒度各跑一次 AIOPS_New 构建（每个限制 60 分钟）：
  - 无粒度（基线）
  - `--granularity "只提取顶层系统架构，忽略命令和参数细节"`
  - `--granularity "详细到每个端口、属性、命令级别"`

  对比输出：
  - 实体/关系数量（粗粒度应明显少于细粒度）
  - 实体类型分布（粗粒度应缺少 CommandDef/AttributeDef）
  - 连通分量（粗粒度应更少、更大）

  **Recommended Agent Profile**: `deep` — 长运行测试 + 对比分析

  **Acceptance Criteria**:
  - [ ] 粗粒度实体数 < 细粒度实体数
  - [ ] 粗粒度缺少某些实体类型
  - [ ] 无粒度模式 = 基线

- [ ] 8. 回归测试

  **What to do**:
  无粒度参数的 AIOPS_New 构建与基线对比：
  - 实体/关系数量在 ±5% 内
  - `.sysml` 格式正确
  - 无 MCP 错误

  **Recommended Agent Profile**: `deep` — 端到端验证

  **Acceptance Criteria**:
  - [ ] 无粒度参数 = 基线（在 LLM 随机性范围内）
  - [ ] 所有 Phase -1 代码在此模式下被跳过

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
- [ ] F2. **Code Quality Review** — `unspecified-high`
- [ ] F3. **Real Manual QA** — `unspecified-high`
- [ ] F4. **Scope Fidelity Check** — `deep`

---

## Commit Strategy
- **1-2**: `feat(granularity): MetaArchitecture data model + MCP schema tools`
- **3-4**: `feat(granularity): Phase -1 integration + dynamic prompt generation`
- **5-6**: `feat(granularity): candidate filtering + CLI --granularity argument`
- **7-8**: `test(granularity): multi-granularity comparison + regression`

## Success Criteria
```bash
# 粗粒度构建
python scripts/build_kg_recursive.py --granularity "只提取顶层架构"

# 细粒度构建
python scripts/build_kg_recursive.py --granularity "详细到端口和命令级别"

# 回归（无粒度 = 旧行为）
python scripts/build_kg_recursive.py
```
