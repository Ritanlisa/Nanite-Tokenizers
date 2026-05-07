# KG-Build Plan: SysML-v2 知识图谱构建系统

## 目标

在 `web_server.py` 的文档树构建流程中，通过 MCP 协议驱动 Build Agent，使用 LLM 对文档进行实体提取与 SysML-v2 知识图谱构建，作为迷你数字孪生的基础。

## 架构总览

```
┌─ web_server.py ──────────────────────────────────────────────┐
│  构建流程 (rag_db_build)                                      │
│    ├── 文档加载 → 树构建 (现有)                                 │
│    │                                                          │
│    ├── [NEW] KG 构建阶段                                       │
│    │    ├── Phase 1: Entity Build Agent (实体提取)             │
│    │    │    ├── Per-Section 独立会话 (无上下文累积)            │
│    │    │    ├── search-before-create 去重策略                  │
│    │    │    └── MCP Client ──→ MCP SysML Server               │
│    │    │                        ├── sysml_search_entity      │
│    │    │                        ├── sysml_add_entity         │
│    │    │                        ├── sysml_update_entity      │
│    │    │                        └── sysml_add_alias          │
│    │    │                                                      │
│    │    ├── Cross-Section Dedup                               │
│    │    │    ├── sysml_suggest_merge                          │
│    │    │    └── sysml_merge_entities                         │
│    │    │                                                      │
│    │    └── Phase 2: Relation Build Agent (关系提取)           │
│    │         ├── Per-Section 独立会话                          │
│    │         └── MCP Client ──→ MCP SysML Server               │
│    │                              ├── sysml_search_entity     │
│    │                              └── sysml_add_relation      │
│    │                                                          │
│    ├── 向量索引构建 (现有)                                      │
│    └── 持久化 (现有 + .sysml KG 文件)                          │
│                                                                
│  未来: Chat Agent 多 Agent 模式 → 同一 MCP Server               │
└──────────────────────────────────────────────────────────────┘
```

## 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 存储格式 | `.sysml` 纯文本文件 | 人/机器可读，SysML 生态兼容 |
| 抽取触发 | 构建时自动执行 | 无缝集成，零用户干预 |
| 抽取模型 | 独立可配置 KG_EXTRACTION_MODEL | 可与主 LLM 不同，专用轻量模型 |
| 抽取粒度 | Per Section/MonoPage | 上下文聚焦，避免窗口溢出 |
| 工具接入 | MCP 协议 | 可扩展，后续支持多 Agent |
| 处理模式 | 迭代式 Agent 自主决定 | 灵活处理复杂关系，支持推理 |

## 上下文管理策略

**Per-Section 独立会话 + MCP Server 持状态：**

- 每个 Section 使用新建 LangChain Agent 会话
- Section 完成后会话丢弃，上下文不累积
- MCP Server 是唯一状态持有者（实体、别名注册表）
- 下一 Section 从空白会话开始，通过 MCP 查询现有状态

## MCP 工具集

### 实体 CRUD 工具

| 工具名 | 功能 | 关键参数 |
|--------|------|---------|
| `sysml_add_entity` | 创建实体（含别名/属性/来源） | entity_type, name, description, aliases, source_sections, source_text, properties, parent_package |
| `sysml_update_entity` | 补充/合并实体信息 | qualified_name, append_description, merge_aliases, append_source_sections, update_properties |
| `sysml_delete_entity` | 删除实体 | qualified_name, parent_package |
| `sysml_search_entity` | 多策略搜索（精确→归一化→别名→子串→正则→Token重叠） | query, regex_pattern, fuzzy, threshold |
| `sysml_get_entity` | 实体详情 | qualified_name |
| `sysml_normalize_name` | 名称归一化 | name → {normalized, tokens} |
| `sysml_list_entities` | 全部实体列表 | include_details |
| `sysml_add_alias` | 追加别名 | qualified_name, alias |

### 关系 CRUD 工具

| 工具名 | 功能 | 关键参数 |
|--------|------|---------|
| `sysml_add_relation` | 创建关系 | relation_type, name, source, target, parent_package, description |
| `sysml_delete_relation` | 删除关系 | name, parent_package |
| `sysml_get_connections` | 实体关联查询 | entity_name |
| `sysml_list_relations` | 全部关系列表 | include_details |

### 合并/去重工具

| 工具名 | 功能 |
|--------|------|
| `sysml_suggest_merge` | 全局去重建议（基于别名/名称相似度） |
| `sysml_merge_entities` | 执行合并（转移别名/来源/关系到目标实体） |

### 通用工具

| 工具名 | 功能 |
|--------|------|
| `sysml_load_model` | 加载 .sysml 文件 |
| `sysml_save_model` | 保存 .sysml 文件 |
| `sysml_export_submodel` | 子模型导出 |
| `sysml_model_summary` | 全局统计摘要 |
| `sysml_semantic_search` | 语义搜索 |
| `sysml_import_doc` | 从文档导入（保留兼容） |

## 搜索匹配策略

```
输入: query="DAQ模块"

策略1: 精确匹配    query == entity.name                                    → confidence: 1.0
策略2: 归一化匹配  normalized(query) == normalized(entity.name)            → confidence: 0.95
策略3: 别名匹配    query in entity.aliases                                 → confidence: 0.90
策略4: 子串匹配    query ⊂ entity.name 或 entity.name ⊂ query             → confidence: 0.70
策略5: 正则匹配    用户提供的 regex 命中                                    → confidence: 0.80
策略6: Token重叠   CN_char 交集 / min(len(a), len(b)) > 0.6               → confidence: 0.60
```

## 两个 Build Agent System Prompt

### 实体提取 Agent

```
你是SysML v2实体提取专家。你的任务是从技术文档章节中识别系统架构实体。

规则：
1. 仔细阅读章节内容，理解上下文
2. 识别任何实质性的系统组成元素：组件、模块、设备、子系统、
   属性参数、接口、需求约束、数据实体
3. 对每个发现，先搜索是否已存在——使用不同的表述尝试搜索
   （例如 "DAQ" 可能已注册为 "数据采集模块" 的别名）
4. 搜索到高置信度匹配时，更新而非新建
5. 搜索无匹配时，创建新实体并注册你识别到的所有别名
6. 每个实体记录原文出处
7. 章节中无架构内容时，直接结束
```

### 关系提取 Agent

```
你是SysML v2关系提取专家。你的任务是从技术文档章节中识别实体间关系。

规则：
1. 仔细阅读章节内容
2. 识别实体之间的：物理连接、数据流、接口实现、
   功能分配、继承/组合、需求满足关系
3. 确认端点实体存在——使用搜索工具验证名称
4. 端点实体必须精确匹配，不确定时尝试不同表述搜索
5. 无明确关系时直接结束
```

## 配置项（新增至 config.py + settings.yaml）

```yaml
KG_EXTRACTION_ENABLED: true           # 总开关
KG_EXTRACTION_MODEL: qwen3-vl:32b     # 独立于主 LLM 的抽取模型
KG_EXTRACTION_TEMPERATURE: 0.1        # 低温保证一致性
KG_EXTRACTION_MAX_TOKENS: 4096
KG_EXTRACTION_TIMEOUT: 120            # 单 Section 超时
KG_EXTRACTION_MAX_ITERATIONS: 10      # Agent 最大迭代轮次
KG_MCP_SERVER_COMMAND: "python scripts/sysml_rag_mcp_server.py serve"
KG_MERGE_CONFIDENCE_THRESHOLD: 0.7

SIMILARITY_TOP_K: 5
ENABLE_RERANK: true
RERANK_MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_DEVICE: cpu
RERANK_TOP_N: 3
```

## 文件变更清单

| 文件 | 动作 | 步骤 |
|------|------|------|
| `sysml/sysml_manager.py` | **修改**: AliasRegistry + metadata_store + CRUD + merge | Step 1 |
| `scripts/sysml_rag_mcp_server.py` | **重写**: 新增 CRUD/合并工具 + 增强搜索 | Step 1 |
| `mcp_client/mcp_session.py` | **新增**: 通用 MCP stdio 会话管理器 | Step 2 |
| `mcp_client/tool_wrapper.py` | **新增**: MCP→LangChain BaseTool 包装器 | Step 2 |
| `mcp_client/client.py` | **重构**: 退化为 MCPSession 特化实例 | Step 2 |
| `agent/kg_build_agent.py` | **新增**: 两阶段构建协调器 | Step 3 |
| `config.py` | **新增**: KG 配置项 | Step 3 |
| `settings.yaml` | **新增**: KG 配置默认值 | Step 3 |
| `rag/engine.py` | **修改**: 构建链插入 KG 阶段 | Step 4 |
| `web_server.py` | **修改**: 新构建回调阶段 + KG API 端点 | Step 4 |
| `agent/tools.py` | **修改**: SysML LangChain 工具标记废弃（逻辑已迁移到 MCP） | Step 2 |

## 实施步骤

### Step 1: 增强 MCP SysML Server (sysml_manager.py + sysml_rag_mcp_server.py)
- [x] sysml_manager.py: AliasRegistry + metadata_store
- [ ] sysml_manager.py: CRUD 方法 (add_entity_with_metadata, update_entity_metadata, delete_entity)
- [ ] sysml_manager.py: Merge 方法 (suggest_merges, merge_entities)
- [ ] sysml_manager.py: Enhanced search (search_entities with multi-strategy)
- [ ] sysml_rag_mcp_server.py: 新增 CRUD MCP 工具 (7个)
- [ ] sysml_rag_mcp_server.py: 新增合并/去重 MCP 工具 (2个)
- [ ] sysml_rag_mcp_server.py: 增强 sysml_search_entity (别名/多策略)
- [ ] sysml_rag_mcp_server.py: 更新 TOOL_DEFINITIONS

### Step 2: 通用化 MCP Client (mcp_client/)
- [ ] mcp_session.py: 通用 MCP stdio 会话管理器
- [ ] tool_wrapper.py: MCP→LangChain BaseTool 包装器
- [ ] agent/tools.py: 标记 SysML 工具废弃

### Step 3: 创建 Build Agent 协调器 (agent/kg_build_agent.py)
- [ ] kg_build_agent.py: 两阶段编排器
- [ ] config.py + settings.yaml: KG 配置项

### Step 4: 构建流程集成 (rag/engine.py + web_server.py)
- [ ] rag/engine.py: 构建链插入 KG 阶段
- [ ] web_server.py: 新构建回调 + KG API 端点

### Step 5: 后续（本次不实现）
- [ ] Chat Agent 多 Agent 模式 → 连接同一 MCP Server
- [ ] 动态链接：KG 实体 ↔ 实时数据源
- [ ] 方法调用：Port ↔ API/函数签名
- [ ] RAG 检索联动：KG 实体 ↔ doc_tree_cache section_id

## 测试要求

每步实现后必须主动测试：
1. 单元测试：新增/修改的方法用 Python 脚本直接调用验证
2. MCP 工具测试：通过 MCP stdio 协议调用验证往返正确性
3. 集成测试：端到端构建流程验证 KG 生成
4. 手动测试命令示例：
   ```bash
   # 测试 MCP Server
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/sysml_rag_mcp_server.py serve
   
   # 测试单工具调用
   python scripts/sysml_rag_mcp_server.py run sysml_model_summary
   ```

## 文件大小

项目现有 22 个目录，本计划新增/修改 11 个文件，预计新增代码约 2500 行。
