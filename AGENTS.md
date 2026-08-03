# AGENTS.md — Nanite-Tokenizers 项目知识

## 项目概述
- **名称**: Nanite-Tokenizers
- **仓库**: https://github.com/Ritanlisa/Nanite-Tokenizers
- **Python**: >=3.12, 包管理 `uv`, pyproject.toml
- **主入口**: `web_server.py` (FastAPI), `main.py` (Gradio)
- **MCP 服务器**: `scripts/sysml_rag_mcp_server.py` (stdio JSON-RPC)

## 核心架构

### KG 构建流程
```
RAG_DB_Document (文档树)
    → KGBuildAgent.build_kg_from_document(rag_doc)
        → DocumentTreeState 构建层次树
        → Phase 1a: 并行 light_llm 调用（Semaphore控制，BATCH_CONCURRENCY=5路并发）
        → Phase 1b: 顺序 MCP 工具调用来创建实体/关系（去重+创建）
        → Phase 2: 跨章节去重（无LLM，<1s）
        → Phase 3: 卸载小模型 → 强模型(llm) LangChain Agent 富化关系/别名
        → 保存 .sysml
```

### 并行架构 (2026-05 新增)
- Phase 1a: 使用 `asyncio.gather` + `Semaphore(BATCH_CONCURRENCY)` 并行调用 light_llm
- 所有 LLM 调用先提交（消除 MCP 处理导致的请求间延迟），然后顺序处理 MCP 操作
- `model_kwargs={"keep_alive": "30s"}` 确保空闲 30s 后自动卸载模型
- Phase 2 结束后显式调用 Ollama `/api/generate` (`keep_alive=0`) 卸载小模型
- 大模型（强模型）用于 Phase 3 富化，此时显存已清理

### 关键文件
| 文件 | 职责 |
|------|------|
| `agent/kg_build_agent.py` | KG 构建 Agent：DocumentTreeState, 导航工具, 统一提取 Agent |
| `scripts/sysml_rag_mcp_server.py` | MCP 服务器：SysML CRUD 工具 (30+) |
| `sysml/sysml_manager.py` | 内存模型管理器：实体/关系 CRUD, 别名注册表, 合并去重 |
| `sysml/sysml_model.py` | SysML v2 AST：PartDef/AttributeDef/ConnectionUsage 等 |
| `rag/document_interface.py` | 文档树类型：Page/Chapter/MonoPage/RAG_DB_Document |
| `rag/documents.py` | 文档加载：LibreOffice (.doc), PyMuPDF (.pdf) |

### MCP 工具架构
```
KGBuildAgent (LangChain Agent)
  ├── 本地导航工具 (get_document_tree, read_section, mark_section_done, get_progress)
  └── MCP 工具 (通过 MCPToolWrapper + MCPSession stdio)
      └── scripts/sysml_rag_mcp_server.py serve
          └── SysMLManager (内存模型)
```

## 设计决策

### 统一 Agent (2026-05 重构)
- 替代了 Phase 1 (实体) + Phase 2 (关系) 分离设计
- 单一 Agent 同时提取实体和关系（更符合阅读直觉）
- LLM 自主导航文档树，从整体到局部
- 打勾机制：提取完成打勾，全部打勾即完成

### 冻结工具
- `sysml_add_command/set_hostname/add_cabinet_instance/add_chapter_ref/add_quantity/add_ip_config/set_display_name`
- 代码保留但未纳入活跃工具列表
- 注意：长分隔线注释包裹 `═══════`

### merge_entities 行为
- source 名称 → target 别名
- source 所有别名 → 转移到 target
- source 所有关系 → .ends[].ref 重定向到 target.name
- source metadata → 合并到 target
- source 删除

### 提示词策略
- 不枚举具体实体类型清单（避免"对着试卷复习"）
- 给方向性指导，让 LLM 自主发现
- 从整体到局部导航

## 已知 Bug 与踩坑记录

### Relation ends 在序列化时丢失 (2026-05 发现并修复)
- **症状**: `.sysml` 文件中 relation 只有名字（如 `connection 'A_B_Connection';`），没有 `connect to` 子句，导致加载后 ends=[]，KG 完全不连通
- **根因**: `ConnectionUsage.to_text()` 对有名字的 connection 直接调用父类 `Usage.to_text()`，不输出 ends。因为 `if not self.name` 判断，有名字就跳过 connect 简写
- **修复**: `sysml/sysml_model.py:328` — 当有名字且有 ends 时，用 body block 格式序列化（`connection name { connect A to B; }`）
- **协同修复**: `sysml/sysml_parser.py:172` — transformer 的 `usage()` 方法检测 body 中的 `connect_usage`，将其 ends 复制到父级 ConnectionUsage
- **影响**: 之前构建的所有 KG 文件（如 AIOPS_New/knowledge_graph.sysml）中 relation 的 ends 全部丢失。重新运行 KG 构建才能使用正确的序列化格式

### _entity_type_name() isinstance 顺序 Bug (2026-05 发现并修复)
- `scripts/sysml_rag_mcp_server.py:83`
- AllocationUsage 和 InterfaceUsage 都继承自 ConnectionUsage，但 dict 中 ConnectionUsage 排在前面
- 导致所有 AllocationUsage 被误判为"连接使用"而非"分配使用"
- 修复：把子类型（InterfaceUsage, AllocationUsage）排在父类型（ConnectionUsage）前面

### KG 可视化前端 (2026-05 新增，2026-06 多次重写)
- 访问: `GET /kg/viz/{db_name}` (如 `/kg/viz/AIOPS_New`)
- 数据源: `GET /api/rag/dbs/{db_name}/kg/graph`
- 技术栈: cytoscape.js CDN
- **当前布局**: 自适应选择 Simple Layout 或 Radial Tree
- 功能: 类型筛选、搜索、连通分量统计、孤立节点标记、节点详情面板、section 透明度过滤、边点击详情
- 启发式边恢复: 当 relation.ends 为空时，从命名约定解析 (A_B_Type → A→B 边)，覆盖率 96% (188/195)

## 20-Type SysML 关系系统 (2026-06 新增)

### 关系类型谱系
原始只有 3 种关系类型 (Connection, Interface, Allocation)，分两步扩展到 20 种:

**Step 1** (ebb1cd7, Jun 12): ContainmentUsage, CompositionUsage, ReferenceUsage
**Step 2** (6697cb4, Jun 12): GeneralizationUsage, DependencyUsage, AbstractionUsage, RealizationUsage, DeriveUsage, TraceUsage, DeriveReqtUsage, RefineUsage, SatisfyUsage, VerifyUsage, CopyUsage, UseCaseAssociationUsage, UseCaseIncludeUsage, UseCaseExtendUsage

全部 20 种类型定义在 `sysml/sysml_model.py`，均继承自 `ConnectionUsage`:
```
Connection, Interface, Allocation,
Containment, Composition, Reference,
Generalization, Dependency, Abstraction,
Realization, Derive, Trace,
DeriveReqt, Refine, Satisfy,
Verify, Copy,
UseCaseAssociation, UseCaseInclude, UseCaseExtend
```

### 序列化格式
所有 20 种类型统一使用 body block 格式序列化:
```
containment 'name' {
    connect A to B;
}
```
`ConnectionEnd` 类 (ref + role) 表达端点。

### 注册与映射
- `sysml/sysml_manager.py:67-88` — `RELATION_CLASS_MAP` 字典: lowercase 类型名 → Python class
- `sysml/sysml_parser.py` — 每种类型的 grammar rule + transformer
- `scripts/sysml_rag_mcp_server.py:2242` — `sysml_add_relation` 的 `relation_type` 参数列出全部 20 种
- `agent/kg_build_agent.py` — `EXTRACTION_CANDIDATES_PROMPT`, `ENRICHMENT_JSON_PROMPT`, `_bridge_components` prompt 均列出全部 20 种

### 前端支持
- `web/kg_viz.html` — 每种类型有独立颜色，`DIRECTIONAL` 集合标记非对称关系类型，CSS 选择器覆盖
- `web_server.py:1807-1817` — `TYPE_EDGE_PRIORITY` 用于边去重时的优先级决策

## BFS Cascade 算法 (2026-06)

### 递归级联管线
位于 `KGBuildAgent.build_kg_recursive()` (`agent/kg_build_agent.py:1866`)，替代原有的简单 Phase 1→2→3→4 管线:

```
Phase 0: gemma4:31b 识别根实体 + 2-4 个起点小节
Phase 1: gemma4:31b 根小节完整提取（使用 EXTRACTION_CANDIDATES_PROMPT 格式）
Phase 2: 实体传播 → 全文搜索每个新实体在所有未处理小节中的提及 → 构建队列
Phase 3: qwen3:8b 级联队列处理（逐节提取+自动连接焦点实体）
循环 Phase 2→3 直到队列为空
收尾: 跨章节去重 + 图聚合
```

### 实体传播 (Phase 2)
1. 对每个新发现的实体，获取其所有别名
2. 遍历所有未处理小节，对每个 (entity, section) 配对:
   - 检查实体名/别名是否在该小节的文本中出现（大小写不敏感 substring match）
   - 匹配则加入 `build_queue`
3. 队列空空时，使用根实体作为 fallback 处理剩余小节（按小节大小排序，每次最多 5 个）

### 级联处理 (Phase 3)
- `_process_queued_section()` — 使用 `EXTRACTION_CANDIDATES_PROMPT` 提取小节中与焦点实体相关的内容
- 新发现的实体自动用 `allocation` 关系连接到焦点实体（保证结构完整性）
- 小节内继续级联: 新实体进入 `section_entity_queue`，在同小节内继续传播

### _connect_global_orphans 移除 (0c0b1f2, Jun 12)
- 原代码在所有处理完成后将所有孤立实体强制连接到根实体
- 移除原因: 创建了大量虚假的双向关系，污染 KG 质量
- 替代方案: Phase 1/3 中的 per-section auto-connection（仅连接本小节内的实体到焦点实体）

## 边去重策略

### 核心逻辑 (web_server.py:1806-1865)
构建 API 返回的边列表时，对 relation 进行去重:

1. **无序对去重**: 边 key = `(min(a, b), max(a, b))`，消除 A→B 和 B→A 重复
2. **类型优先级**: 同一无序对出现多次时，保留优先级最高的类型:
   - Containment/Composition = 5 (最高)
   - Reference = 4
   - Allocation/Dependency/Generalization/Realization/Refine/Satisfy/Verify = 3
   - Connection/Abstraction/Derive/Trace/DeriveReqt/UseCaseAssociation/UseCaseInclude/UseCaseExtend = 2
   - Interface/Copy = 1 (最低)
3. **同优先级时**: 保留描述文本更长的边
4. **双向消除**: 不存在 A→B 和 B→A 两条边同时出现在最终结果中

## 前端布局设计 (2026-06 多次重写)

### 布局演化历程
提交历史记录了从力导向到最终 Radial Tree 的完整旅程:

| 提交 | 算法 | 结果 |
|------|------|------|
| 587652a | Force layout (edge crossing penalty + ideal spring) | 有重叠 |
| 426a7fa | Spectral (Laplacian eigenmap per component) | 力导向样分布 |
| ace5cd9 | Shift-invert power iteration (Fiedler vector) | 仍不收敛 |
| c50a8dd | powerIter 200→5000 | 仍稀疏 |
| 0f53718 | KK stress majorization | 数学错误 |
| df346fb | KK fix (correct math) | 树状结构好 |
| 12a545b | MST radial tree (min-spanning tree → BFS levels) | 有重叠 |
| fbeffbf | MST 触发修复 | - |
| 6ebe2d7 | BFS tree + layered layout (H_GAP spacing) | 列重叠 |
| 0f0f511 | 修复 cx/cy 变量名冲突 | - |
| 5e0f3b0 | 移除 d<maxDepth guard (最深层叶子不布局) | - |
| e6a98b4 | Top-down tree layout (唯一列，零重叠保证) | 宽图 |
| dff40c7 | **Radial tree (leaf-count angular)** | 当前使用 |

### 最终 Radial Tree 算法 (kg_viz.html:277-382)
- `LEVEL_SPACING = 120` (每层半径增量)
- BFS 从最高度节点构建生成树
- 自底向上计算每个子树的叶子数
- 角度分配按子树叶子数比例 (`leafCount[v] / leafCount[u]`)
- 每个连通分量独立计算布局
- 自动缩放到画布 + 居中
- 2 节点分量: 水平排列; 1 节点: 居中

## 断点续跑 & 防数据丢失 (CRITICAL)

### build.json 生命周期
- 文件: `database/{db_name}/knowledge_graph.build.json`
- **永远不删除** — 这是崩溃恢复的唯一状态源
- 原子写入: 先写 `.tmp` 文件，再用 `os.replace` 原子替换
- 状态结构: `{"documents": {doc_name: {phase, pipeline, processed_sections, build_queue, ...}}}`

### Ctrl+C 安全
- `KeyboardInterrupt` 在 `build_kg_from_document()` 和 `build_kg_recursive()` 中捕获
- 中断时: 保存 build.json 状态 + 触发 save_model → 下次启动从断点继续
- `_save_build_state()` 每节结束后调用，保存队列和已处理集合

### 历史教训 (AIOPS_New 数据丢失)
- 2026-06 某次 Ubuntu 崩溃导致 `database/AIOPS_New/` 下所有文件丢失
- 丢失: knowledge_graph.sysml (448 实体, 1166 关系, 149KB), build.json, docs/
- 根因: build.json 与 .sysml 在同一目录，目录级损坏导致全部丢失
- **教训**: build.json 是唯一的断点续跑状态，失去它就失去一切

### rm-f 历史
- 曾经误用 `rm -f build.json` 导致中间状态丢失，事后重新加入
- **禁止**手动删除 build.json (除非确认已完成构建且不需要断点续跑)

## 根实体 section 名称

- `_process_root_sections()` 创建根实体时 `source_sections=["根实体"]`
- 不是文件路径，不是 "根实体根" 等冗余名称
- 确保所有根实体在 source_sections 过滤时能被找到

## source_sections 传递

- `_process_candidates()`: `pages_list = source_pages or [section_title or f"p{page}"]`
- Enrichment Phase 3: 当 sec_key 为空时 fallback 到 `["富化"]`
- 创建关系时: `source_sections` 参数传递 rel_src_sections (关系自身) + pages_list (当前页面)
- Auto-connect 关系 (allocation 到焦点实体) 带 `source_pages` 参数

## Ograg2 对比分析 (2026-06 完成)

- 分析文件: `/home/ritanlisa/some-path/ograg2-analysis/` (不在 repo 内)
- 对比维度: 实体提取准确率、关系覆盖率、去重效率
- 结论: Ograg2 在 attribute/port 实体提取上更优，但 command 实体为 Nanite-Tokenizers 特有
- **尚未实施**: Ograg2 的改进尚未集成到当前管线中
- 属于未来优化方向，非当前阻塞项

## 构建配置

### 入口脚本
```bash
python scripts/build_kg_recursive.py
```

### 配置选项 (build_kg_recursive.py)
| 键 | 值 |
|----|-----|
| `SELECTED_OPTION` | `"AIOPS_New"` (当前) 或 `"Intel"` |
| `ROOT_MODEL` | `"qwen3:8b"` |
| `EXTRACT_MODEL` | `"qwen3:8b"` |
| `DOC_PATH` (Intel) | `/home/ritanlisa/下载/WinDownloads/Intel® 64 和 IA-32 架构软件开发者手册合集.pdf` |
| `DOC_PATH` (AIOPS_New) | `/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc` |
| `BATCH_CONCURRENCY` | `1` (递归管线保持串行以保证级联顺序) |
| `KG_KEEP_ALIVE` | `"3600s"` |

### 构建时间估算
- AIOPS_New (57 内容页): ~45 分钟 (含 Phase 0-3 + 去重 + 图聚合)
- Intel (4660 内容页): 未测试（极大文档，需要分批）

### 日志与输出
- 日志文件: `tmp/kg_builds/{DB_NAME}.log` (追加模式)
- MCP 操作详细日志: 每次 `sysml_add_entity/relation/merge` 记录完整参数
- LLM 交互调试日志: 每次 prompt/response 长度和时间
- 构建完成后 `.sysml` 文件路径: `database/{DB_NAME}/knowledge_graph.sysml`

### 相关 settings.yaml 键
```
KG_EXTRACTION_ENABLED, KG_EXTRACTION_MODEL, KG_LIGHT_MODEL,
KG_EXTRACTION_TEMPERATURE, KG_EXTRACTION_TIMEOUT,
KG_EXTRACTION_MAX_ITERATIONS, BATCH_CONCURRENCY,
KG_KEEP_ALIVE, PERSIST_DIR, OPENAI_API_KEY, OPENAI_API_URL
```

## 当前状态 (2026-06-26)

### 数据状态
- `database/AIOPS_New/` **为空** — 目录存在但无文件（Ubuntu 崩溃导致数据丢失）
- AIOPS_New 构建需要重新运行:
  ```bash
  # 编辑 scripts/build_kg_recursive.py, 修改 SELECTED_OPTION = "AIOPS_New"
  python scripts/build_kg_recursive.py
  ```
- 上次成功构建 (已丢失): 448 实体, 1166 关系, 149KB .sysml

### 代码状态
- `build_kg_recursive.py:28-29`: `exit(-1)` guard 已注释 (用 `#`)，构建可执行
- 所有代码修复均已提交，最新 commit: `5956ff1` (fix: always pass source_sections when creating relations)
- 递归管线 (`build_kg_recursive`) 为新标准，旧 `build_kg_from_document` 仍可用但不推荐
- Ollama 模型: `qwen3:8b` 用于 Phase 1/3 提取，`qwen3-vl:32b` 可用于 Phase 3 富化
