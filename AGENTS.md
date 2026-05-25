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
        → 统一 Agent (LangChain) + MCP 工具 + 导航工具
        → 自主导航提取 + 打勾
        → 跨章节去重 → 保存 .sysml
```

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

### KG 可视化前端 (2026-05 新增)
- 访问: `GET /kg/viz/{db_name}` (如 `/kg/viz/AIOPS_New`)
- 数据源: `GET /api/rag/dbs/{db_name}/kg/graph`
- 技术栈: cytoscape.js CDN + cose-bilkent 布局
- 功能: 类型筛选、搜索、连通分量统计、孤立节点标记、节点详情面板
- 启发式边恢复: 当 relation.ends 为空时，从命名约定解析 (A_B_Type → A→B 边)，覆盖率 96% (188/195)
