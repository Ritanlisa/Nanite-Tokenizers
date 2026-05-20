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
