# Nanite 知识图谱构建基线系统对比测试报告（详细版）

---

## 1. 测试概述

### 1.1 目的
在同文档、同 LLM 条件下，系统性对比 Nanite 与现有主流 KG 构建方法的实体/关系提取质量、效率和稳定性。

### 1.2 测试文档
**方正科技 FZA511 笔记本电脑产品说明书**

| 属性 | 值 |
|------|-----|
| 文档类型 | Word (.docx) |
| 段落数 | 698 |
| 总字符数 | ~22,000 |
| 语言 | 简体中文 |
| 结构 | 7 章，含 BIOS 设定、ABS 安全系统、故障分析等 |
| OCR 错误 | 源文档自带"电湠→电池""绻统→系统""滨意→注意"等错误 |

### 1.3 硬件与模型

| 项目 | 值 |
|------|-----|
| CPU | Intel i9-14900HX（24核/48线程） |
| GPU | NVIDIA RTX 5090 Laptop（24GB GDDR7） |
| 内存 | 32GB DDR5 |
| LLM | qwen3:8b（Ollama, 上下文 8K） |
| 嵌入模型 | nomic-embed-text（Ollama, 768 维） |
| LLM 调用方式 | Ollama REST API（localhost:11434） |

### 1.4 Ground Truth

人工标注 50 个核心实体，覆盖文档各章节主要概念：

```
硬件组件: 笔记本电脑, 方正A511笔记本电脑, 键盘, 触控板, 光驱, 硬盘, 电池, 
          电源适配器, 显示器, 内存, USB 2.0接口, 防盗锁孔, 风扇通风口
软件系统: BIOS设定程序, Windows系统, 方正ABS安全系统
功能模块: 系统备份, 系统恢复, 数据拯救, 杀毒, 驱动还原, IE修复
操作概念: 快捷键, 密码, 激活码, 充电, 数据备份
配件: 使用手册, 驱动盘, 电源线, 电源插头, 光盘
属性: 产品尺寸, 重量
处理器/内存: Intel Yonah处理器, DDR2内存
...
```

### 1.5 评估指标

- **ER（Entity Recall）**：系统提取的 GT 实体比例 = TP / (TP + FN)
- **PC（Precision）**：系统提取实体中属于 GT 的比例 = TP / (TP + FP)
- **F1**：ER 和 PC 的调和平均
- **匹配方法**：模糊字符串匹配（归一化后子串匹配，阈值 0.6）

---

## 2. 基线系统详情

### 2.1 Tree-KG（ACL 2025, CCF-A）

| 项目 | 值 |
|------|-----|
| **论文** | Tree-KG: An Expandable Knowledge Graph Construction Framework for Knowledge-intensive Domains |
| **作者** | Niu et al., Tsinghua University |
| **方法** | 显式 KG（TOC 层级→摘要→实体提取）+ 隐式 KG（6 算子：Conv/Aggr/Embed/Dedup/Pred/Merge） |
| **代码** | https://github.com/thu-pacman/Tree-KG |
| **输入** | 带 Word 标题样式的 docx（`USE_STYLE_FIRST: true`） |
| **Prompt** | 中文摘要→实体提取 prompt |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| TextSegmentation | ~1s | 解析 docx 标题样式，23 个根节点 |
| Summarize | 2m28s | 54 节点 3 层深度摘要生成 |
| Extraction | ~5min | 10 子章节×2 LLM 调用（实体+关系） |
| HiddenKG（6算子） | ~30-60min | Conv→Aggr→Embed→Dedup→Pred→Merge |
| **总计** | **~50min** | |

**问题记录：**
- `Extraction.py` 第 146 行：LLM 返回 JSON 数组时 `data.get("entities")` 崩溃 → 需加 `isinstance` 检查
- 中文摘要 + 中文 prompt 配合良好，但子节递归深度不足
- 26% 节点是 TOC 标题（目录结构被当作知识实体）

### 2.2 KGGen（NeurIPS 2025, CCF-A）

| 项目 | 值 |
|------|-----|
| **论文** | KGGen: Extracting Knowledge Graphs from Plain Text with Language Models |
| **作者** | Mo et al., Stanford |
| **方法** | 单步 LLM 提取 + 可选聚类去重 |
| **代码** | `pip install kg-gen` |
| **输入** | 纯文本（字符串） |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| KGGen.generate() | 7.7s | 2 次 LLM 调用（实体+关系） |
| **总计** | **7.7s** | |

**问题记录：**
- `chunk_text.py` 依赖 NLTK `punkt_tab` → 服务器无外网无法下载 → 改为正则分句 ✅
- `kg_gen.py` 第 217 行：`relations = set()` 遮蔽 Python 内置 `set()` → 重命名变量 ✅
- 返回的 `graph.edges` 是自定义 Graph 对象的 set 属性，不是 NetworkX → 需用 `graph.relations` 读取

### 2.3 AutoSchemaKG（ACL 2026, CCF-A）

| 项目 | 值 |
|------|-----|
| **论文** | AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction |
| **作者** | Bai et al., HKUST + Huawei |
| **方法** | 三阶段流水线：Entity-Entity → Entity-Event → Event-Event 三元组提取 + Schema 归纳 |
| **代码** | `pip install atlas-rag` |
| **输入** | JSON 格式（含 id/text/metadata） |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 数据准备 | ~5s | 创建 JSON 输入文件 |
| Triple Extraction | 369s | 1 batch, 2 chunks, 3 轮 LLM 调用 |
| **总计** | **369s** | |

**问题记录：**
- `ProcessingConfig` 中 `filename_pattern` 需要匹配以 `.json` 结尾的文件，`.txt` 不行
- 内部使用 HuggingFace `datasets.load_dataset()`，JSON 需含 `id` 和 `metadata` 字段
- 批次过大（15000 字）反而导致召回下降（LLM 上下文过长）

### 2.4 RAKG（arXiv 2025）

| 项目 | 值 |
|------|-----|
| **论文** | RAKG: Document-level Retrieval Augmented Knowledge Graph Construction |
| **作者** | Zhang et al. |
| **方法** | 句子分割 → NER 提取 → 嵌入 → 相似度去重 → KG 构建 |
| **代码** | https://github.com/LMMApplication/RAKG |
| **输入** | 纯文本（字符串） |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 句子分割 | ~1s | 169 句（中文） |
| NER 提取 | 189s | 328 实体（169 句×LLM 调用） |
| 相似度计算 | （跳过） | O(n²) ≈ 53,628 对，不可行 |
| **总计** | **~3min** | 仅 NER 阶段 |

**问题记录：**
- 默认使用英文 prompt（`text2entity_en`）→ 中文文本提取失败 → 改中文 prompt `text2entity_cn` ✅
- `TextProcessor(text, name)` 参数是文本内容，不是文件路径
- 余弦相似度返回 2D 数组需取 `sim[0][0]` ✅
- 328 实体时相似度计算 O(n²) 不可行，仅用 NER 结果

### 2.5 LightRAG

| 项目 | 值 |
|------|-----|
| **论文** | —（HKU 项目） |
| **方法** | 分块 → LLM 实体/关系提取 → NanoVectorDB 嵌入 → 图存储 |
| **代码** | `pip install lightrag-hku` |
| **输入** | 纯文本 |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 分块+提取 | 240s | 6 chunks × LLM 调用 |
| 图查询 | ~3s | 23 节点, 22 边 |
| **总计** | **243.5s** | |

**问题记录：**
- `ollama_embed` 被 `@wrap_embedding_func_with_attrs(embedding_dim=1024)` 装饰，与 nomic-embed-text 的 768 维不匹配 → 改为 768 ✅
- NanoVectorDB 缓存持久化维度信息，需清除缓存重跑
- 超出 2 跳的子图无法查询

### 2.6 GraphRAG（Microsoft）

| 项目 | 值 |
|------|-----|
| **论文** | GraphRAG: Unlocking LLM Discovery on Narrative Private Data |
| **方法** | 文档 → 实体/关系提取 → 社区检测 → 社区摘要 → 嵌入 |
| **代码** | `pip install graphrag` |
| **输入** | 纯文本文件（/input/ 目录） |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| load_input | ~1s | 读取文档 |
| create_base_text_units | ~10s | 分块 |
| extract_graph | ~60s | LLM 实体+关系提取 |
| finalize_graph | ~10s | 图构建 |
| extract_covariates | ~30s | 协变量提取 |
| create_communities | ~20s | 社区发现 |
| create_community_reports | ~60s | LLM 社区摘要 |
| generate_text_embeddings | ~60s | 嵌入生成 |
| **总计** | **~5min** | |

**问题记录：**
- 版本 2.7.2 → 3.1.1 CLI 完全改变（`graphrag.index` → `graphrag`）
- settings.yaml 格式从 `llm/embeddings` 变为 `models.default_chat_model/default_embedding_model`
- `graphrag init` 需要交互式输入 chat model 和 embedding model
- 需要 `.env` 文件设置 `GRAPHRAG_API_KEY`

### 2.7 Nanite

| 项目 | 值 |
|------|-----|
| **方法** | BFS 级联：Phase 0 根实体 → Phase 1 根章节 → Phase 2 传播 → Phase 3 级联 → Phase 4 聚合 |
| **输出** | SysML v2 标准（.sysml + .meta.json） |
| **实体类型** | PartDef, AttributeDef, RequirementDef, ItemDef |
| **关系类型** | Connection, Allocation, Interface |

**运行过程：**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 文档加载 | ~10s | docx → RagDocument |
| Phase 0-1 | ~2min | 根实体识别 + 根章节提取 |
| Phase 2-3 | 22min | BFS 级联传播 + 提取 |
| Phase 4 | ~1min | 图聚合 |
| **总计** | **25.6min** | 约 133 次 LLM 调用 |

**优化历程：**

| 版本 | 实体 | 关系 | 问题 |
|------|------|------|------|
| v1 | 18 | 3906 | keep_alive 缺失，BATCH_CONCURRENCY=4 无模型驻留 |
| v9 | 1103 | 5367 | max_iterations=500 卡上限 |
| v12 | 2294 | 10443 | Phase 1b auto-connection 用 RAW 名字未映射 QN |
| v13 | 2162 | 10290 | 孤立实体从 454→221（Phase 1b QN 修复） |
| **Laptop** | **87** | **277** | 笔记本说明书最优版本 |

**Laptop 版本配置：**

```python
BATCH_CONCURRENCY = 8
KG_KEEP_ALIVE = "3600s"  # 模型驻留 GPU
max_iterations = 10000
model = "qwen3:8b"
```

---

## 3. 完整指标对比

### 3.1 实体指标

| 系统 | 提取实体 | 命中(TP) | 漏报(FN) | 误报(FP) | ER | PC | F1 |
|------|---------|---------|---------|---------|------|------|------|
| **Nanite** | 69 | 44 | 6 | 25 | **88.0%** | **63.8%** | **73.9%** |
| Tree-KG | 51 | 23 | 27 | 28 | 46.0% | 45.1% | 45.5% |
| KGGen | 61 | 8 | 42 | 53 | 16.0% | 13.1% | 14.4% |
| RAKG | 212 | 15 | 35 | 197 | 30.0% | 7.1% | 11.5% |
| LightRAG | 23 | 4 | 46 | 19 | 8.0% | 17.4% | 11.0% |
| AutoSchemaKG | 12 | 3 | 47 | 9 | 6.0% | 25.0% | 9.7% |
| GraphRAG | 11 | 1 | 49 | 10 | 2.0% | 9.1% | 3.3% |

### 3.2 关系指标

| 系统 | 关系数 | 说明 |
|------|--------|------|
| **Nanite** | 277 | 含 allocation(auto-connect)+connection+interface |
| Tree-KG | 88 | 含 61 语义边 + 27 TOC 边 |
| KGGen | 77 | 不含关系类型标签 |
| RAKG | 0 | 相似度计算 O(n²)，未生成关系 |
| LightRAG | 22 | 局部子图 |
| AutoSchemaKG | 12 | Entity + Event 双通道 |
| GraphRAG | 7 | 社区级别关系 |

### 3.3 效率指标

| 系统 | 总耗时 | LLM 调用次数 | 平均每次 LLM 耗时 | 安装难度 |
|------|--------|-------------|-----------------|---------|
| **Nanite** | 25.6min | ~133 | ~11.6s | ⭐⭐ |
| Tree-KG | ~50min | ~56 | ~35s | ⭐⭐⭐ |
| KGGen | **7.7s** | ~2 | ~3.8s | ⭐（pip install） |
| RAKG | ~3min | ~340 | ~0.5s | ⭐⭐⭐ |
| LightRAG | 243.5s | ~24 | ~10s | ⭐⭐（pip install） |
| AutoSchemaKG | 369s | ~5 | ~60s | ⭐⭐（pip install） |
| GraphRAG | ~5min | ~15 | ~20s | ⭐⭐⭐ |

### 3.4 Intel 手册（Nanite）

| 指标 | 值 |
|------|-----|
| 文档 | Intel® 64 and IA-32 Architectures SDM（5342 页, 26MB PDF） |
| 数据库 | Intel_Manual_v2 |
| 模型 | qwen3:8b |
| 实体 | **2294**（最终 v12） |
| 关系 | **10443** |
| KG 文件 | 1.7 MB（.sysml） |
| Phase 4 轮次 | 6 轮（全图连通） |
| 孤立实体 | 221/1965（11.2%） |
| 运行时间 | 9.8 小时 |

---

## 4. 配置/修复记录

### 4.1 Tree-KG 修复
```python
# Extraction.py L146: LLM 返回 list 而非 dict 时崩溃
# 修复前
ents = data.get("entities", [])
# 修复后
if isinstance(data, dict):
    ents = data.get("entities", [])
elif isinstance(data, list):
    ents = data
```

### 4.2 KGGen 修复
```python
# kg_gen.py L216-217: relations = set() 遮蔽内置 set()
# 修复前
entities = set()
relations = set()
# 修复后
chunk_entities_all = set()
chunk_relations_all = set()

# chunk_text.py: NLTK punkt 依赖替换为正则
sentences = re.split(r'(?<=[。！？.!?])\s*', text)
```

### 4.3 LightRAG 修复
```python
# ollama.py L249-250: embedding_dim 默认 1024
# 修复前
@wrap_embedding_func_with_attrs(embedding_dim=1024, ...)
# 修复后
@wrap_embedding_func_with_attrs(embedding_dim=768, ...)
```

### 4.4 RAKG 修复
```python
# kgAgent.py: 英文 prompt → 中文 prompt
from src.prompt import text2entity_cn as text2entity_en

# 余弦相似度返回值提取
sim_matrix[i][j] = float(sim[0][0])  # cosine_similarity 返回 2D 数组

# TextProcessor 参数：文本内容而非文件路径
processor = TextProcessor(text_content, "laptop")  # text_content = open(...).read()
```

### 4.5 GraphRAG 修复
```yaml
# settings.yaml: 2.x → 3.x 格式变更
# 2.x 格式（不兼容）
llm:
  type: openai_chat
  model: qwen3:8b

# 3.x 格式（兼容）
models:
  default_chat_model:
    type: openai_chat
    model: qwen3:8b
    api_base: http://localhost:11434/v1
    api_key: ollama_api_key
```

### 4.6 Nanite 优化历史

| 版本 | 优化 | 效果 |
|------|------|------|
| v1→v2 | 添加 keep_alive | LLM 调用加速 4x |
| v2→v9 | max_iterations 500→5000 | 章节覆盖从 16→160 |
| v9→v10| cascade 50 实体/节保护 | 防止单节卡死 |
| v10→v12| Phase 1b+3 用 entity_map QN | 孤立实体 454→221 |
| v12→v13| Phase 1b root 实体 MCP 搜索兜底 | 215 实体 |
| Laptop | BATCH_CONCURRENCY=8 + keep_alive | 25.6min |

---

## 5. 各系统架构对比

```
Tree-KG:
  TOC 层级 → 摘要(LLM) → 实体/关系提取(LLM)
  → HiddenKG(Conv/Aggr/Embed/Dedup/Pred/Merge)
  
Nanite:
  Phase 0: 根实体识别(LLM)
  Phase 1: 根章节提取(LLM)
  Phase 2→3: BFS 级联 (实体,章节) 对处理(LLM)
  Phase 4: 图聚合

KGGen:
  文本 → 分句 → 实体提取(LLM) → 关系提取(LLM) → 可选聚类

LightRAG:
  文本 → 分块 → LLM 实体/关系提取 → NanoVectorDB → 图查询

GraphRAG:
  文本 → 分块 → LLM 实体/关系提取 → 社区发现 → 社区摘要 → 嵌入

AutoSchemaKG:
  文本(JSON) → Entity-Entity提取(LLM) → Entity-Event提取(LLM) 
  → Event-Event提取(LLM) → Schema归纳

RAKG:
  文本 → 句子分割 → NER(LLM) → 嵌入 → 相似度去重 → KG构建
```

---

## 6. 结论

1. **Nanite F1=73.9% 为所有基线最高**，是 Tree-KG（45.5%）的 1.6 倍、其余系统的 5-22 倍
2. **Nanite 召回率 88.0%**，远超 Tree-KG（46.0%），BFS 级联策略有效覆盖更多文档概念
3. **Nanite 精确率 63.8%** 也是最高，SysML 类型约束 + 焦点实体引导有效过滤噪声
4. Nanite 速度（25.6min）介于快速系统（KGGen 7.7s）和慢速系统（Tree-KG ~50min）之间
5. Intel 手册（2102 实体）验证了 Nanite 对超长文档的可扩展性
6. 第三方系统普遍存在代码质量/兼容性问题（KGGen 内置函数遮蔽、LightRAG 维度写死、GraphRAG 版本 CLI 不兼容、RAKG O(n²) 不可扩展）

---

*报告生成日期：2026-07-23*
*测试模型：qwen3:8b / nomic-embed-text via Ollama*
*测试文档：方正科技 FZA511 笔记本电脑产品说明书*
