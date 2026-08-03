# Nanite 设计分析与跨数据集 GT 比对说明

> 本文档两部分内容：
> 1. **Nanite 项目设计汇总** —— 基于实际代码（`agent/kg_build_agent.py`、`sysml/sysml_manager.py`、`rag/`、`config.py`）归纳的 5 个设计要点 + benchmark 数据集 GT 信息
> 2. **跨数据集 GT 比对分析（方向六）** —— 对"方向六"分析报告的逐条解读
>
> 说明：项目根目录的 `Nanite基线测试报告_详细版.md` 可能已过时（部分数字与代码/最新实验结果不一致），本文档以代码为准。

---

# 第一部分：Nanite 项目设计汇总

## 1. 本体格式设计（限制 LLM 输出）

**输出格式**：SysML v2 标准（`.sysml` 纯文本 + `.meta.json`），通过 MCP 工具集落库。

**实体类型（LLM 输出 JSON 数组）**：

```json
{"type":"PartDef","name":"FT计算柜","description":"...","aliases":["..."]}
```

- `PartDef`：系统组件/模块/设备/子系统
- `AttributeDef`：属性/参数/指标（如"带宽400Gbps"）
- `PortDef`：接口/端口/连接点
- `ItemDef`：数据结构/信息流
- `RequirementDef`：需求/约束
- `CommandDef`：Shell 命令/CLI 工具（如 yhst, smu_tranfer_cmd）

**关系类型（18 种 SysML 标准）**：

`Connection`（双向物理/数据流）、`Interface`、`Allocation`、`Containment`、`Composition`、`Reference`、`Generalization`、`Dependency`、`Abstraction`、`Realization`、`Derive`、`Trace`、`DeriveReqt`、`Refine`、`Satisfy`、`Verify`、`Copy`、`UseCaseAssociation/Include/Extend`

**限制手段**：

- `EXTRACTION_CANDIDATES_PROMPT`：严格 JSON 格式约束（无内容输出 `[]`）
- `KG_EXTRACTION_TEMPERATURE=0.1`（低温保证一致性）
- 实体搜索置信度 `>=0.7` 更新、`<0.7` 新建（阈值由 `KG_MERGE_CONFIDENCE_THRESHOLD=0.7` 控制）
- 关系创建前必须验证端点实体存在

## 2. 多模态输入处理

| 模态 | 处理方式 | 代码 |
|---|---|---|
| **Word/PDF** | LibreOffice UNO 按页抽取文本（`_extract_office_text_by_uno_pages`），PDF 用 PyMuPDF | `rag/documents.py`, `document_docx/pdf.py` |
| **图片（文档内插图）** | `ImageAsset` 提取 → **OCR 管线**：`OCR_API_URL` + `OCR_MODEL`（配置为 qwen3-vl 视觉模型）→ `ocr_text` 字段并入文档文本 | `rag/ocr.py` |
| **扫描版 PDF** | `OCRPDFReader`：dpi=200 渲染 → 检测 has_images → 视觉模型 OCR | `rag/ocr.py:140` |
| **表格/电子表格** | `document_spreadsheet.py` 支持，页面布局合并去重（`_merge_page_layouts`） | `rag/documents.py` |
| **监控指标数据** | Prometheus 客户端指标（`monitoring.py`：Counter/Histogram + `/metrics` 8000 端口）——这是**输出监控**，非输入 | `monitoring.py` |
| **软件日志数据** | `log_dataset.py` 是**训练用的模拟日志 Dataset**（ECC 内存错误格式），非 KG 输入管线；`CommandDef` 实体类型专门覆盖日志/命令中的运维操作 | `src/.../data/log_dataset.py` |

⚠️ **注意**：当前 benchmark 跑的是纯文本 NER 数据集（resume/cluener/msra），多模态能力（OCR/表格）在 Nanite 主项目代码中，但 benchmark 脚本 `script_nanite` 只用了纯文本分批提取。

## 3. 领域 RAG 增强实体抽取

**"RAG 增强"体现在两个层面**：

1. **文档导航式 RAG**（`UNIFIED_EXTRACTION_SYSTEM_PROMPT`）：Agent 自主导航文档树——`get_document_tree` → `read_section` → 逐页提取 → `mark_section_done`，从整体到局部
2. **实体检索增强（search-before-create）**：每个候选实体先 `sysml_search_entity` 查重（6 层匹配策略），再决定新建/更新/补别名——相当于**用图库历史状态作为检索增强上下文**
3. **RAG 引擎**（`rag/engine.py`）：llama_index + FAISS 向量库 + `SIMILARITY_TOP_K=5` + cross-encoder 重排（`RERANK_TOP_N=3`），用于 Chat 查询侧（非抽取侧）
4. **长文本滑动窗口**：>4000 字符分块（window=3000, stride=2500 重叠 500），分块提取后合并去重

## 4. 隐含关系推理

**`_bridge_components`（Phase 4 桥接）**：

- 找出**共同出现在同一章节**的实体对（`shared_sections`）
- 并行 LLM 分类：给定 实体A/实体B/共同章节 → 输出 19 种关系类型之一（含 `none`）
- 答案非 none → 创建桥接关系（`num_predict=8` 限制输出，快）
- 每批最多 20 对，`BATCH_CONCURRENCY` 并发

**`_enrich_entities`（Phase 3 富化）**：

- 按章节分组实体 → LLM 输出 JSON 富化指令（`add_relation`/`add_alias`/`update_entity`）
- 硬性规则："**每个实体至少创建 1 条关系**，禁止孤立实体"，同章节优先
- 属性/端口类实体强制关联到所属 PartDef

**Phase 4 `_aggregate_graph`**：图聚合 + 连通性检查。

## 5. 实体与关系校准和消歧

**多策略搜索匹配（`AliasRegistry.search` + `search_entities`）**：

```
精确匹配(1.0) → 归一化匹配(0.95) → 别名匹配(0.90) → 子串(0.70) → 正则(0.80) → Token重叠(0.60)
```

**跨章节去重（`_deduplicate_entities` → `suggest_merges`）**：

- 启发式打分：同 QN=1.0 / 同 name 不同包=0.95 / 名称子串=0.7 / 别名匹配名称=0.85 / 共享别名=0.9
- 仅同类型实体对比较，阈值 0.5（build plan 建议 0.7）
- `merge_entities`：合并 metadata + 转移别名 + **重定向所有关系引用**（通过 source 全部引用形式匹配）+ 删除源实体

**别名体系**：每个实体注册多语言别名，`AliasRegistry` 全局维护（normalize 处理大小写/空格/标点）。

---

## 数据集 GT 信息（本次 benchmark）

**数据来源**（重建，已在 `/home/hjq/benchmark_data/`）：

| 数据集 | 来源 | 文本 | GT 实体数 | GT 构建方式 |
|---|---|---|---|---|
| **resume** | ResumeNER（`ttxy/resume_ner`）**test split** | 15576 字符 / 477 行 | **889** | 从 BIO 标注提取所有实体文本（NAME/CONT/EDU/TITLE/ORG/RACE/PRO/LOC） |
| **cluener** | CLUENER2020（`nlhappy/CLUE-NER`）**validation split** | 51602 字符 / 1343 行 | **2181** | 从 `ents` 字段（indices/text/label）提取实体文本 |
| **msra** | MSRA（`PassbyGrocer/msra-ner`）**test split** | 176965 字符 / 4365 行 | **2093** | 从 BIO 标注（B-LOC/I-LOC/B-ORG/I-ORG/B-PER/I-PER）提取实体 |

**验证方法**：

- resume 文本 MD5 = `af8523fa22d5adb149321a731d8dea06`，与旧日志 lightrag doc-id **完全一致**（铁证）
- GT 规模交叉验证：从旧 rerun 日志 P/R/F1 反推 → resume ~890 / cluener ~2182 / msra ~2030，与重建值误差 <0.1%（msra 3% 差异因数据集版本）
- GT 实体 100% 出现在输入文本中

**GT 是实体名集合（`set`）**，无实体类型标签；评估 = 精确字符串匹配（`s & g` 交集），非模糊匹配。这与 Nanite 主项目（laptop 文档 50 实体人工标注、模糊匹配 0.6）不同——**本次 benchmark 是 3 个标准 NER 数据集的纯文本实体抽取对比**。

---

# 第二部分：跨数据集 GT 比对分析（方向六）逐条解读

> 这是对 Nanite 在 3 个标准 NER 数据集上的**跨数据集对比分析**——通过看 Nanite 在不同特点数据集上的表现差异，反推它方法上的优缺点。

## 2.1 现状表格（读法）

| 列 | 含义 |
|---|---|
| **类型数** | 该数据集标注了几种实体类别。CLUENER=10 类（人名/地名/公司/职位/地址/电影/游戏/政府/书名/景点），MSRA=3 类（仅人名/地名/组织名），RESUME=8 类（人名/职称/学历/专业/国籍等） |
| **GT 实体** | 标准答案里实体的**总数**（"应该找出来多少个"），2181/2030/889 |
| **Nanite F1** | Nanite 方法在该数据集上的 F1（精确率×召回率的调和平均，兼顾"提得准不准"和"提得全不全"） |
| **排名** | 与其余 6 个基线（KGGen/RAKG/LightRAG 等）比，Nanite 排第几（🥇第一 🥈第二） |
| **特征** | 数据集的领域属性，用来解释表现差异 |

**一句话**：同一种方法打三个不同风格的"靶子"，看哪个打得准、哪个打得偏。

## 2.2 四条关键分析（每条都是"现象 → 推论"）

**① 类型多反而 F1 高 → 短板在召回不在分类**

直觉上类型越多越难分，F1 应该越低。但 CLUENER 10 类反而 F1 最高（0.598）——说明 Nanite 的**分类能力没问题**（否则类型多的数据集会因分错类大量扣分）。既然分类没拖后腿，那 F1 不如意的原因只能出在**召回**（该找的实体没找全）。

**② MSRA 上 KGGen 召回更高 → 分块粒度差异**

KGGen 召回 0.548 > Nanite 0.450。原因：KGGen 按 3000 字符小 chunk **逐块单独调 LLM**，每块上下文干净聚焦，实体不容易漏；Nanite 每 5 段合并一次调用，**多个段落混在一个上下文里**，句子互相干扰，部分实体被 LLM 忽略。→ 结论：Nanite 的分块粒度偏粗，牺牲了召回。

**③ RESUME 上 RAKG 精确率更高 → prompt 专精度差异**

RAKG 精确率 0.857 > Nanite 0.732。原因：RAKG 是**逐句 NER**（一句一调，prompt 专为"提取实体"优化）；Nanite 是通用提取 prompt（实体+关系一起抽）。简历文本高度结构化（"XX，男，XX 年出生，XX 职称"），通用 prompt 容易**过度发散**——把不是 GT 实体的内容也提出来 → FP 变多 → 精确率下降。→ 结论：Nanite 的通用 prompt 在结构化文本上不够收敛。

**④ MSRA 只标 3 类 → 多提的类型全是 FP**

MSRA 的 GT 只有 PER/ORG/LOC 三类。Nanite 若提取出 GT 之外的类型（比如时间、数值、职位），评估时**永远无法匹配 GT** → 全部记入 FP。所以"限制 LLM 输出类型"（方向一）在这种 GT 结构下**直接有效**：约束输出 = 从源头减少这类 FP。

## 2.3 落地方案（下一步要做什么）

| 方案 | 大白话 |
|---|---|
| **① 类型级混淆矩阵** | 现在只看总 F1，看不出"哪类被错分成了哪类"。输出一个 (预测类型 × 真实类型) 矩阵，比如发现"公司→组织"错分密集，就针对这类改 prompt |
| **② 按 GT 结构自适应 prompt** | 知道 GT 有哪些类型时，把类型列表直接写进提取 prompt（约束输出）；不知道时保持开放提取，事后把结果映射到 GT 类型 |
| **③ 嵌套实体评估** | CLUENER/RESUME 存在嵌套标注（如"北京市朝阳区"内含"北京市"）。精确字符串匹配对嵌套极不友好——专门测 Nanite 对嵌套实体的处理能力，量化"标注粒度"对 F1 的影响 |
| **④ 跨数据集迁移实验** | 同一个 prompt 打 3 个数据集，看表现差异多大。差异小 → 一套通用 prompt 就够；差异大 → 值得为每个数据集维护特化 prompt |

## 2.4 预期收益 & 验证方式

- **类型级 F1 热力图** → 直观看到哪类实体最弱，指导 prompt 往哪个方向特化
- **prompt 通用性曲线** → 决策依据：要不要维护多套 prompt（维护成本 vs 收益）
- **嵌套实体报告** → 判断 F1 损失是"方法不行"还是"评估方式吃亏"
- 验证闭环：混淆矩阵附录可查证 + 特化前后 F1 对比证明改进真实有效

## 2.5 数据一致性提醒

⚠️ 方向六分析中的数字（CLUENER 0.598 / MSRA 0.452 / RESUME 0.490，KGGen R=0.548，RAKG P=0.857）**与 rerun_missing.log 的实际结果不一致**（那里的 nanite 是 cluener 0.646 / resume 0.678 / msra 0.483）。该分析可能基于另一轮实验或另一份报告数据——引用前应先核对数字来源，避免分析结论建立在错误数字上。
