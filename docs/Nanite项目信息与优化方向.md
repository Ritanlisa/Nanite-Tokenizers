# Nanite 项目信息汇总与优化方向

> 本文档基于 **实际代码** 整理（`agent/kg_build_agent.py`、`sysml/`、`rag/`、`config.py`、`settings.yaml`、`AGENTS.md`、`scripts/`），并附最新 benchmark 结果与优化方向分析。
> 更新日期：2026-08-03

---

# 第一部分：项目架构全貌

## 1. 顶层结构

| 目录/文件 | 职责 |
|---|---|
| `web_server.py` | **主入口** — FastAPI Web 服务器 + 静态 UI + KG 可视化 API |
| `main.py` | Gradio 入口（备用） |
| `agent/kg_build_agent.py` | **KG 构建核心**（2895 行）：DocumentTreeState、导航工具、统一提取 Agent |
| `sysml/sysml_manager.py` | 内存模型管理器：实体/关系 CRUD、别名注册表、合并去重（929 行） |
| `sysml/sysml_model.py` | SysML v2 AST：PartDef/AttributeDef/ConnectionUsage 等（738 行） |
| `sysml/sysml_parser.py` | .sysml 文本解析器（388 行） |
| `sysml/hv_resolver.py` | 高价值实体解析器（400 行） |
| `scripts/sysml_rag_mcp_server.py` | **MCP 服务器**：SysML CRUD 工具（30+），stdio JSON-RPC |
| `mcp_client/` | 通用 MCP stdio 会话管理器 + 工具包装器 |
| `rag/` | RAG 引擎（llama_index + FAISS/Chroma）、文档加载、OCR、关键词提取 |
| `rag/documents.py` | 文档加载：LibreOffice UNO (.doc/.docx)、PyMuPDF (.pdf)、表格 |
| `rag/ocr.py` | 图像 OCR 管线（视觉模型） |
| `rag/tfidf_keyword_extractor.py` | TF-IDF 关键词提取 |
| `rag/logprob_keyword_extractor.py` | logprobs 关键词提取（jieba 分词 + logprob 筛选） |
| `monitoring.py` | Prometheus 指标（Counter/Histogram，/metrics 8000 端口） |
| `config.py` | pydantic-settings 配置（优先级：init > env > YAML > secret） |
| `settings.yaml` | 实际生效配置 |
| `database/` | KG 数据库（Intel_Manual_v2、Laptop_Manual） |
| `docs/` | 设计文档（kg-build-plan、Nanite设计分析与GT比对说明） |
| `src/nanite_tokenizers/` | 训练/推理包（training/simplier、tools/download_tokenizer、data/log_dataset） |

## 2. 关键运行配置（settings.yaml 实际值）

```yaml
LLM_MODEL: qwen3.6:27b          # 主模型（Phase 3 富化用强模型）
EMBED_MODEL: bge-m3              # 嵌入模型（注意：benchmark 用的是 nomic-embed-text）
OPENAI_API_URL: http://localhost:11434/v1
OPENAI_API_KEY: ollama_api_key
TEMPERATURE: 0.0
ENABLE_RAG: true
ENABLE_RERANK: false
OFFLINE_ONLY: true
KG_EXTRACTION_ENABLED: false     # ⚠️ 生产配置中 KG 构建未启用！
KG_MCP_SERVER_COMMAND: "python scripts/sysml_rag_mcp_server.py serve"
```

config.py 中 KG 相关默认值：
- `KG_EXTRACTION_MODEL: qwen3:8b`（默认，实际可用 qwen3.6:27b）
- `KG_LIGHT_MODEL: qwen3:8b`（轻量提取模型）
- `KG_KEEP_ALIVE: 30s`（模型驻留时间，**Laptop 优化版用 3600s**）
- `KG_EXTRACTION_TEMPERATURE: 0.1`
- `KG_EXTRACTION_MAX_ITERATIONS: 10`
- `KG_MERGE_CONFIDENCE_THRESHOLD: 0.7`
- `BATCH_CONCURRENCY: 5`（Laptop 版 8）
- `OCR_MODEL: qwen3-vl`（视觉 OCR）
- `SIMILARITY_TOP_K: 5` / `RERANK_TOP_N: 3`

## 3. KG 构建流程（KGBuildAgent）

```
RAG_DB_Document (文档树)
  → DocumentTreeState 构建层次树（章节/页面）
  → Phase 1a: 并行 light_llm 提取实体/关系候选（Semaphore=BATCH_CONCURRENCY）
       · EXTRACTION_CANDIDATES_PROMPT（JSON 数组输出）
       · >4000 字符滑动窗口分块（window=3000, stride=2500 重叠500）
  → Phase 1b: 顺序 MCP 工具创建（search-before-create 去重 + 创建/更新/别名）
  → Phase 2: 跨章节去重（suggest_merge，无 LLM，<1s）
  → Phase 3: 卸载小模型 → 强模型富化（ENRICHMENT_JSON_PROMPT，add_relation/add_alias/update_entity）
       · 硬性规则：每个实体至少 1 条关系，禁止孤立
  → Phase 4: 图聚合 + _bridge_components（LLM 分类共同出现实体对的关系）
  → 保存 .sysml + .meta.json
```

并行架构要点（AGENTS.md 2026-05）：
- Phase 1a 用 `asyncio.gather` + `Semaphore(BATCH_CONCURRENCY)` 并行
- 所有 LLM 调用先提交（消除 MCP 处理导致请求间延迟），再顺序处理 MCP 操作
- `keep_alive=30s` 空闲自动卸载；Phase 2 后显式 `keep_alive=0` 卸载小模型，为强模型腾显存

---

# 第二部分：五大设计要点（基于代码）

## 1. 本体格式设计（限制 LLM 输出）

**输出**：SysML v2（`.sysml` + `.meta.json`），MCP 工具落库。

**实体类型**（LLM 输出 JSON）：
```json
{"type":"PartDef","name":"FT计算柜","description":"...","aliases":["..."]}
```
PartDef（组件/模块/设备）· AttributeDef（属性/参数/指标）· PortDef（接口/端口）· ItemDef（数据结构）· RequirementDef（需求/约束）· CommandDef（Shell 命令/CLI）

**关系类型（18 种 SysML）**：Connection（双向）/Interface/Allocation/Containment/Composition/Reference/Generalization/Dependency/Abstraction/Realization/Derive/Trace/DeriveReqt/Refine/Satisfy/Verify/Copy/UseCaseAssociation/Include/Extend

**限制手段**：
- EXTRACTION_CANDIDATES_PROMPT 严格 JSON 格式（无内容输出 `[]`）
- 温度 0.1；`num_predict` 限制（bridge 用 8）
- 搜索置信度 ≥0.7 更新 / <0.7 新建
- 关系创建前验证端点实体存在（缺失自动创建 PartDef）

## 2. 多模态输入处理

| 模态 | 处理方式 | 代码 |
|---|---|---|
| Word/PDF | LibreOffice UNO 按页抽文本（.doc 需 UNO）、PyMuPDF (.pdf) | `rag/documents.py` |
| 文档内图片 | ImageAsset 提取 → OCR 管线（qwen3-vl 视觉模型）→ ocr_text 并入文本 | `rag/ocr.py` |
| 扫描 PDF | OCRPDFReader：dpi=200 渲染 → 检测 has_images → OCR | `rag/ocr.py:140` |
| 表格 | document_spreadsheet.py + 页面布局合并去重 | `rag/documents.py` |
| 监控指标 | Prometheus 输出指标（非输入） | `monitoring.py` |
| 软件日志 | log_dataset.py 是训练用模拟数据；CommandDef 类型覆盖日志命令 | `src/.../data/log_dataset.py` |

⚠️ benchmark 仅用纯文本 NER 数据集，多模态能力未参与本次对比。

## 3. 领域 RAG 增强实体抽取

- **文档导航式**：Agent 自主 get_document_tree → read_section → 提取 → mark_section_done（从整体到局部）
- **search-before-create**：候选实体先 sysml_search_entity 查重（6 层匹配），决定新建/更新/补别名——图库历史状态即检索上下文
- **RAG 引擎**：llama_index + FAISS/Chroma + cross-encoder 重排（用于 Chat 查询侧，非抽取侧）
- **关键词提取**：TF-IDF 与 logprobs（jieba 分词）双通道

## 4. 隐含关系推理

- **_bridge_components**：共同出现章节的实体对 → LLM 分类 19 种关系/none → 建边（num_predict=8，每批 20 对）
- **_enrich_entities**：按章节分组 → LLM 输出 JSON 富化指令；硬规则"每实体至少 1 关系"
- **Phase 4 _aggregate_graph**：图聚合 + 连通性

## 5. 实体与关系校准和消歧

- **搜索匹配 6 层**：精确(1.0) → 归一化(0.95) → 别名(0.90) → 子串(0.70) → 正则(0.80) → Token重叠(0.60)
- **suggest_merges 启发式**：同 QN=1.0 / 同 name 不同包=0.95 / 名称子串=0.7 / 别名匹配=0.85 / 共享别名=0.9；仅同类型比较
- **merge_entities**：名称→别名、转移别名、关系重定向（全引用形式）、metadata 合并、删除源
- **别名体系**：AliasRegistry 全局维护，normalize 处理大小写/空格/标点

---

# 第三部分：Benchmark 数据集与 GT

数据在 `/home/hjq/benchmark_data/`（从公开数据集重建，原 /tmp 数据因重启丢失）：

| 数据集 | 来源（HF） | 文本 | GT 实体 | 构建方式 |
|---|---|---|---|---|
| resume | ttxy/resume_ner **test** | 15576 字符/477 行 | 889 | BIO 标注提取（8 类） |
| cluener | nlhappy/CLUE-NER **validation** | 51602 字符/1343 行 | 2181 | ents 字段提取（10 类） |
| msra | PassbyGrocer/msra-ner **test** | 176965 字符/4365 行 | 2093 | BIO 标注提取（3 类） |

**验证**：
- resume MD5 = `af8523fa22d5adb149321a731d8dea06` = lightrag 日志 doc-id（铁证）
- 反推 GT：resume ~890 / cluener ~2182 / msra ~2030（与重建 <0.1% 差）
- GT 实体 100% 在文本中；评估 = 精确字符串匹配（set 交集）

**结果目录**：`/home/hjq/benchmark_results/`（持久，避免再丢）

---

# 第四部分：优化方向分析

> 来源：跨数据集 GT 比对分析 + 本次重跑观察 + 代码审查

## 方向一：类型约束（限制 LLM 输出类型）

**依据**：MSRA GT 只有 PER/ORG/LOC 三类，Nanite 开放提取多出的类型全部记为 FP。CLUENER 10 类反而 F1 最高（0.598）→ **短板在召回不在分类**。

**落地**：已知 GT 类型枚举时 → 提取 prompt 内嵌类型列表；未知 → 开放提取 + 事后类型映射。

## 方向二：分块粒度优化（★ 本次重跑已暴露）

**依据**：KGGen（每 3000 字独立调用）R=0.548 > Nanite（每 5 段合并）0.450。合并调用上下文注意力稀释，召回下降。

**落地**：评估 batch=1/2/3/5/10 的 F1 曲线；或两阶段（独立提取 → 统一去重消歧，复用 suggest_merge）。

## 方向三：Prompt 特化 vs 通用

**依据**：RAKG（逐句 NER 专精 prompt）P=0.857 > Nanite 0.732。通用提取 prompt 在结构化简历上过度发散。

**落地**：跨数据集迁移实验，判断是否值得为每个数据集维护独立 prompt。

## 方向四：嵌套实体评估

**依据**：CLUENER/RESUME 有嵌套标注（"北京市朝阳区"内含"北京市"），精确字符串匹配对嵌套不友好。

**落地**：嵌套实体报告，量化标注粒度对 F1 影响。

## 方向五：跨数据集 GT 比对（即"方向六"分析）

**落地**：类型级混淆矩阵（预测×真实）定位 type confusion 热点；特化前后 F1 对比验证。

## 方向六（本次观察）：LLM 输出稳定性

**依据**：lightrag cluener 重跑失败——qwen3:8b 返回**中文叙述**（"以下是您提供的内容的整理..."）而非 entity 格式，lightrag 解析器无法处理 → 0 实体。这是**模型输出格式漂移**问题（非脚本 bug）。

**落地**：
- 增加输出格式校验与重试（检测非 JSON/非 entity 格式 → 重试或降级）
- lightrag 场景：增大 worker timeout（480s→900s）+ 减少并发（max_async 4→2）降低超时概率
- 脚本侧：`asyncio.run(main())` 后 `os._exit(0)` 防止非 daemon 线程挂起（已修复）

## 方向七（本次观察）：环境稳定性与可复现性

**依据**：
1. Ollama 共享服务被其他用户大模型占用 → qwen3:8b 被踢出 GPU → 全部超时（lightrag×3 各浪费 4h）
2. 修复：`CUDA_VISIBLE_DEVICES=0` 固定 GPU0 + `OLLAMA_KEEP_ALIVE=30m`（已改 /etc/systemd/.../override.conf）
3. 结果目录在 /tmp 被重启清空 → 已迁移 /home/hjq

**落地**：
- benchmark 脚本增加 Ollama 健康检查前置（/api/ps 确认模型可加载）
- 每项完成后立即落盘（已有）+ 断点续跑（已有 rerun_missing.py）
- 监控 GPU 占用，发现异常自动暂停

## 方向八：配置一致性审查

**观察**：
- `KG_EXTRACTION_ENABLED: false`（生产未启用 KG 构建！）
- `LLM_MODEL: qwen3.6:27b` vs KG 默认 `qwen3:8b`——两套模型配置
- `EMBED_MODEL: bge-m3` vs benchmark 用 nomic-embed-text
- `KG_KEEP_ALIVE: 30s` vs Laptop 优化版 3600s（差异巨大）

**落地**：统一配置管理，明确生产/benchmark 两套 profile。

---

# 附录：已知 Bug 与踩坑（AGENTS.md 节选）

1. **Relation ends 序列化丢失**（2026-05）：有名字的 connection 调父类 to_text() 不输出 ends → 已修复 sysml_model.py:328 + parser.py:172；**历史 KG 文件 ends 全丢需重建**
2. **_entity_type_name() isinstance 顺序**：Allocation/InterfaceUsage 继承 ConnectionUsage 但 dict 顺序在前者 → 误判 → 已修复（子类型排前）
3. **KG 可视化**：/kg/viz/{db_name}，cytoscape.js，启发式边恢复覆盖率 96%（188/195）
