# Nanite-Tokenizers 中文说明

## 目标
- 训练、推理、工具分模块，结构清晰。
- 复用型包结构，主代码位于 `src/`。
- 保持 legacy 脚本兼容。

## 项目结构
- `src/nanite_tokenizers/cli.py`：统一 CLI 入口
- `src/nanite_tokenizers/training/`：训练相关
- `src/nanite_tokenizers/inference/`：推理与 demo
- `src/nanite_tokenizers/models/`：模型与压缩
- `src/nanite_tokenizers/data/`：数据集
- `src/nanite_tokenizers/tools/`：下载工具
- `src/nanite_tokenizers/utils/`：通用工具

## 快速开始
- 环境安装：`uv sync`
- Demo 运行：`python -m nanite_tokenizers demo`
- 训练：`python -m nanite_tokenizers train --model-index 0`
- 下载分词器：`python -m nanite_tokenizers download`

## 传统入口
- `simplier.py`、`download.py`、`compressor.py` 作为兼容壳保留。

## RAG + MCP Agent
- 根目录新增：
  - `config.py`：配置与校验
  - `rag/`：RAG 引擎、预处理、向量库
  - `mcp/`：MCP 抓取客户端
  - `agent/`：工具、记忆、agent 执行器
  - `tests/`：pytest 示例

## Agent 技能
- 默认内置技能：
  - `skill_shell`：安全 shell 命令
  - `skill_file_io`：读写/列目录/建目录（限工作区）
  - `skill_search`：基于 Selenium 的可配置网页搜索（URL/XPath/Regex）
  - `skill_web_visit`：基于 Selenium 的网页访问与正文提取（适合 JS 渲染页面）
  - Copilot 兼容别名（如 `createDirectory`、`createFile`、`readFile`、`listDirectory`、`runInTerminal`、`runCommand`、`fileSearch`、`textSearch`、`fetch`、`changes` 等）。部分 notebook/终端相关为占位符，返回“未实现”。
- 相关开关见 `settings.yaml`（如 `ENABLE_AGENT_SKILLS`、`ENABLE_*_SKILL`）。

## 启动 agent
- 配置 `settings.yaml`（需设置 `OPENAI_API_KEY`）
- 启动：`python main.py`
