# Nanite-Tokenizers 中文说明

## 目标
- 训练、推理、工具分模块，结构清晰。
- 复用型包结构，主代码位于 `src/`。

## 项目结构
- `src/nanite_tokenizers/`：主包
- `web_server.py`：**入口 / 唯一入口** — FastAPI Web 服务器 + 静态 UI

## 快速开始
- 环境安装：`uv sync`
- 启动 Web UI：`python web_server.py --host 0.0.0.0 --port 7860`
- 打开浏览器访问：`http://localhost:7860`

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
  - Copilot 兼容别名（如 `createDirectory`、`createFile`、`readFile`、`listDirectory`、`runInTerminal`、`runCommand`、`fileSearch`、`textSearch`、`fetch`、`changes` 等）。部分 notebook/终端相关为占位符，返回"未实现"。
- 相关开关见 `settings.yaml`（如 `ENABLE_AGENT_SKILLS`、`ENABLE_*_SKILL`）。

## 启动 Web UI
- 安装依赖：`python -m pip install fastapi uvicorn`
- 启动：`python web_server.py --host 0.0.0.0 --port 7860`
- 打开：`http://localhost:7860`
