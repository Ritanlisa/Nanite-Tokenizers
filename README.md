Nanite-Tokenizers
=================

Goals
-----
- Separate training, inference demos, and tooling into clear modules.
- Provide a reusable package layout under `src/`.
- Keep legacy entry scripts working as wrappers.

Project layout
--------------
- `src/nanite_tokenizers/cli.py`: unified CLI entrypoint
- `src/nanite_tokenizers/training/`: training routines
- `src/nanite_tokenizers/inference/`: demo and inference scripts
- `src/nanite_tokenizers/models/`: model and compression code
- `src/nanite_tokenizers/data/`: datasets
- `src/nanite_tokenizers/tools/`: download utilities
- `src/nanite_tokenizers/utils/`: shared helpers

Quickstart
----------
- Install dependencies: `uv sync`
- Demo: `python -m nanite_tokenizers demo`
- Train: `python -m nanite_tokenizers train --model-index 0`
- Download tokenizer: `python -m nanite_tokenizers download`

Legacy entrypoints
------------------
These files remain as thin wrappers:
- `simplier.py`
- `download.py`
- `compressor.py`

RAG + MCP Agent
---------------
New components live at the repo root:
- `config.py`: settings and validation
- `rag/`: RAG engine, preprocessing, vector store
- `mcp/`: MCP fetch client
- `agent/`: tools, memory, agent executor
- `tests/`: pytest examples

Agent Skills
------------
Built-in skill tools are enabled by default and exposed to the agent via tool-calling:
- `skill_shell`: execute shell commands in workspace (with timeout/safety filter)
- `skill_file_io`: read/write/list/mkdir inside workspace root only
- `skill_search`: Selenium-driven search tool with configurable URL/XPath/regex parsing
- `skill_web_visit`: Selenium webpage visiting and text extraction for JS-rendered pages
- Copilot-compatible aliases are also registered (e.g. `createDirectory`, `createFile`,
  `readFile`, `listDirectory`, `runInTerminal`, `runCommand`, `fileSearch`, `textSearch`,
  `fetch`, `changes`, etc.). Some advanced notebook/terminal/session APIs are placeholders
  in local runtime and will return a clear "not implemented" message.

Related switches are in `settings.yaml` (`ENABLE_AGENT_SKILLS`, `ENABLE_*_SKILL`).

MCP Tool Input Protocol Extension
---------------------------------
For tool-calling, the agent supports an input sugar for referencing previous tool outputs:
- `tool[-1]`: latest completed tool output
- `tool[1]`: first completed tool output (positive index starts at 1)
- Accessor chain: `tool[-1][3]["link"]`, `tool[-1]["results"][0]`
- Pipe transform: `| regex <pattern> [replacement]`
- If replacement is omitted, default is full match (`$0`)
- Replacement supports `$0/$1/$2...` capture groups

Examples:
- `tool[-1][3]["link"]`
- `tool[-1][3]["link"] | regex https?://(.*?)/ $1`
- `tool[-1][3]["link"] | regex https?://(.*?)\.(.*?)\.(.*?)/ $1 $2 $3`

Run the agent
-------------
- Configure `settings.yaml` (set `OPENAI_API_KEY`)
- Start: `python main.py`

Run the Gradio GUI
------------------
- Install Gradio: `python -m pip install gradio`
- Start: `python gradio_app.py --host 0.0.0.0 --port 7860`

Run the static Web UI
---------------------
- Install web deps: `python -m pip install fastapi uvicorn`
- Start API + UI: `python web_server.py --host 0.0.0.0 --port 7860`
- Open: `http://localhost:7860`
