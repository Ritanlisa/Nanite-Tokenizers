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
- Install editable: `python -m pip install -e .`
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
