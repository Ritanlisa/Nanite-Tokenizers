#!/bin/bash
cd /home/hjq/Nanite-Tokenizers-lite || exit 1
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=sk-dummy12345678901234567890
export OPENAI_API_URL=http://localhost:11434/v1
export KG_MCP_SERVER_COMMAND=/home/hjq/Nanite-Tokenizers-lite/.venv/bin/python /home/hjq/Nanite-Tokenizers-lite/scripts/sysml_rag_mcp_server.py serve

exec .venv/bin/python scripts/build_kg_recursive.py
