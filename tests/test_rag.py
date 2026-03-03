import os

import pytest

from rag.engine import RAGEngine


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_rag_query():
    engine = RAGEngine()
    result = engine.query("What is RAG?")
    assert "answer" in result
    assert "sources" in result
