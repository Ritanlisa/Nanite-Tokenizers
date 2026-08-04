"""RAG engine package: singleton facade + responsibility-separated collaborators.

Backward-compatible import surface:
- ``from rag.engine import RAGEngine``
- ``from rag.engine import SUPPORTED_RAG_EXTENSIONS``
"""

from rag.engine.constants import (
    DOC_TREE_CACHE_FILENAME,
    DOC_TREE_CACHE_VERSION,
    DOC_TREE_KEYWORD_VERSION,
    INDEX_METADATA_DROP_KEYS,
    INDEX_METADATA_MAX_VALUE_LENGTH,
    SUPPORTED_RAG_EXTENSIONS,
)
from rag.engine.embedding_factory import OpenAICompatibleEmbedding, RAGEmbeddingFactory
from rag.engine.facade import RAGEngine
from rag.engine.index_manager import DocumentRegistry, RAGIndexManager
from rag.engine.query_engine import RAGQueryEngine

__all__ = [
    "RAGEngine",
    "RAGIndexManager",
    "RAGQueryEngine",
    "RAGEmbeddingFactory",
    "OpenAICompatibleEmbedding",
    "DocumentRegistry",
    "SUPPORTED_RAG_EXTENSIONS",
    "INDEX_METADATA_MAX_VALUE_LENGTH",
    "INDEX_METADATA_DROP_KEYS",
    "DOC_TREE_CACHE_FILENAME",
    "DOC_TREE_CACHE_VERSION",
    "DOC_TREE_KEYWORD_VERSION",
]
