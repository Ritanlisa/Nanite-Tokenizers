"""agent.tools package -- split of the former monolithic agent/tools.py (T6)."""

from .rag_tools import (
    rag_engine,
    RAGDocListTool,
    RAGDocCatalogTool,
    RAGKeywordSearchTool,
    RAGRegexSearchTool,
    RAGVectorSearchTool,
    RAGLastSearchPagingTool,
    RAGGetPagesTool,
)
from .misc_tools import BugTool, FeedbackTool, MathComputeTool
from .shell_tools import FileIOTool, ShellTool
from .skill_tools import _build_skill_tools
from .web_tools import FetchWebpageTool, SearchTool, URLRegSearchTool, WebVisitTool

__all__ = [
    "tools",
    "rag_engine",
    "RAGDocListTool",
    "RAGDocCatalogTool",
    "RAGKeywordSearchTool",
    "RAGRegexSearchTool",
    "RAGVectorSearchTool",
    "RAGLastSearchPagingTool",
    "RAGGetPagesTool",
    "FeedbackTool",
    "BugTool",
    "MathComputeTool",
    "FetchWebpageTool",
    "ShellTool",
    "FileIOTool",
    "SearchTool",
    "URLRegSearchTool",
    "WebVisitTool",
    "_build_skill_tools",
]

tools = [
    RAGDocListTool(),
    RAGDocCatalogTool(),
    RAGKeywordSearchTool(),
    RAGRegexSearchTool(),
    RAGVectorSearchTool(),
    RAGLastSearchPagingTool(),
    RAGGetPagesTool(),
    FeedbackTool(),
    BugTool(),
    MathComputeTool(),
    FetchWebpageTool(),
    ShellTool(),
    FileIOTool(),
    SearchTool(),
    URLRegSearchTool(),
    WebVisitTool(),
] + _build_skill_tools()
