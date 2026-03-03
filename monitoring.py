import functools

from prometheus_client import Counter, Histogram, start_http_server

rag_query_count = Counter("rag_query_total", "Total RAG queries", ["success"])
rag_query_latency = Histogram("rag_query_latency_seconds", "RAG query latency")
rag_cache_hit_ratio = Counter("rag_cache_hit_total", "RAG cache hit/miss", ["hit"])

agent_tool_call = Counter("agent_tool_call_total", "Agent tool calls", ["tool_name"])
agent_step_duration = Histogram("agent_step_duration_seconds", "Agent step duration")
agent_token_usage = Counter("agent_token_total", "Total tokens used")

mcp_restart_count = Counter("mcp_restart_total", "MCP restart count")


def start_metrics_server(port: int = 8000) -> None:
    start_http_server(port)


def monitor_rag_query(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with rag_query_latency.time():
            try:
                result = func(*args, **kwargs)
            except Exception:
                rag_query_count.labels(success="false").inc()
                raise
            rag_query_count.labels(success="true").inc()
            return result

    return wrapper
