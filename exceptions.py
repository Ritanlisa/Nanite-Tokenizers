class RAGError(Exception):
    pass


class IndexNotFoundError(RAGError):
    pass


class APIThrottlingError(RAGError):
    pass


class QueryTimeoutError(RAGError):
    pass


class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPTimeoutError(MCPError):
    pass


class MCPFatalError(MCPError):
    pass


class TokenLimitExceeded(Exception):
    pass
