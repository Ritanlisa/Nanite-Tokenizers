def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"
