from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    cwd = Path.cwd()
    if (cwd / "web_server.py").exists():
        sys.path.insert(0, str(cwd))
        return

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "web_server.py").exists():
        sys.path.insert(0, str(repo_root))

def main() -> None:
    _ensure_repo_root_on_path()
    from web_server import main as root_main

    root_main()


if __name__ == "__main__":
    main()
