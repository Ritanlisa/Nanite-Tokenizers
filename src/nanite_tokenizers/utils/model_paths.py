from __future__ import annotations

import os


def resolve_model_path(repo_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    base_dir = os.path.abspath(base_dir)
    local_path = os.path.join(base_dir, repo_id)
    if os.path.isdir(local_path):
        return local_path
    return repo_id
