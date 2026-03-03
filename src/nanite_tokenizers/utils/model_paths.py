from __future__ import annotations

import os

from nanite_tokenizers.utils.env import is_env_flag_enabled


def resolve_model_path(repo_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    base_dir = os.path.abspath(base_dir)
    local_path = os.path.join(base_dir, repo_id)
    offline_only = is_env_flag_enabled("OFFLINE_ONLY")
    if os.path.isdir(local_path):
        return local_path
    if offline_only:
        raise FileNotFoundError(
            f"OFFLINE_ONLY is enabled and local model is missing: {local_path}"
        )
    return repo_id
