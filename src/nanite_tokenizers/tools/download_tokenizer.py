from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from nanite_tokenizers.utils.env import is_env_flag_enabled


_TOKENIZER_REPO_ID = "TeichAI/Devstral-Small-2505-Deepseek-V3.2-Speciale-Distill"
_TOKENIZER_ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer.*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "spiece.model",
]


def download_tokenizer() -> None:
    offline_only = is_env_flag_enabled("OFFLINE_ONLY")
    if offline_only:
        raise RuntimeError("OFFLINE_ONLY is enabled: tokenizer download is disabled")

    destination = Path("./models/TeichAI-Devstral-tokenizer")
    snapshot_download(
        repo_id=_TOKENIZER_REPO_ID,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        tqdm_class=tqdm,
        allow_patterns=_TOKENIZER_ALLOW_PATTERNS,
    )
    print(f"Done. Tokenizer saved to: {destination.resolve()}")
