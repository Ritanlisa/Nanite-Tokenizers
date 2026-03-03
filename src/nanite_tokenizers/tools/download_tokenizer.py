from __future__ import annotations

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


def download_tokenizer() -> None:
    snapshot_download(
        repo_id="TeichAI/Devstral-Small-2505-Deepseek-V3.2-Speciale-Distill",
        local_dir="./models/TeichAI-Devstral-tokenizer",
        local_dir_use_symlinks=False,
        tqdm_class=tqdm,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer.*",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "spiece.model",
        ],
    )
    print("Done.")
