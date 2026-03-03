import os
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download


class CachedSeq2Seq:
    def __init__(self, model_id, cache_dir, **model_kwargs):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).resolve()
        self.local_model_dir = self.cache_dir / model_id
        model_path = self._ensure_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **model_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _ensure_model(self):
        if not self.local_model_dir.exists():
            print(f"[INFO] 本地未检测到模型，将自动下载到 {self.local_model_dir} ...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.local_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                use_auth_token=None,
            )
            print("[INFO] 模型下载完成。")
        else:
            print(f"[INFO] 检测到本地模型 {self.local_model_dir}，将离线加载。")
        os.environ["HF_HUB_OFFLINE"] = "1"
        return str(self.local_model_dir)

    def generate(self, prompts, **gen_kwargs):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
