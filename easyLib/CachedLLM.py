import os
from pathlib import Path
from vllm import LLM
from huggingface_hub import snapshot_download

class CachedLLM(LLM):
    def __init__(self, model_id, cache_dir, **llm_kwargs):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).resolve()
        self.local_model_dir = self.cache_dir / model_id
        model_path = self._ensure_model()
        super().__init__(model=model_path, **llm_kwargs)

    def _ensure_model(self):
        if not self.local_model_dir.exists():
            print(f"[INFO] 本地未检测到模型，将自动下载到 {self.local_model_dir} ...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.local_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("[INFO] 模型下载完成。")
        else:
            print(f"[INFO] 检测到本地模型 {self.local_model_dir}，将离线加载。")
        os.environ["HF_HUB_OFFLINE"] = "1"
        return str(self.local_model_dir)


if __name__ == "__main__":
    from vllm import SamplingParams
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    model_id = "TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill" # "facebook/opt-125m"
    cache_dir = "./models"
    llm = CachedLLM(
        model_id=model_id,
        cache_dir=cache_dir,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")