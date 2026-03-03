from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nanite_tokenizers.data.log_dataset import LogDataset
from nanite_tokenizers.models.compressor import CompleteCompressorSystem
from nanite_tokenizers.utils.model_paths import resolve_model_path

MODEL_IDS = [
    "TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Math-Distill",
    "TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill",
    "TeichAI/Qwen3-8B-DeepSeek-v3.2-Speciale-Distill",
]


def train_simplier(model_index: int = 0) -> None:
    accelerator = Accelerator()
    device = accelerator.device

    d_model = 768
    nhead = 12
    num_encoder_layers = 6
    max_k = 64

    learning_rate = 1e-4
    batch_size = 4
    epochs = 3
    seq_max_len = 512
    k_min, k_max = 4, 32

    print("Loading frozen target LLM...")
    llm_model_id = MODEL_IDS[model_index]
    llm_model_path = resolve_model_path(llm_model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    target_llm_dim = getattr(llm.config, "hidden_size", None) or getattr(llm.config, "n_embd", None)
    if target_llm_dim is None:
        raise ValueError("Cannot infer hidden_size/n_embd from model config")
    config_nhead = getattr(llm.config, "num_attention_heads", None) or getattr(llm.config, "n_head", None)

    d_model = target_llm_dim
    if config_nhead is not None:
        nhead = config_nhead
    if d_model % nhead != 0:
        raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")

    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= vocab_size:
        raise ValueError(f"pad_token_id({tokenizer.pad_token_id}) out of range for vocab size {vocab_size}")

    compressor = CompleteCompressorSystem(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        max_k=max_k,
        target_llm_dim=target_llm_dim,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    for name, param in compressor.named_parameters():
        param.requires_grad = "query_embed" in name or "adapter" in name

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, compressor.parameters()),
        lr=learning_rate,
    )

    dataset = LogDataset(tokenizer, num_samples=500, seq_len=seq_max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Start training...")
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for input_ids, attention_mask in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_seq_len = attention_mask.sum(dim=1).float()

            k = random.randint(k_min, k_max)
            compressed = compressor(input_ids, k, attention_mask)

            with torch.no_grad():
                orig_outputs = llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                orig_hidden = orig_outputs.hidden_states[-2]
                orig_mask = attention_mask.unsqueeze(-1).float()
                orig_pooled = (orig_hidden * orig_mask).sum(dim=1) / orig_mask.sum(dim=1).clamp(min=1e-9)

            comp_outputs = llm(
                inputs_embeds=compressed.to(dtype=llm.dtype),
                output_hidden_states=True,
            )
            comp_hidden = comp_outputs.hidden_states[-2]
            comp_pooled = comp_hidden.mean(dim=1)

            cos_sim = F.cosine_similarity(orig_pooled, comp_pooled, dim=-1)
            fidelity_reward = (cos_sim + 1) / 2

            compression_ratio = k / batch_seq_len
            compression_reward = 1 - compression_ratio

            reward = fidelity_reward * compression_reward
            loss = -reward.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, compressor.parameters()),
                1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "fidelity": f"{fidelity_reward.mean().item():.3f}",
                    "comp_ratio": f"{compression_ratio.mean().item():.3f}",
                    "reward": f"{reward.mean().item():.3f}",
                }
            )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} done, avg loss: {avg_loss:.4f}")

    torch.save(compressor.state_dict(), "compressor_final.pt")
    print("Training done, saved to compressor_final.pt")
