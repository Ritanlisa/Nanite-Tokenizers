from __future__ import annotations

from torch.utils.data import Dataset


class LogDataset(Dataset):
    def __init__(self, tokenizer, num_samples: int = 1000, seq_len: int = 256) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.texts = [
            (
                "2026-02-12 10:23:45 node{idx} kernel: "
                "ECC memory error at address 0x3f8a1c, corrected"
            ).format(idx=i % 100)
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)
