from nanite_tokenizers.models.compressor import AdaptiveCompressor, CompleteCompressorSystem

__all__ = ["AdaptiveCompressor", "CompleteCompressorSystem"]


if __name__ == "__main__":
    batch, seq_len, d_model = 4, 128, 768
    k = 16

    model = AdaptiveCompressor(d_model=d_model, max_k=64)
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.zeros(batch, seq_len).bool()

    out = model(x, k, mask)
    print(f"Input shape: {x.shape}, compressed: {out.shape}")

    vocab_size = 32000
    system = CompleteCompressorSystem(
        vocab_size=vocab_size,
        d_model=768,
        max_k=64,
        target_llm_dim=4096,
    )
    input_ids = torch.randint(0, vocab_size, (2, 100))
    compressed = system(input_ids, k=16)
    print(f"System output shape: {compressed.shape}")