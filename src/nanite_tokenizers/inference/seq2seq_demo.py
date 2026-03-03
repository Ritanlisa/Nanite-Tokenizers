from __future__ import annotations

from easyLib.CachedSeq2Seq import CachedSeq2Seq

from nanite_tokenizers.utils.console import color


PROMPTS = [
    "Hello, my name is",
    "translate English to German: The house is wonderful.",
    (
        "summarize: Numerous epidemiological studies suggest that moderate coffee "
        "consumption-typically 3-5 cups per day-is linked to a lower risk of all-cause "
        "mortality. The potential mechanisms include its high antioxidant content and "
        "anti-inflammatory properties, which may protect against conditions like "
        "cardiovascular disease, stroke, and certain cancers. However, researchers note "
        "that these findings primarily show an association, not definitive causation, and "
        "that individual factors like genetics and overall diet play crucial roles."
    ),
    "The future of AI is",
]


def _format_generation_output(generated_text: str) -> tuple[str, str | None]:
    if "<think>" not in generated_text:
        return generated_text, None

    think_text = generated_text.replace("<think>", "").strip()
    if "</think>" not in think_text:
        return think_text, None

    think_block, remainder = think_text.split("</think>", maxsplit=1)
    return think_block.strip(), remainder.strip() or None


def run_demo() -> None:
    model_id = "google-t5/t5-large"
    cache_dir = "./models"
    llm = CachedSeq2Seq(model_id=model_id, cache_dir=cache_dir)
    outputs = llm.generate(PROMPTS, max_new_tokens=64)

    header = color("=== Generation Results ===", "1;36")
    print(header)
    for idx, (prompt, generated_text) in enumerate(zip(PROMPTS, outputs), start=1):
        print(color(f"\n[{idx}] Prompt:", "1;33"))
        print(color(prompt, "0;37"))
        print(color("Generated:", "1;32"))
        think_text, final_text = _format_generation_output(generated_text)
        if final_text is None:
            print(color(think_text, "0;37"))
            continue
        print(color(think_text, "1;34"))
        print(color(final_text, "0;37"))
