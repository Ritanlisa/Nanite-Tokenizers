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
        if "<think>" in generated_text:
            if "</think>" in generated_text:
                parts = generated_text.split("</think>")
                parts[0] = parts[0].replace("<think>", "").strip()
                print(color(parts[0], "1;34"))
                print(color(parts[1].strip(), "0;37"))
            else:
                print(color(generated_text.replace("<think>", "").strip(), "1;34"))
        else:
            print(color(generated_text, "0;37"))
