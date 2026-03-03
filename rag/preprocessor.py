import re


def clean_document(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"^Page \d+\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s\.\,!\?\-\:\;]", "", text)
    return text.strip()
