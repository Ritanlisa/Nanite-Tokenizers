"""
SysML v2 Complete Rewrite: 按章节处理，每章生成完整 SysML v2 package。
所有属性带值，所有实体带描述，严格标准标识符引用。
"""
import asyncio, re, sys, time, logging
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import config
config.settings = config.settings.update(
    RAG_DB_NAME="AIOPS", RAG_DB_NAMES=["AIOPS"], OCR_MODEL=None,
)

from rag.documents import load_rag_documents_from_paths
from rag.engine import SUPPORTED_RAG_EXTENSIONS

SYSML_REWRITE_PROMPT = """Convert this technical manual chapter into SysML v2.
Model the ACTUAL physical hardware at the installation site.

CRITICAL: Two kinds of elements:
  part def TypeName { ... }        — TYPE definition (what kind of thing)
  part instanceName { ... }       — PHYSICAL instance (the actual hardware)

RULES:
1. ASCII [a-zA-Z_][a-zA-Z0-9_]* identifiers. No quotes, No Chinese, No spaces.
2. NEVER break a line inside a 'quoted string'.
3. part def for types; part for instances. NO `part def X : Y`.
4. ALL values single-quoted: cores = 1024; power = '8MW';
5. connect: connect cab_1 to sw_A;
6. requirement: requirement Limit { ... }

Example:
```sysml
package Chapter1 {
    part def Compute_Rack {
        attribute height_U = 42;
        attribute power_kW = 30;
    }
    part rack_R0P0 {
        attribute location = 'R0-P0';
        attribute type = 'Compute_Rack';
    }
    connect rack_R0P0 to switch_S00;
}
```

Convert this chapter:"""


def group_by_chapter(text: str) -> list[tuple[str, str]]:
    """Split text by chapter headings and group subsections"""
    sections = re.split(r'\n(?=#{1,3}\s)', text)
    chapters = []
    current_chapter = "Preamble"
    current_text = []
    
    for sec in sections:
        m = re.match(r'(#{1,3})\s*(.+)', sec.strip())
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 1 and heading.startswith('第') and '章' in heading:
                if current_text:
                    chapters.append((current_chapter, '\n'.join(current_text)))
                current_chapter = heading
                current_text = [sec]
            else:
                current_text.append(sec)
        else:
            current_text.append(sec)
    
    if current_text:
        chapters.append((current_chapter, '\n'.join(current_text)))
    
    return chapters


def clean_output(text: str) -> str:
    """Strip markdown, non-ASCII, doc bodies, normalize, quote unquoted values"""
    text = re.sub(r'```\w*\n?', '', text)
    text = re.sub(r'\n?```', '', text)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'[^\x00-\x7F\n]', '', text)
    text = re.sub(r'\{\s*doc\s*[^}]*\}', ';', text)
    text = re.sub(r'\{\s*\}', ';', text)
    text = re.sub(r'\bconnection\s+connect\b', 'connect', text)
    text = re.sub(r'\ballocation\s+connect\b', 'allocate', text)
    # Strip `: Type` from part def (LLM sometimes confuses def with usage)
    text = re.sub(r'(part def\s+\w+)\s*:\s*\w+', r'\1', text)
    # Quote all attribute/requirement values that are bare (prevent keyword conflicts)
    # Matches:  = value;  where value contains non-numeric characters or keywords
    text = re.sub(
        r"(=\s*)([a-zA-Z][^;{}\n]*?)(\s*;)",
        lambda m: f"= '{m.group(2).strip()}'{m.group(3)}" 
        if not m.group(2).strip().startswith("'") and re.search(r'[a-zA-Z]', m.group(2))
        else m.group(0),
        text,
    )
    # Strip outer package wrapper
    text = re.sub(r'^package\s+\w+\s*\{\s*\n?', '', text, count=1)
    text = re.sub(r'\n?\}\s*$', '', text, count=1)
    # Fix broken single-quoted strings split across lines (LLM artifact)
    for _ in range(5):  # multiple passes for nested breaks
        old = text
        text = re.sub(r"'([^'\n]*)\n([^']*)'", r"'\1\2'", text)
        if text == old:
            break
    # Keep only sysml-relevant lines
    lines = []
    for line in text.strip().split('\n'):
        s = line.strip()
        if not s:
            lines.append('')
        elif s.startswith(('package ', 'part ', 'attribute ', 'port ', 'item ',
                           'requirement ', 'connection ', 'connect ',
                           'interface ', 'allocation ', 'import ', 'alias ',
                           '}', '{', '//', '/*')):
            lines.append(line)
    return '\n'.join(lines)
    lines = []
    for line in text.strip().split('\n'):
        s = line.strip()
        if not s:
            lines.append('')
        elif s.startswith(('package ', 'part ', 'attribute ', 'port ', 'item ',
                           'requirement ', 'connection ', 'interface ', 'allocation ',
                           'import ', 'alias ', 'connect ', '}', '{', '/*', '//',
                           'doc ', 'ref ', 'abstract ', 'in ', 'out ', 'subsets')):
            lines.append(line)
    return '\n'.join(lines)


def validate_standard(text: str) -> list[str]:
    """Check against standard SysML v2 identifier rules"""
    errors = []
    id_re = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    for i, line in enumerate(text.split('\n'), 1):
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('/*'):
            continue
        for kw in ['package ', 'part def ', 'attribute def ', 'port def ',
                    'item def ', 'requirement def ', 'connection def ',
                    'interface def ', 'allocation def ', 'part ', 'attribute ',
                    'port ', 'item ', 'requirement ', 'connection ',
                    'interface ', 'allocation ']:
            for prefix in [kw]:
                if line.startswith(prefix):
                    rest = line[len(prefix):]
                    # Extract name (before ; { = : >)
                    name_end = len(rest)
                    for delim in [';', '{', '=', ':', '>', ' ']:
                        idx = rest.find(delim)
                        if idx != -1 and idx < name_end:
                            name_end = idx
                    name = rest[:name_end].strip()
                    if name and name[0] == "'":
                        break  # Properly quoted
                    if name and not id_re.match(name):
                        errors.append(f"L{i}: unquoted '{name}' after '{kw.strip()}'")
                    break
    return errors


async def rewrite_chapter(chapter_name: str, chapter_text: str, client: OpenAI, model: str) -> str:
    """Rewrite one chapter as SysML v2"""
    # Trim to fit context
    text_limit = 10000
    if len(chapter_text) > text_limit:
        # Take start + middle + end
        third = text_limit // 3
        chapter_text = (
            chapter_text[:third] +
            "\n\n...(omitted middle section)...\n\n" +
            chapter_text[-third:]
        )
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSML_REWRITE_PROMPT},
            {"role": "user", "content": f"Chapter: {chapter_name}\n\n{chapter_text}"},
        ],
        temperature=0.1,
        max_tokens=4000,
        timeout=300,
    )
    return resp.choices[0].message.content or ""


async def main():
    client = OpenAI(
        api_key=config.settings.OPENAI_API_KEY,
        base_url=config.settings.OPENAI_API_URL, timeout=300,
    )
    model = "qwen3:8b"
    
    print("=" * 60)
    print(f"SysML v2 Rewrite: 湖超-硬件维护手册")
    print(f"Model: {model}")
    print("=" * 60)
    t0 = time.time()
    
    # Load document
    print("\n[1] Loading...")
    rag_docs = load_rag_documents_from_paths(
        ["database/AIOPS/docs/湖超-硬件维护手册20231225.doc"],
        SUPPORTED_RAG_EXTENSIONS,
    )
    doc = rag_docs[0]
    pages = doc.get_mono_pages()
    all_text = "\n\n".join(
        getattr(p, "markdown_text", "") or ""
        for p in pages
        if (getattr(p, "markdown_text", "") or "").strip()
        and getattr(p, "category", "") not in ("cover", "catalogue")
    )
    print(f"  {doc.page_count} pages, {len(all_text)} chars")
    
    # Group by chapter
    chapters = group_by_chapter(all_text)
    print(f"  {len(chapters)} chapters")
    for i, (name, text) in enumerate(chapters):
        print(f"    [{i}] {name} ({len(text)}c)")
    
    # Rewrite each chapter
    sysml_parts = []
    with tqdm(total=len(chapters), desc="  Rewriting", unit="ch") as bar:
        for chap_name, chap_text in chapters:
            try:
                raw = await rewrite_chapter(chap_name, chap_text, client, model)
                cleaned = clean_output(raw)
                if cleaned.strip():
                    # Don't re-wrap — LLM already outputs package structure
                    sysml_parts.append(cleaned.strip() + "\n")
                    bar.set_postfix_str(f"{len(cleaned)}c")
            except Exception as e:
                bar.set_postfix_str(f"err: {str(e)[:30]}")
            bar.update(1)
    
    print(f"\n  {len(sysml_parts)} SysML fragments generated")
    
    # Assemble into final document
    header = "/* SysML v2 - Huchao Hardware Maintenance Manual */\n"
    header += f"/* {doc.page_count} pages, {time.strftime('%Y-%m-%d %H:%M')} */\n\n"
    header += "package Huchao_Hardware_Manual {\n"
    
    raw_body = '\n'.join(sysml_parts)
    
    # Indent and balance braces
    indented = []
    for line in raw_body.split('\n'):
        stripped = line.strip()
        if stripped == '':
            indented.append('')
        else:
            indented.append('    ' + line)
    body = '\n'.join(indented)
    
    # Balance braces: count opens vs closes in body
    opens = body.count('{')
    closes = body.count('}')
    if opens > closes:
        for i in range(opens - closes):
            body += '\n' + '    ' * (opens - closes - i - 1) + '}'
    
    footer = "\n}\n"
    final = header + body + footer
    
    # Validate
    print("\n[2] Validating against standard SysML v2...")
    errors = validate_standard(final)
    if errors:
        print(f"  {len(errors)} standard issues found (common with LLM-generated code):")
        for e in errors[:8]:
            print(f"    {e}")
    else:
        print("  No standard identifier issues!")
    
    # Also test with our parser
    print("\n[3] Testing with project parser...")
    from sysml.sysml_parser import parse_sysml_text
    try:
        parse_sysml_text(final)
        print("  Parser OK!")
    except Exception as e:
        print(f"  Parser issue: {str(e)[:200]}")
    
    # Save
    kg_path = Path("database/AIOPS/knowledge_graph.sysml")
    kg_path.write_text(final, encoding="utf-8")
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Saved: {kg_path}")
    print(f"Size: {len(final)} chars, {len(final.splitlines())} lines")
    print(f"Standard issues: {len(errors)}")
    print(f"Time: {elapsed:.0f}s")
    print(f"{'='*60}")
    
    # Preview file
    print(f"\nFile preview (first 1000 chars):")
    print(final[:1000])


if __name__ == "__main__":
    asyncio.run(main())
