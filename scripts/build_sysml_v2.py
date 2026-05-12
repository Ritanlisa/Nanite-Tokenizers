"""
SysML v2 Builder: LLM knowledge extraction → Python SysML generation
Focus: individual part instances with real attributes (no count-as-instance)
"""

import asyncio, json, re, sys, time, logging
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

_SYSML_KW = {
    'abstract','alias','allocation','attribute','connect','connection','def',
    'flow','id','import','in','interface','item','out','part','port','private',
    'protected','public','ref','requirement','return','to','doc','inout',
    'true','false','not','and','or','xor','if','else','for','while',
}

EXTRACT_PROMPT = """Extract ALL hardware from this text. Each physical item = one JSON entry.

Output {"parts": [...], "conns": [...]}

Part format: {"name": "id", "type": "Kind", "attrs": {"key": "val", ...}}
Conn format:  {"from": "id", "to": "id"}

Rules:
- ASCII names only [a-zA-Z_][a-zA-Z0-9_]*
- Every named item gets its own entry (not counts!)
- attrs: ALL specs as string values
- Output ONLY valid JSON"""


def sn(n: str) -> str:
    """Safe name for SysML v2"""
    n = re.sub(r'[^\x00-\x7F]', '', str(n))
    n = re.sub(r'[^a-zA-Z0-9_]', '_', n)
    n = re.sub(r'_+', '_', n).strip('_')
    if not n or n[0].isdigit():
        n = '_' + n
    if n.lower() in _SYSML_KW:
        n = n + '_attr'
    return n


def sv(v: str) -> str:
    """Safe value (preserve digits)"""
    return re.sub(r'[^\x00-\x7F]', '', str(v)).strip().strip("'\" ")


async def extract(chapter_text: str, client: OpenAI, model: str) -> dict:
    results = {"parts": [], "conns": []}
    for start in range(0, len(chapter_text), 5000):
        chunk = chapter_text[start:start+5000]
        if len(chunk.strip()) < 300:
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":EXTRACT_PROMPT},{"role":"user","content":chunk}],
            temperature=0.1, max_tokens=2500, timeout=180,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r'```\w*\n?', '', raw).replace('\n', ' ')
        try:
            data = json.loads(raw)
        except:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try: data = json.loads(m.group())
                except: continue
            else:
                continue
        results["parts"].extend(data.get("parts", []))
        results["conns"].extend(data.get("conns", data.get("connections", [])))
    return results


def build_sysml(all_data: list[dict], pages: int) -> str:
    types = {}
    instances = []
    conns = []
    
    for d in all_data:
        for p in d.get("parts", []):
            nm = sn(p.get("name", ""))
            tp = sn(p.get("type", p.get("type", "")))
            if not nm or not tp:
                continue
            attrs = p.get("attrs", p.get("attributes", {}))
            inst_attrs = {}
            for k, v in (attrs.items() if isinstance(attrs, dict) else []):
                k = sn(k)
                if not k or k == '_':
                    continue
                v = sv(str(v))
                if v and v.lower() not in ('', 'none', 'null', 'n/a'):
                    inst_attrs[k] = v
            
            instances.append((nm, inst_attrs))
            
            if tp not in types:
                types[tp] = {}
            for k, v in inst_attrs.items():
                types[tp][k] = v
        
        for c in d.get("conns", d.get("connections", [])):
            f = sn(str(c.get("from", c.get("source", ""))))
            t = sn(str(c.get("to", c.get("target", ""))))
            if f and t:
                conns.append((f, t))
    
    # Build output
    lines = [
        f"/* SysML v2 - Huchao Hardware Maintenance Manual */",
        f"/* {pages} pages, {time.strftime('%Y-%m-%d %H:%M')} */",
        f"/* {len(types)} types, {len(instances)} instances, {len(conns)} connections */",
        "",
        "package Huchao_Hardware_Manual {",
    ]
    
    # Type definitions
    if types:
        lines.append("    /* === Component Types === */")
        for tp, attrs in sorted(types.items()):
            non_empty = {k: v for k, v in attrs.items() if v and k != '_'}
            if non_empty:
                lines.append(f"    part def {tp} {{")
                for k, v in sorted(non_empty.items()):
                    if re.match(r'^\d+$', v):
                        lines.append(f"        attribute {k} = {v};")
                    else:
                        lines.append(f"        attribute {k} = '{v}';")
                lines.append("    }")
            else:
                lines.append(f"    part def {tp};")
    
    # Physical instances
    if instances:
        lines.append("")
        lines.append("    /* === Physical Instances === */")
        used = set()
        for nm, attrs in instances:
            un = nm
            c = 0
            while un in used:
                c += 1
                un = f"{nm}_{c}"
            used.add(un)
            
            if attrs:
                lines.append(f"    part {un} {{")
                for k, v in sorted(attrs.items()):
                    if re.match(r'^\d+$', v):
                        lines.append(f"        attribute {k} = {v};")
                    else:
                        lines.append(f"        attribute {k} = '{v}';")
                lines.append("    }")
            else:
                lines.append(f"    part {un};")
    
    # Connections
    if conns:
        lines.append("")
        lines.append("    /* === Connections === */")
        seen = set()
        for f, t in conns:
            key = f"{f}->{t}"
            if key not in seen:
                seen.add(key)
                lines.append(f"    connect {f} to {t};")
    
    lines.append("}")
    return '\n'.join(lines)


async def main():
    client = OpenAI(api_key=config.settings.OPENAI_API_KEY, base_url=config.settings.OPENAI_API_URL, timeout=300)
    model = "qwen3:8b"
    
    print(f"SysML v2: Huchao Hardware Manual ({model})")
    t0 = time.time()
    
    rag_docs = load_rag_documents_from_paths(
        ["database/AIOPS/docs/湖超-硬件维护手册20231225.doc"], SUPPORTED_RAG_EXTENSIONS,
    )
    doc = rag_docs[0]
    pages = doc.get_mono_pages()
    all_text = "\n\n".join(
        getattr(p, "markdown_text", "") or ""
        for p in pages
        if (getattr(p, "markdown_text", "") or "").strip()
        and getattr(p, "category", "") not in ("cover", "catalogue")
    )
    
    chapters = re.split(r'\n(?=#\s*第[一二三四五六七八九十\d]+章)', all_text)
    print(f"  {len(chapters)} chapters, {len(all_text)} chars")
    
    all_data = []
    with tqdm(total=len(chapters), desc="  Extracting") as bar:
        for ch in chapters:
            data = await extract(ch, client, model)
            all_data.append(data)
            pc = len(data.get("parts", []))
            cc = len(data.get("conns", []))
            bar.set_postfix_str(f"{pc}p {cc}c")
            bar.update(1)
    
    total_p = sum(len(d.get("parts", [])) for d in all_data)
    total_c = sum(len(d.get("conns", [])) for d in all_data)
    print(f"  Knowledge: {total_p} items, {total_c} connections")
    
    sysml = build_sysml(all_data, doc.page_count)
    
    from sysml.sysml_parser import parse_sysml_text
    try:
        parse_sysml_text(sysml)
        print(f"  Parser: OK")
    except Exception as e:
        print(f"  Parser: {str(e)[:150]}")
    
    kw = re.findall(r'attribute\s+(\w+)', sysml)
    conflicts = [a for a in kw if a.lower() in _SYSML_KW]
    print(f"  Keywords: {'OK' if not conflicts else str(conflicts)}")
    
    kg_path = Path("database/AIOPS/knowledge_graph.sysml")
    kg_path.write_text(sysml, encoding="utf-8")
    
    elapsed = time.time() - t0
    print(f"\n  Saved: {kg_path}")
    print(f"  {len(sysml)} chars, {len(sysml.splitlines())} lines, {elapsed:.0f}s")
    print(f"\n{sysml[:2000]}")


if __name__ == "__main__":
    asyncio.run(main())
