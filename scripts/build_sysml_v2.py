"""
SysML v2 Builder: 3-pass builder
  Pass 1: Collect parts, hypervariables, document sections, connections
  Pass 2: Resolve ref target types (entity → type map)
  Pass 3: Generate SysML v2 with HV type defs, ref bindings, tagged doc sections
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

EXTRACT_PROMPT = """Extract hardware, hypervariables, and document text from this chapter.

Output JSON with:
{
  "parts": [{"name":"id","type":"Kind","attrs":{},"refs":{}}],
  "hypervariables": {"id": {"hv_type":"InstantVariable","source":"...","selector":"...",...}},
  "sections": [{"id":"sec_id","title":"Section Title","chapter":"Ch X","parent":"","text":"..."}],
  "conns": [{"from":"id","to":"id"}]
}

--- PART RULES ---
Each physical component = one entry.
name/type: ASCII only [a-zA-Z_][a-zA-Z0-9_]*
attrs: all specs as key:value strings
refs: {"slot_name": "target_id"} — links to hypervariables or other parts

--- HYPERVARIABLE RULES ---
5 types:
  InstantVariable   — fast read, no side effects (sensor, clock)
  BlockVariable     — blocking read, takes time (network request)
  ModifierVariable   — read=consume, changes state (message queue)
  Parameter         — writable, affects behavior (fan curve)
  Operation         — executable action (reboot, shutdown)
Fields: source (protocol://), selector (identifier), unit, timeout, etc.

--- SECTION RULES ---
Section text MUST embed hypervariable references with tags:
  <instvar:id:DisplayName>
  <blockvar:id:DisplayName>
  <modifier:id:DisplayName>
  <param:id:DisplayName>
  <operation:id:DisplayName>
Example: "If <instvar:fan_speed:Fan Speed> exceeds <param:threshold:Threshold>, run <operation:underclock:Underclock>"

--- OUTPUT ONLY VALID JSON ---"""


def sn(n: str) -> str:
    n = re.sub(r'[^\x00-\x7F]', '', str(n))
    n = re.sub(r'[^a-zA-Z0-9_]', '_', n)
    n = re.sub(r'_+', '_', n).strip('_')
    if not n or n[0].isdigit():
        n = '_' + n
    if n.lower() in _SYSML_KW:
        n = n + '_attr'
    return n


def sv(v: str) -> str:
    return re.sub(r'[^\x00-\x7F]', '', str(v)).strip().strip("'\" ")


def esc_doc(s: str) -> str:
    """Escape a string for embedding in doc \"...\" """
    return s.replace('\\', '\\\\').replace('"', '\\"')


async def extract(chapter_text: str, client: OpenAI, model: str) -> dict:
    results = {"parts": [], "hypervariables": {}, "sections": [], "conns": []}
    for start in range(0, len(chapter_text), 4000):
        chunk = chapter_text[start:start+4000]
        if len(chunk.strip()) < 300:
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":EXTRACT_PROMPT},{"role":"user","content":chunk}],
            temperature=0.1, max_tokens=3000, timeout=180,
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

        for hv_id, hv_info in data.get("hypervariables", {}).items():
            if isinstance(hv_info, dict):
                results["hypervariables"][hv_id] = hv_info

        for s in data.get("sections", []):
            if isinstance(s, dict) and s.get("id"):
                results["sections"].append(s)

    return results


def build_sysml(all_data: list[dict], pages: int) -> str:
    """
    Three-pass builder:
    1. Collect all data from extraction results
    2. Resolve ref target types
    3. Generate SysML v2 text
    """
    # ── Pass 1: Collect ──
    parts = []               # [(name, type, attrs_dict, refs_dict)]
    hypervariables = {}       # hv_id → hv_info dict
    sections = []             # [(id, title, chapter, parent, text)]
    conn_pairs = []           # [(from, to)]
    type_attrs = {}           # type_name → set of attribute names
    entity_types = {}         # entity_name → type_name (for ref resolution)

    for d in all_data:
        # Parts
        for p in d.get("parts", []):
            nm = sn(p.get("name", ""))
            tp = sn(p.get("type", ""))
            if not nm or not tp:
                continue
            attrs = p.get("attrs", p.get("attributes", {}))
            refs = p.get("refs", {})
            inst_attrs = {}
            for k, v in (attrs.items() if isinstance(attrs, dict) else []):
                ks = sn(k)
                if not ks or ks == '_':
                    continue
                vs = sv(str(v))
                if vs and vs.lower() not in ('', 'none', 'null', 'n/a'):
                    inst_attrs[ks] = vs
            parts.append((nm, tp, inst_attrs, refs))
            entity_types[nm] = tp
            if tp not in type_attrs:
                type_attrs[tp] = set()
            type_attrs[tp].update(inst_attrs.keys())

        # Hypervariables
        for hv_id, hv_info in d.get("hypervariables", {}).items():
            hv_id_clean = sn(hv_id)
            if hv_id_clean and isinstance(hv_info, dict):
                hypervariables[hv_id_clean] = hv_info
                hv_tp = hv_info.get("hv_type", "InstantVariable")
                entity_types[hv_id_clean] = hv_tp

        # Sections
        for s in d.get("sections", []):
            sec_id = sn(s.get("id", ""))
            title = s.get("title", "")
            chapter = s.get("chapter", "")
            parent = s.get("parent", "")
            text = s.get("text", "")
            if sec_id:
                sections.append((sec_id, title, chapter, parent, text))

        # Connections
        for c in d.get("conns", d.get("connections", [])):
            f = sn(str(c.get("from", c.get("source", ""))))
            t = sn(str(c.get("to", c.get("target", ""))))
            if f and t:
                conn_pairs.append((f, t))

    # ── Post-process: Normalize HV tag IDs + auto-create missing HVs ──
    hv_ids = set(hypervariables.keys())
    _HV_TAG_RE = re.compile(r'<(\w+):([a-zA-Z_][a-zA-Z0-9_]*):([^>]+)>')
    _TAG_TO_TYPE = {
        'instvar': 'InstantVariable', 'blockvar': 'BlockVariable',
        'modifier': 'ModifierVariable', 'param': 'Parameter',
        'operation': 'Operation',
    }
    
    def _fuzzy_match_hv(tag_id: str) -> str:
        """Find closest HV ID for an unmatched tag ID."""
        if tag_id in hv_ids:
            return tag_id
        tl = tag_id.lower()
        # Case-insensitive
        for hid in hv_ids:
            if hid.lower() == tl:
                return hid
        # Strip common prefixes
        for prefix in ('instvar_', 'param_', 'blockvar_', 'operation_', 'modifier_', 'instvar', 'hv_', 'var_'):
            if tl.startswith(prefix):
                stripped = tl[len(prefix):]
                for hid in hv_ids:
                    if hid.lower() == stripped:
                        return hid
        # Add common prefixes
        for prefix in ('instvar_', 'param_', 'blockvar_', 'operation_', 'modifier_'):
            tagged = prefix + tl
            for hid in hv_ids:
                if hid.lower() == tagged:
                    return hid
        # Substring
        for hid in hv_ids:
            hl = hid.lower()
            if tl in hl or hl in tl:
                return hid
        # Token overlap
        tag_tokens = set(re.split(r'[_\s]+', tl))
        best_score = 0
        best_id = tag_id
        for hid in hv_ids:
            hl = hid.lower()
            hv_tokens = set(re.split(r'[_\s]+', hl))
            overlap = len(tag_tokens & hv_tokens)
            if overlap > best_score:
                best_score = overlap
                best_id = hid
        return best_id if best_score > 0 else None  # None = no match at all

    def _ensure_hv(hv_id: str, hv_type: str, display_name: str):
        """Auto-create a missing HV instance from a section tag."""
        clean_id = sn(hv_id)
        if clean_id not in hypervariables:
            hypervariables[clean_id] = {
                "hv_type": hv_type,
                "source": "auto",
                "selector": clean_id,
                "displayName": display_name,
            }
            hv_ids.add(clean_id)
            entity_types[clean_id] = hv_type

    def _normalize_section_tags(text: str) -> str:
        def repl(m):
            tag_type = m.group(1)
            hv_id = m.group(2)
            display = m.group(3)
            matched = _fuzzy_match_hv(hv_id)
            if matched:
                return f'<{tag_type}:{matched}:{display}>'
            # Auto-create missing HV
            hv_tp = _TAG_TO_TYPE.get(tag_type, 'InstantVariable')
            _ensure_hv(hv_id, hv_tp, display)
            return f'<{tag_type}:{sn(hv_id)}:{display}>'
        return _HV_TAG_RE.sub(repl, text)

    sections = [
        (sec_id, title, chapter, parent, _normalize_section_tags(text))
        for sec_id, title, chapter, parent, text in sections
    ]

    # ── Pass 2: Resolve ref target types ──
    # For each part's refs, determine the target's SysML type
    type_refs = {}  # type_name → {slot_name: target_type}
    for nm, tp, _, refs in parts:
        if tp not in type_refs:
            type_refs[tp] = {}
        for slot_name, target_name in refs.items():
            slot_name_clean = sn(slot_name)
            target_name_clean = sn(target_name)
            # Skip self-referencing refs
            if target_name_clean == nm:
                continue
            target_type = entity_types.get(target_name_clean)
            # Skip refs pointing to the same type
            if target_type and target_type != tp:
                type_refs[tp][slot_name_clean] = target_type

    # ── Pass 3: Generate SysML ──
    lines = [
        f"/* SysML v2 - Huchao Hardware Maintenance Manual */",
        f"/* {pages} pages, {time.strftime('%Y-%m-%d %H:%M')} */",
        f"/* {len(type_attrs)} hardware types, {len(parts)} instances, "
        f"{len(hypervariables)} HVs, {len(sections)} sections, {len(conn_pairs)} connections */",
        "",
        "package Huchao_Hardware_Manual {",
    ]

    # 3a. Hypervariable type definitions
    hv_type_attrs = {}  # hv_type_name → set of attr keys
    for hv_id, hv_info in hypervariables.items():
        hv_tp = hv_info.get("hv_type", "InstantVariable")
        if hv_tp not in hv_type_attrs:
            hv_type_attrs[hv_tp] = set()
        for k in hv_info:
            if k not in ("hv_type",):
                hv_type_attrs[hv_tp].add(sn(k))

    if hv_type_attrs:
        lines.append("")
        lines.append("    /* === Hypervariable Type Definitions === */")
        for hv_tp in sorted(hv_type_attrs.keys()):
            attrs = hv_type_attrs[hv_tp]
            if attrs:
                lines.append(f"    part def {hv_tp} {{")
                for a in sorted(attrs):
                    lines.append(f"        attribute {a};")
                lines.append("    }")
            else:
                lines.append(f"    part def {hv_tp};")

    # 3b. Hypervariable instances
    if hypervariables:
        lines.append("")
        lines.append("    /* === Hypervariable Instances === */")
        # Group by type for readability
        for hv_tp in sorted(set(inf.get("hv_type", "InstantVariable") for inf in hypervariables.values())):
            for hv_id in sorted(hypervariables.keys()):
                hv_info = hypervariables[hv_id]
                if hv_info.get("hv_type", "InstantVariable") != hv_tp:
                    continue
                hv_attrs = {sn(k): sv(str(v)) for k, v in hv_info.items()
                           if k not in ("hv_type",) and sv(str(v))}
                # Normalize source to protocol URL (so real drivers get used)
                if "source" in hv_attrs:
                    src = hv_attrs["source"]
                if not src.startswith("file://") and not src.startswith("bash://") and not src.startswith("cmd://") and not src.startswith("ps://") and not src.startswith("ipmi://") and not src.startswith("http://") and not src.startswith("https://") and not src.startswith("snmp://") and not src.startswith("config://") and not src.startswith("sensor://") and not src.startswith("prom://"):
                    hv_attrs["source"] = f"bash://{src}"
                if hv_attrs:
                    lines.append(f"    part {hv_id} : {hv_tp} {{")
                    for k in sorted(hv_attrs.keys()):
                        v = hv_attrs[k]
                        if re.match(r'^\d+$', v):
                            lines.append(f"        attribute {k} = {v};")
                        else:
                            lines.append(f"        attribute {k} = '{v}';")
                    lines.append("    }")
                else:
                    lines.append(f"    part {hv_id} : {hv_tp};")

    # 3c. Hardware type definitions (with ref slots)
    if type_attrs:
        lines.append("")
        lines.append("    /* === Hardware Type Definitions === */")
        for tp in sorted(type_attrs.keys()):
            attrs = type_attrs.get(tp, set())
            refs = type_refs.get(tp, {})
            if attrs or refs:
                lines.append(f"    part def {tp} {{")
                for a in sorted(attrs):
                    lines.append(f"        attribute {a};")
                for slot_name, target_type in sorted(refs.items()):
                    lines.append(f"        ref part {slot_name} : {target_type};")
                lines.append("    }")
            else:
                lines.append(f"    part def {tp};")

    # 3d. Hardware instances
    if parts:
        lines.append("")
        lines.append("    /* === Hardware Instances === */")
        used = set()
        for nm, tp, inst_attrs, refs in parts:
            un = nm
            c = 0
            while un in used:
                c += 1
                un = f"{nm}_{c}"
            used.add(un)

            if inst_attrs or refs:
                lines.append(f"    part {un} : {tp} {{")
                for k in sorted(inst_attrs.keys()):
                    v = inst_attrs[k]
                    if re.match(r'^\d+$', v):
                        lines.append(f"        attribute {k} = {v};")
                    else:
                        lines.append(f"        attribute {k} = '{v}';")
                for slot_name, target_name in refs.items():
                    slot_clean = sn(slot_name)
                    target_clean = sn(target_name)
                    lines.append(f"        ref part {slot_clean} = {target_clean};")
                lines.append("    }")
            else:
                lines.append(f"    part {un} : {tp};")

    # 3e. DocumentSection type + sections with doc text
    if sections:
        lines.append("")
        lines.append("    /* === Document Sections === */")
        lines.append("    part def DocumentSection {")
        lines.append("        attribute title;")
        lines.append("        attribute chapter;")
        lines.append("    }")
        lines.append("")

        for sec_id, title, chapter, parent, text in sections:
            lines.append(f"    part {sec_id} : DocumentSection {{")
            cleaned_title = esc_doc(title) if title else ""
            if cleaned_title:
                lines.append(f'        doc "{cleaned_title}";')
            lines.append(f'        attribute title = \'{esc_doc(title)}\';')
            if chapter:
                lines.append(f'        attribute chapter = \'{esc_doc(chapter)}\';')
            cleaned_text = esc_doc(text) if text else ""
            if cleaned_text:
                lines.append(f'        doc "{cleaned_text}";')
            lines.append("    }")

    # 3f. Connections
    if conn_pairs:
        lines.append("")
        lines.append("    /* === Connections === */")
        seen = set()
        for f, t in conn_pairs:
            key = f"{f}->{t}"
            if key not in seen:
                seen.add(key)
                lines.append(f"    connect {f} to {t};")

    lines.append("}")
    return '\n'.join(lines)


async def main():
    client = OpenAI(api_key=config.settings.OPENAI_API_KEY,
                    base_url=config.settings.OPENAI_API_URL, timeout=300)
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
            hc = len(data.get("hypervariables", {}))
            sc = len(data.get("sections", []))
            bar.set_postfix_str(f"{pc}p {hc}hv {sc}s")
            bar.update(1)

    total_p = sum(len(d.get("parts", [])) for d in all_data)
    total_h = sum(len(d.get("hypervariables", {})) for d in all_data)
    total_s = sum(len(d.get("sections", [])) for d in all_data)
    total_c = sum(len(d.get("conns", [])) for d in all_data)
    print(f"  Knowledge: {total_p} parts, {total_h} HVs, {total_s} sections, {total_c} connections")

    sysml = build_sysml(all_data, doc.page_count)

    from sysml.sysml_parser import parse_sysml_text
    try:
        parse_sysml_text(sysml)
        print(f"  Parser: OK")
    except Exception as e:
        print(f"  Parser: {str(e)[:200]}")

    kw = re.findall(r'attribute\s+(\w+)', sysml)
    conflicts = [a for a in kw if a.lower() in _SYSML_KW]
    print(f"  Keywords: {'OK' if not conflicts else str(conflicts)}")

    kg_path = Path("database/AIOPS/knowledge_graph.sysml")
    kg_path.write_text(sysml, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n  Saved: {kg_path}")
    print(f"  {len(sysml)} chars, {len(sysml.splitlines())} lines, {elapsed:.0f}s")
    print(f"\n{sysml}")


if __name__ == "__main__":
    asyncio.run(main())
