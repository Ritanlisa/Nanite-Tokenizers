# Fix KG Build Section Granularity

## TL;DR

`_build_section_map` groups page nodes by `parent_id`, producing ~6 sections each with 2500+ pages — LLM context too small to process meaningfully. Fix: group by `node_id` so each of 200 leaf nodes is its own section (~65K chars each, processable).

## Changes

### `agent/kg_build_agent.py:2124-2133` — `_build_section_map`

Replace `parent_id` grouping with `node_id` grouping:

```python
def _build_section_map(self, tree_state):
    sections = {}
    for nid in tree_state._all_page_ids:
        if nid not in tree_state.nodes:
            continue
        node = tree_state.nodes[nid]
        sections[nid] = [node]
    return sections
```

This produces ~200 sections instead of ~6, each with 1 page node (~65K chars).
Combined with the earlier fallback fix, all 200 sections will be processed.
