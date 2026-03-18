from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List


def tokenize_keyword_terms(text: str) -> List[str]:
    if not text:
        return []

    normalized = str(text).lower()
    english_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", normalized)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)

    stop_terms = {
        "and", "the", "for", "with", "from", "that", "this", "are", "was", "were", "have", "has",
        "had", "not", "you", "your", "our", "their", "they", "them", "its", "into", "than", "then",
        "also", "can", "could", "would", "should", "will", "shall", "may", "might", "about", "after",
        "before", "above", "below", "there", "here", "when", "where", "while", "which", "what", "who",
        "how", "why", "all", "any", "each", "other", "some", "more", "most", "such", "only", "very",
        "使用", "功能", "可以", "进行", "如果", "一个", "一种", "我们", "你们", "他们", "以及", "通过", "相关",
        "当前", "文档", "内容", "说明", "支持", "需要", "用于", "配置", "系统", "页面", "章节", "部分", "其中",
    }

    tokens: List[str] = []
    for token in english_terms + chinese_terms:
        item = token.strip("_-")
        if len(item) < 2:
            continue
        if item in stop_terms:
            continue
        if item.isdigit():
            continue
        tokens.append(item)
    return tokens


def tfidf_extract(
    texts_by_doc_name: Dict[str, List[str]],
    *,
    top_k: int = 12,
) -> Dict[str, List[str]]:
    if not texts_by_doc_name:
        return {}

    docs_tokens: Dict[str, List[str]] = {}
    docs_counter: Dict[str, Counter[str]] = {}
    for doc_name, texts in texts_by_doc_name.items():
        merged_text = "\n".join(part for part in texts if part)
        tokens = tokenize_keyword_terms(merged_text)
        if not tokens:
            continue
        docs_tokens[doc_name] = tokens
        docs_counter[doc_name] = Counter(tokens)

    total_docs = len(docs_counter)
    if total_docs == 0:
        return {name: [] for name in texts_by_doc_name.keys()}

    doc_freq: Counter[str] = Counter()
    for tokens in docs_tokens.values():
        doc_freq.update(set(tokens))

    keyword_map: Dict[str, List[str]] = {}
    limit = max(1, min(int(top_k), 30))
    for doc_name, counter in docs_counter.items():
        token_total = sum(counter.values())
        if token_total <= 0:
            keyword_map[doc_name] = []
            continue

        scored: List[tuple[str, float]] = []
        for token, freq in counter.items():
            df = max(1, int(doc_freq.get(token) or 1))
            tf = float(freq) / float(token_total)
            idf = math.log((1.0 + total_docs) / (1.0 + df)) + 1.0
            scored.append((token, tf * idf))

        scored.sort(key=lambda item: item[1], reverse=True)
        keyword_map[doc_name] = [term for term, _ in scored[:limit]]

    for doc_name in texts_by_doc_name.keys():
        keyword_map.setdefault(doc_name, [])
    return keyword_map
