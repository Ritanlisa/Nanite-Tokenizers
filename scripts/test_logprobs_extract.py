from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
import webbrowser
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_RAG_EXTENSIONS: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".xlsx",
    ".xls",
    ".csv",
}


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", str(name).strip())
    return cleaned or "document"


def build_texts_by_doc_name(file_path: str) -> Dict[str, List[str]]:
    from rag.documents import load_rag_documents_from_paths

    rag_docs = load_rag_documents_from_paths([file_path], SUPPORTED_RAG_EXTENSIONS)
    if not rag_docs:
        raise RuntimeError(f"无法加载文档: {file_path}")

    texts_by_doc_name: Dict[str, List[str]] = {}
    for rag_doc in rag_docs:
        doc_name = str(getattr(rag_doc, "doc_name", "") or "").strip()
        if not doc_name:
            continue

        parts: List[str] = []
        title = str(getattr(rag_doc, "title", "") or "").strip()
        if title:
            parts.append(title)

        cleaned_text = str(getattr(rag_doc, "cleaned_text", "") or "").strip()
        if cleaned_text:
            parts.append(cleaned_text)

        if parts:
            texts_by_doc_name.setdefault(doc_name, []).append("\n".join(parts))

    if not texts_by_doc_name:
        raise RuntimeError("文档已加载，但没有可用于关键词提取的文本")
    return texts_by_doc_name


def show_chart(image_path: Path) -> None:
    try:
        matplotlib = importlib.import_module("matplotlib")
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception as exc:
        raise RuntimeError(
            f"无法显示图表，缺少 matplotlib 或图形环境不可用: {exc}"
        ) from exc

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\).*", category=UserWarning)

    image = plt.imread(str(image_path))
    plt.figure(figsize=(14, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"logprobs 分布图: {image_path.name}")
    plt.tight_layout()

    backend = str(matplotlib.get_backend() or "").lower()
    if "agg" in backend:
        print(f"当前 matplotlib 后端为 {backend}，无法弹窗显示，已生成图像：{image_path}")
        plt.close()
        return

    plt.show()


def _plot_logprob_bars(
    *,
    doc_name: str,
    probs: List[float],
    minus_log2_probs: List[float],
    sampled_probs: List[float],
    sampled_minus_log2_probs: List[float],
    jieba_probs: List[float],
    jieba_minus_log2_probs: List[float],
    softmax_denominators: List[float],
    softmax_denominator_log2: List[float],
    top_k: int,
    top_p: float,
    out_path: Path,
) -> None:
    if not probs:
        return

    eps = 1e-12
    safe_logits = [
        float(math.log(value))
        for value in probs
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    safe_inv_neg_logprobs = [
        float(-1.0 / value)
        for value in safe_logits
        if math.isfinite(value) and value < -eps
    ]
    safe_probs = [
        max(float(value), eps)
        for value in probs
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    safe_minus_log2 = [
        float(value)
        for value in minus_log2_probs
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    safe_inv_minus_log2 = [
        float(1.0 / value)
        for value in safe_minus_log2
        if math.isfinite(value) and value > eps
    ]
    safe_sampled_probs = [
        float(value)
        for value in sampled_probs
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    safe_sampled_minus_log2 = [
        float(value)
        for value in sampled_minus_log2_probs
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    safe_jieba_probs = [
        float(value)
        for value in jieba_probs
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    safe_jieba_minus_log2 = [
        float(value)
        for value in jieba_minus_log2_probs
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    safe_softmax_denominators = [
        float(value)
        for value in softmax_denominators
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    safe_softmax_denominator_log2 = [
        float(value)
        for value in softmax_denominator_log2
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]

    count = min(
        len(safe_probs),
        len(safe_minus_log2),
        len(safe_logits),
        len(safe_inv_neg_logprobs),
        len(safe_inv_minus_log2),
        len(safe_sampled_probs),
        len(safe_sampled_minus_log2),
    )
    if count <= 0:
        print(f"文档 {doc_name} 的可视化数据无有效值，跳过绘图。")
        return

    safe_logits = safe_logits[:count]
    safe_probs = safe_probs[:count]
    safe_minus_log2 = safe_minus_log2[:count]
    safe_inv_neg_logprobs = safe_inv_neg_logprobs[:count]
    safe_inv_minus_log2 = safe_inv_minus_log2[:count]
    safe_sampled_probs = safe_sampled_probs[:count]
    safe_sampled_minus_log2 = safe_sampled_minus_log2[:count]

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Microsoft YaHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\).*", category=UserWarning)
    except Exception as exc:
        print(f"matplotlib 不可用，跳过绘图：{exc}")
        return

    fig, axes = plt.subplots(5, 2, figsize=(16, 32))
    bins = min(360, max(120, int(math.sqrt(count) * 2)))

    raw_log_counts, raw_log_edges = np.histogram(np.asarray(safe_logits, dtype=np.float64), bins=bins)
    raw_log_centers = (raw_log_edges[:-1] + raw_log_edges[1:]) / 2.0
    axes[0, 0].plot(raw_log_centers, raw_log_counts / float(count), linewidth=1.4, label="logits")
    axes[0, 0].set_xlabel("logits")
    axes[0, 0].set_ylabel("Normalized frequency")
    axes[0, 0].set_title(f"{doc_name} - logits distribution")
    axes[0, 0].legend()

    raw_m2_counts, raw_m2_edges = np.histogram(np.asarray(safe_minus_log2, dtype=np.float64), bins=bins)
    raw_m2_centers = (raw_m2_edges[:-1] + raw_m2_edges[1:]) / 2.0
    axes[0, 1].plot(raw_m2_centers, raw_m2_counts / float(count), linewidth=1.4, label="-log2(probs)")
    axes[0, 1].set_xlabel("-log2(probs)")
    axes[0, 1].set_ylabel("Normalized frequency")
    axes[0, 1].set_title(f"{doc_name} - -log2(probs) distribution")
    axes[0, 1].legend()

    sampled_counts, sampled_edges = np.histogram(np.asarray(safe_sampled_probs, dtype=np.float64), bins=bins)
    sampled_centers = (sampled_edges[:-1] + sampled_edges[1:]) / 2.0
    axes[1, 0].plot(
        sampled_centers,
        sampled_counts / float(count),
        linewidth=1.4,
        label=f"sampled probs (top_k={top_k}, top_p={top_p})",
    )
    axes[1, 0].set_xlabel("probs after top-k/top-p softmax")
    axes[1, 0].set_ylabel("Normalized frequency")
    axes[1, 0].set_title(f"{doc_name} - sampled probs distribution")
    axes[1, 0].legend()

    sampled_m2_counts, sampled_m2_edges = np.histogram(np.asarray(safe_sampled_minus_log2, dtype=np.float64), bins=bins)
    sampled_m2_centers = (sampled_m2_edges[:-1] + sampled_m2_edges[1:]) / 2.0
    axes[1, 1].plot(
        sampled_m2_centers,
        sampled_m2_counts / float(count),
        linewidth=1.4,
        label=f"sampled -log2(probs) (top_k={top_k}, top_p={top_p})",
    )
    axes[1, 1].set_xlabel("-log2(sampled probs)")
    axes[1, 1].set_ylabel("Normalized frequency")
    axes[1, 1].set_title(f"{doc_name} - sampled -log2(probs) distribution")
    axes[1, 1].legend()

    raw_log_counts, raw_log_edges = np.histogram(np.asarray(safe_inv_neg_logprobs, dtype=np.float64), bins=bins)
    raw_log_centers = (raw_log_edges[:-1] + raw_log_edges[1:]) / 2.0
    axes[2, 0].plot(raw_log_centers, raw_log_counts / float(count), linewidth=1.4, label="-1/logprobs")
    axes[2, 0].set_xlabel("-1/logprobs")
    axes[2, 0].set_ylabel("Normalized frequency")
    axes[2, 0].set_title(f"{doc_name} - -1/logprobs distribution")
    axes[2, 0].legend()

    raw_m2_counts, raw_m2_edges = np.histogram(np.asarray(safe_inv_minus_log2, dtype=np.float64), bins=bins)
    raw_m2_centers = (raw_m2_edges[:-1] + raw_m2_edges[1:]) / 2.0
    axes[2, 1].plot(raw_m2_centers, raw_m2_counts / float(count), linewidth=1.4, label="-1/log2(probs)")
    axes[2, 1].set_xlabel("-1/log2(probs)")
    axes[2, 1].set_ylabel("Normalized frequency")
    axes[2, 1].set_title(f"{doc_name} - -1/log2(probs) distribution")
    axes[2, 1].legend()

    jieba_count = min(len(safe_jieba_probs), len(safe_jieba_minus_log2))
    if jieba_count > 0:
        safe_jieba_probs = safe_jieba_probs[:jieba_count]
        safe_jieba_minus_log2 = safe_jieba_minus_log2[:jieba_count]
        jieba_bins = min(360, max(80, int(math.sqrt(jieba_count) * 2)))

        jieba_counts, jieba_edges = np.histogram(np.asarray(safe_jieba_probs, dtype=np.float64), bins=jieba_bins)
        jieba_centers = (jieba_edges[:-1] + jieba_edges[1:]) / 2.0
        axes[3, 0].plot(jieba_centers, jieba_counts / float(jieba_count), linewidth=1.4, label="jieba probs")
        axes[3, 0].set_xlabel("jieba probs (token probs product)")
        axes[3, 0].set_ylabel("Normalized frequency")
        axes[3, 0].set_title(f"{doc_name} - jieba probs distribution")
        axes[3, 0].legend()

        jieba_m2_counts, jieba_m2_edges = np.histogram(np.asarray(safe_jieba_minus_log2, dtype=np.float64), bins=jieba_bins)
        jieba_m2_centers = (jieba_m2_edges[:-1] + jieba_m2_edges[1:]) / 2.0
        axes[3, 1].plot(jieba_m2_centers, jieba_m2_counts / float(jieba_count), linewidth=1.4, label="jieba -log2(probs)")
        axes[3, 1].set_xlabel("-log2(jieba probs)")
        axes[3, 1].set_ylabel("Normalized frequency")
        axes[3, 1].set_title(f"{doc_name} - jieba -log2(probs) distribution")
        axes[3, 1].legend()
    else:
        axes[3, 0].text(0.5, 0.5, "no jieba probs", ha="center", va="center")
        axes[3, 0].set_axis_off()
        axes[3, 1].text(0.5, 0.5, "no jieba -log2(probs)", ha="center", va="center")
        axes[3, 1].set_axis_off()

    softmax_count = min(len(safe_softmax_denominators), len(safe_softmax_denominator_log2))
    if softmax_count > 0:
        safe_softmax_denominators = safe_softmax_denominators[:softmax_count]
        safe_softmax_denominator_log2 = safe_softmax_denominator_log2[:softmax_count]
        softmax_bins = min(360, max(80, int(math.sqrt(softmax_count) * 2)))

        denom_counts, denom_edges = np.histogram(np.asarray(safe_softmax_denominators, dtype=np.float64), bins=softmax_bins)
        denom_centers = (denom_edges[:-1] + denom_edges[1:]) / 2.0
        axes[4, 0].plot(denom_centers, denom_counts / float(softmax_count), linewidth=1.4, label="softmax denominator")
        axes[4, 0].set_xlabel("softmax denominator")
        axes[4, 0].set_ylabel("Normalized frequency")
        axes[4, 0].set_title(f"{doc_name} - softmax denominator distribution")
        axes[4, 0].legend()

        denom_log2_counts, denom_log2_edges = np.histogram(np.asarray(safe_softmax_denominator_log2, dtype=np.float64), bins=softmax_bins)
        denom_log2_centers = (denom_log2_edges[:-1] + denom_log2_edges[1:]) / 2.0
        axes[4, 1].plot(denom_log2_centers, denom_log2_counts / float(softmax_count), linewidth=1.4, label="log2(softmax denominator)")
        axes[4, 1].set_xlabel("log2(softmax denominator)")
        axes[4, 1].set_ylabel("Normalized frequency")
        axes[4, 1].set_title(f"{doc_name} - log2(softmax denominator) distribution")
        axes[4, 1].legend()
    else:
        axes[4, 0].text(0.5, 0.5, "no softmax denominators", ha="center", va="center")
        axes[4, 0].set_axis_off()
        axes[4, 1].text(0.5, 0.5, "no log2(softmax denominator)", ha="center", va="center")
        axes[4, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=240)
    plt.close(fig)


def _load_logprobs_payload(json_path: Path) -> Dict[str, object]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"无效 JSON 内容: {json_path}")
    return payload


def _calc_ab_roles(
    tokens: List[str],
    probs: List[float],
    a_prob_upper: float,
    b_prob_lower: float,
) -> tuple[Dict[int, str], List[str]]:
    n = min(len(tokens), len(probs))
    roles: Dict[int, str] = {}
    phrases: List[str] = []

    for idx in range(0, n):
        a_prob = float(probs[idx])
        if not math.isfinite(a_prob):
            continue
        if a_prob > a_prob_upper:
            continue

        roles[idx] = "A"
        parts = [str(tokens[idx])]

        tail = idx + 1
        max_tail = min(n, idx + 24)
        while tail < max_tail:
            p_tail = float(probs[tail])
            token_tail = str(tokens[tail])
            if (not math.isfinite(p_tail)) or p_tail < b_prob_lower:
                break
            if "\n" in token_tail:
                break
            roles[tail] = "B"
            parts.append(token_tail)
            if re.search(r"[。！？.!?]", token_tail):
                break
            tail += 1

        phrase = re.sub(r"\s+", " ", "".join(parts).strip())
        if not phrase:
            fallback_token = str(tokens[idx]).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").strip()
            phrase = fallback_token if fallback_token else f"<WS@{idx}>"
        phrases.append(phrase)

    return roles, phrases


def _preview_token(token: str, limit: int = 80) -> str:
    escaped = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    if len(escaped) > limit:
        return escaped[:limit] + "..."
    return escaped


def _escape_html(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_float_full(value: float) -> str:
    return format(float(value), ".17g")


def _map_token_to_jieba_indices(tokens: List[str], jieba_words: List[str]) -> List[int]:
    n_tokens = len(tokens)
    if n_tokens <= 0:
        return []
    if not jieba_words:
        return [-1 for _ in range(n_tokens)]

    def _normalize(value: str) -> str:
        text = str(value)
        text = text.replace("▁", "").replace("Ġ", "")
        text = re.sub(r"^##", "", text)
        text = re.sub(r"\s+", "", text)
        return text

    mapping = [-1 for _ in range(n_tokens)]
    token_cursor = 0

    for jieba_idx, word in enumerate(jieba_words):
        if token_cursor >= n_tokens:
            break

        target = str(word)
        target_norm = _normalize(target)
        merged = ""
        merged_norm = ""
        start = token_cursor

        while token_cursor < n_tokens:
            merged += str(tokens[token_cursor])
            merged_norm = _normalize(merged)
            token_cursor += 1

            if merged == target:
                break
            if target_norm and merged_norm == target_norm:
                break
            if target_norm and len(merged_norm) >= len(target_norm):
                break
            if (not target_norm) and len(merged) >= len(target):
                break

        end = token_cursor
        if end <= start and start < n_tokens:
            end = start + 1
            token_cursor = end

        for tok_idx in range(start, min(end, n_tokens)):
            mapping[tok_idx] = jieba_idx

    jieba_norm = [_normalize(word) for word in jieba_words]
    corrected: List[int] = []
    prev_idx = 0
    for tok_idx, token in enumerate(tokens):
        token_norm = _normalize(token)
        mapped_idx = mapping[tok_idx] if tok_idx < len(mapping) else -1

        if token_norm and 0 <= mapped_idx < len(jieba_words):
            cur_norm = jieba_norm[mapped_idx]
            if (token_norm in cur_norm) or (cur_norm in token_norm):
                corrected.append(mapped_idx)
                prev_idx = mapped_idx
                continue

        candidate_idx = -1
        if token_norm:
            for j in range(max(0, prev_idx - 3), min(len(jieba_words), prev_idx + 4)):
                cur_norm = jieba_norm[j]
                if (token_norm in cur_norm) or (cur_norm in token_norm):
                    candidate_idx = j
                    break
            if candidate_idx < 0:
                for j, cur_norm in enumerate(jieba_norm):
                    if (token_norm in cur_norm) or (cur_norm in token_norm):
                        candidate_idx = j
                        break

        corrected.append(candidate_idx if candidate_idx >= 0 else mapped_idx)
        if candidate_idx >= 0:
            prev_idx = candidate_idx

    return corrected


def _build_python_stopword_filter_for_debug(jieba_words: List[str]) -> tuple[List[bool], bool]:
    if not jieba_words:
        return [], False

    try:
        from rag.logprob_keyword_extractor import _is_noise_token, _load_jieba_open_source_stop_words
    except Exception as exc:
        print(f"警告：无法加载 Python 停用词规则，debug 前端将不启用该过滤：{exc}")
        return [False for _ in jieba_words], False

    try:
        stop_words = _load_jieba_open_source_stop_words()
    except Exception as exc:
        print(f"警告：加载 Python 停用词失败，debug 前端将不启用该过滤：{exc}")
        return [False for _ in jieba_words], False

    flags: List[bool] = []
    for word in jieba_words:
        token = str(word).strip()
        filtered = _is_noise_token(token) or token.lower() in stop_words
        flags.append(bool(filtered))

    return flags, True


def show_debug_gui_from_payload(payload: Dict[str, object]) -> None:
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, list):
        raw_tokens = []
    tokens = [str(item) for item in raw_tokens]

    raw_logprobs = payload.get("logprobs")
    if not isinstance(raw_logprobs, list):
        raw_logprobs = []
    probs = [
        max(math.exp(float(item)), 1e-12)
        for item in raw_logprobs
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]
    prob_source = "exp(logprobs)"
    if not probs:
        raw_probs = payload.get("sampled_probs")
        if not isinstance(raw_probs, list):
            raw_probs = []
        probs = [float(item) for item in raw_probs if isinstance(item, (int, float)) and math.isfinite(float(item))]
        prob_source = "sampled_probs(回退)"
    if not tokens or not probs:
        print("调试GUI跳过：tokens 或 probs 为空")
        return

    n = min(len(tokens), len(probs))
    tokens = tokens[:n]
    probs = probs[:n]
    logits = [float(math.log(max(p, 1e-12))) for p in probs]
    minus_log2_probs = [float(-math.log2(max(p, 1e-12))) for p in probs]
    raw_softmax_denominators = payload.get("softmax_denominators")
    if not isinstance(raw_softmax_denominators, list):
        raw_softmax_denominators = []
    softmax_denominators = [
        max(float(item), 1e-300)
        for item in raw_softmax_denominators
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]
    if len(softmax_denominators) < n:
        softmax_denominators.extend([1e-300] * (n - len(softmax_denominators)))
    softmax_denominators = softmax_denominators[:n]
    softmax_denominator_log2 = [float(math.log2(max(item, 1e-300))) for item in softmax_denominators]

    minus_log2_logits: List[Optional[float]] = []
    for value in logits:
        if value > 0.0 and math.isfinite(value):
            minus_log2_logits.append(float(-math.log2(value)))
        else:
            minus_log2_logits.append(None)

    raw_jieba_words = payload.get("jieba_words")
    if not isinstance(raw_jieba_words, list):
        raw_jieba_words = []
    jieba_words = [str(item) for item in raw_jieba_words]

    raw_jieba_probs = payload.get("jieba_probs")
    if not isinstance(raw_jieba_probs, list):
        raw_jieba_probs = []
    jieba_probs = [
        max(float(item), 1e-300)
        for item in raw_jieba_probs
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]

    jieba_n = min(len(jieba_words), len(jieba_probs))
    jieba_words = jieba_words[:jieba_n]
    jieba_probs = jieba_probs[:jieba_n]
    jieba_filtered_flags, python_stopword_filter_available = _build_python_stopword_filter_for_debug(jieba_words)
    if len(jieba_filtered_flags) < jieba_n:
        jieba_filtered_flags.extend([False] * (jieba_n - len(jieba_filtered_flags)))
    jieba_filtered_flags = jieba_filtered_flags[:jieba_n]
    raw_token_to_jieba_idx = payload.get("token_to_jieba_idx")
    mapping_source = "fallback"
    if isinstance(raw_token_to_jieba_idx, list):
        mapped = [
            int(item) if isinstance(item, (int, float)) and 0 <= int(item) < jieba_n else -1
            for item in raw_token_to_jieba_idx[: len(tokens)]
        ]
        if len(mapped) < len(tokens):
            mapped.extend([-1] * (len(tokens) - len(mapped)))

        if jieba_n > 0:
            last_valid = -1
            for idx in range(len(mapped)):
                current = mapped[idx]
                if 0 <= current < jieba_n:
                    last_valid = current
                elif last_valid >= 0:
                    mapped[idx] = last_valid

            next_valid = -1
            for idx in range(len(mapped) - 1, -1, -1):
                current = mapped[idx]
                if 0 <= current < jieba_n:
                    next_valid = current
                elif next_valid >= 0:
                    mapped[idx] = next_valid

            for idx in range(len(mapped)):
                if mapped[idx] < 0:
                    mapped[idx] = 0

        token_to_jieba_idx = mapped
        mapping_source = "backend"
    else:
        token_to_jieba_idx = _map_token_to_jieba_indices(tokens, jieba_words)

    roles: Dict[int, str] = {}

    doc_name = str(payload.get("doc_name") or "(unknown)")
    output_dir = Path("tmp") / "logprobs_extract"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(doc_name)
    html_path = output_dir / f"{safe_name}.debug.html"

    jieba_words_js_literal = json.dumps([str(v) for v in jieba_words], ensure_ascii=False)
    jieba_probs_js_literal = json.dumps([float(v) for v in jieba_probs], ensure_ascii=False)
    jieba_filtered_flags_js_literal = json.dumps([bool(v) for v in jieba_filtered_flags], ensure_ascii=False)
    initial_top_k_raw = payload.get("top_k")
    if isinstance(initial_top_k_raw, (int, float)):
        initial_top_k = int(initial_top_k_raw)
    else:
        raw_ab_star = payload.get("ab_star_phrases")
        if isinstance(raw_ab_star, list) and len(raw_ab_star) > 0:
            initial_top_k = int(len(raw_ab_star))
        else:
            initial_top_k = 50
    initial_top_k = max(1, initial_top_k)

    token_spans: List[str] = []
    for idx, token in enumerate(tokens):
        role = roles.get(idx, "-")
        role_class = "role-a" if role == "A" else ("role-b" if role == "B" else "")
        minus_log2_logit_value = minus_log2_logits[idx]
        minus_log2_logit_text = (
            _fmt_float_full(minus_log2_logit_value)
            if isinstance(minus_log2_logit_value, float)
            else "N/A (logits<=0)"
        )
        jieba_idx = token_to_jieba_idx[idx] if idx < len(token_to_jieba_idx) else -1
        if 0 <= jieba_idx < jieba_n:
            jieba_word = str(jieba_words[jieba_idx])
            jieba_prob = float(jieba_probs[jieba_idx])
            jieba_logprob = float(math.log(max(jieba_prob, 1e-300)))
            jieba_word_text = _preview_token(jieba_word)
            jieba_prob_text = _fmt_float_full(jieba_prob)
            jieba_logprob_text = _fmt_float_full(jieba_logprob)
        else:
            jieba_word_text = "N/A"
            jieba_prob_text = "N/A"
            jieba_logprob_text = "N/A"
        title = (
            f"idx={idx}\n"
            f"token='{_preview_token(token)}'\n"
            f"logits={_fmt_float_full(logits[idx])}\n"
            f"-log2(logits)={minus_log2_logit_text}\n"
            f"probs={_fmt_float_full(probs[idx])}\n"
            f"-log2(probs)={_fmt_float_full(minus_log2_probs[idx])}\n"
            f"softmax_denominator={_fmt_float_full(softmax_denominators[idx])}\n"
            f"log2(softmax_denominator)={_fmt_float_full(softmax_denominator_log2[idx])}\n"
            f"jieba_token='{jieba_word_text}'\n"
            f"jieba_probs={jieba_prob_text}\n"
            f"jieba_logprobs={jieba_logprob_text}\n"
            f"type={role}"
        )
        token_spans.append(
            f"<span class='tok {role_class}' data-idx='{idx}' data-prob='{_fmt_float_full(probs[idx])}' data-token=\"{_escape_html(token)}\" title=\"{_escape_html(title)}\">{_escape_html(token)}</span>"
        )

    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Logprobs Debug - {_escape_html(Path(doc_name).name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 12px; }}
    .meta {{ margin-bottom: 10px; color: #333; font-size: 14px; }}
    .hint {{ margin-bottom: 10px; color: #666; font-size: 13px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    .ctrl {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }}
    .ctrl input[type=range] {{ width: 240px; }}
    .btn {{ border: 1px solid #bbb; border-radius: 6px; background: #f7f7f7; padding: 4px 10px; cursor: pointer; }}
    .stats {{ margin-bottom: 10px; color: #444; font-size: 13px; }}
    .phrases {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; margin-bottom: 10px; max-height: 24vh; overflow: auto; font-family: 'DejaVu Sans Mono', monospace; font-size: 12px; white-space: pre; line-height: 1.35; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; max-height: 78vh; overflow: auto; }}
    .text {{ white-space: pre-wrap; word-break: break-word; font-family: 'DejaVu Sans Mono', 'Noto Sans Mono CJK SC', monospace; line-height: 1.4; font-size: 13px; }}
    .tok {{ cursor: default; }}
    .role-a {{ color: red; }}
    .role-b {{ color: #d9a300; }}
  </style>
</head>
<body>
    <div class=\"meta\">doc={_escape_html(doc_name)} | token_count={n} | jieba_count={jieba_n} | probs来源={prob_source} | 映射来源={mapping_source} | AB*=已禁用</div>
  <div class=\"hint\">将鼠标悬停在 token 上查看 prob/logprob；当前列表按 jieba 分词聚合概率排序。</div>
    <div class="controls">
                <label class="ctrl"><input id="usePythonStopwords" type="checkbox" checked /> 使用 Python 停用词规则</label>
        <label class="ctrl">排序字段
            <select id="sortBy">
                <option value="jieba_idx">jieba位置</option>
                <option value="jieba_prob">jieba-probs</option>
                <option value="jieba_logit">jieba-logits</option>
                <option value="jieba_sum_minus_log2_unique">Sum(-log2(jieba probs), 去重token)</option>
                <option value="jieba_avg_minus_log2_unique">Aveg(-log2(jieba probs), 去重token)</option>
                <option value="jieba_sum_minus_log2_unique_plus">Sum(-log2(jieba probs)/log2(cnt+1), 去重token)</option>
                <option value="jieba_square_surprise_unique" selected>平方惊喜度(Σ惊喜度², 去重token)</option>
                <option value="jieba_max_surprise_unique">最大惊喜度(max 惊喜度, 去重token)</option>
                <option value="jieba_geometric_surprise_unique">几何惊喜度(惊喜度之积, 去重token)</option>
                <option value="jieba_adjusted_geometric_surprise_unique">调整几何惊喜度((惊喜度+1)之积, 去重token)</option>
                <option value="jieba_harmonic_surprise_unique">调和惊喜度(词频*调和平均惊喜度, 去重token)</option>
            </select>
        </label>
        <label class="ctrl"><input id="showSelectedMetricOnly" type="checkbox" checked /> 仅显示当前排序字段</label>
        <label class="ctrl">顺序
            <select id="sortOrder">
                <option value="asc">升序</option>
                <option value="desc" selected>降序</option>
            </select>
        </label>
        <label class="ctrl">topk
            <input id="topKInput" type="number" min="1" step="1" value="{initial_top_k}" style="width:120px;" />
        </label>
        <button id="resetBtn" class="btn" type="button">复位默认阈值</button>
    </div>
    <div id="stats" class="stats"></div>
    <div id="phrases" class="phrases"></div>
  <div class=\"panel\"><div class=\"text\">{''.join(token_spans)}</div></div>

    <script>
        const DEFAULT_TOP_K = {initial_top_k};
        const JIEBA_WORDS = {jieba_words_js_literal};
        const JIEBA_PROBS = {jieba_probs_js_literal};
        const JIEBA_FILTERED_FLAGS = {jieba_filtered_flags_js_literal};
        const PYTHON_STOPWORD_FILTER_AVAILABLE = {str(bool(python_stopword_filter_available)).lower()};

        const sortBy = document.getElementById('sortBy');
        const sortOrder = document.getElementById('sortOrder');
        const usePythonStopwords = document.getElementById('usePythonStopwords');
        const showSelectedMetricOnly = document.getElementById('showSelectedMetricOnly');
        const topKInput = document.getElementById('topKInput');
        const stats = document.getElementById('stats');
        const phrasesEl = document.getElementById('phrases');
        const resetBtn = document.getElementById('resetBtn');

        usePythonStopwords.disabled = !PYTHON_STOPWORD_FILTER_AVAILABLE;
        if (!PYTHON_STOPWORD_FILTER_AVAILABLE) {{
            usePythonStopwords.checked = false;
            usePythonStopwords.title = 'Python 停用词规则不可用（导入失败）';
        }}

        function escapeHtml(v) {{
            return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\"/g, '&quot;');
        }}

        function formatAlignedColumns(columns) {{
            const normalized = columns.map(function(item) {{
                const key = String(item[0] ?? '');
                const value = String(item[1] ?? '');
                return [key, value];
            }});
            const keyWidth = normalized.reduce(function(maxV, pair) {{
                return Math.max(maxV, pair[0].length);
            }}, 0);
            return normalized
                .map(function(pair) {{
                    return pair[0].padEnd(keyWidth, ' ') + ' = ' + pair[1];
                }})
                .join(' | ');
        }}

        function recompute() {{
            const sortField = String(sortBy.value || 'jieba_prob');
            const order = String(sortOrder.value || 'desc');
            const enablePythonStopwordFilter = Boolean(usePythonStopwords.checked) && PYTHON_STOPWORD_FILTER_AVAILABLE;
            const selectedOnly = Boolean(showSelectedMetricOnly.checked);
            const topK = Math.max(1, Math.floor(Number(topKInput.value) || DEFAULT_TOP_K));
            topKInput.value = String(topK);
            const n = Math.min(JIEBA_WORDS.length, JIEBA_PROBS.length);
            const phraseItems = [];
            let filteredOutCount = 0;
            for (let i = 0; i < n; i++) {{
                const prob = Number(JIEBA_PROBS[i]);
                if (!Number.isFinite(prob)) continue;
                if (enablePythonStopwordFilter && Boolean(JIEBA_FILTERED_FLAGS[i])) {{
                    filteredOutCount += 1;
                    continue;
                }}
                phraseItems.push({{
                    jieba_idx: i,
                    jieba_prob: prob,
                    jieba_logit: Math.log(Math.max(prob, 1e-300)),
                    text: String(JIEBA_WORDS[i] ?? ''),
                }});
            }}

            phraseItems.sort(function(x, y) {{
                const vx = Number(x[sortField]);
                const vy = Number(y[sortField]);
                const sign = order === 'desc' ? -1 : 1;
                if (!Number.isFinite(vx) && !Number.isFinite(vy)) return 0;
                if (!Number.isFinite(vx)) return 1;
                if (!Number.isFinite(vy)) return -1;
                if (vx < vy) return -1 * sign;
                if (vx > vy) return 1 * sign;
                if (x.jieba_idx < y.jieba_idx) return -1;
                if (x.jieba_idx > y.jieba_idx) return 1;
                return 0;
            }});

            let topItems = [];
            if (
                sortField === 'jieba_sum_minus_log2_unique' ||
                sortField === 'jieba_avg_minus_log2_unique' ||
                sortField === 'jieba_sum_minus_log2_unique_plus' ||
                sortField === 'jieba_square_surprise_unique' ||
                sortField === 'jieba_max_surprise_unique' ||
                sortField === 'jieba_geometric_surprise_unique' ||
                sortField === 'jieba_adjusted_geometric_surprise_unique' ||
                sortField === 'jieba_harmonic_surprise_unique'
            ) {{
                const grouped = new Map();
                for (const item of phraseItems) {{
                    const token = String(item.text || '');
                    if (!token) continue;
                    const p = Number(item.jieba_prob);
                    if (!Number.isFinite(p) || p <= 0) continue;
                    const surprise = -Math.log2(Math.max(p, 1e-300));
                    const safeSurprise = Math.max(surprise, 1e-12);
                    if (!grouped.has(token)) {{
                        grouped.set(token, {{
                            text: token,
                            jieba_idx: Number(item.jieba_idx),
                            sum_minus_log2: surprise,
                            sum_square_surprise: safeSurprise * safeSurprise,
                            max_surprise: safeSurprise,
                            sum_inv_surprise: 1.0 / safeSurprise,
                            log_product_surprise: Math.log(safeSurprise),
                            log_product_surprise_plus_one: Math.log(safeSurprise + 1.0),
                            count: 1,
                        }});
                    }} else {{
                        const cur = grouped.get(token);
                        cur.sum_minus_log2 += surprise;
                        cur.sum_square_surprise += safeSurprise * safeSurprise;
                        cur.max_surprise = Math.max(Number(cur.max_surprise), safeSurprise);
                        cur.sum_inv_surprise += 1.0 / safeSurprise;
                        cur.log_product_surprise += Math.log(safeSurprise);
                        cur.log_product_surprise_plus_one += Math.log(safeSurprise + 1.0);
                        cur.count += 1;
                        if (Number(item.jieba_idx) < Number(cur.jieba_idx)) {{
                            cur.jieba_idx = Number(item.jieba_idx);
                        }}
                    }}
                }}

                const groupedItems = Array.from(grouped.values());
                for (const item of groupedItems) {{
                    const cnt = Math.max(1, Number(item.count) || 1);
                    item.avg_minus_log2 = Number(item.sum_minus_log2) / cnt;
                    const denom = Math.log2(cnt + 1);
                    item.sum_minus_log2_plus = Number(item.sum_minus_log2) / Math.max(denom, 1e-12);
                    item.square_surprise = Number(item.sum_square_surprise);
                    item.max_surprise = Number(item.max_surprise);

                    const expLimit = 700.0;
                    const logProd = Number(item.log_product_surprise);
                    const logProdPlusOne = Number(item.log_product_surprise_plus_one);
                    item.geometric_surprise = logProd > expLimit ? Infinity : Math.exp(logProd);
                    item.adjusted_geometric_surprise = logProdPlusOne > expLimit ? Infinity : Math.exp(logProdPlusOne);

                    const sumInv = Math.max(Number(item.sum_inv_surprise), 1e-12);
                    const harmonicMeanSurprise = cnt / sumInv;
                    item.harmonic_surprise = cnt * harmonicMeanSurprise;
                }}
                groupedItems.sort(function(x, y) {{
                    const sign = order === 'desc' ? -1 : 1;
                    let field = 'sum_minus_log2';
                    if (sortField === 'jieba_avg_minus_log2_unique') {{
                        field = 'avg_minus_log2';
                    }} else if (sortField === 'jieba_sum_minus_log2_unique_plus') {{
                        field = 'sum_minus_log2_plus';
                    }} else if (sortField === 'jieba_square_surprise_unique') {{
                        field = 'square_surprise';
                    }} else if (sortField === 'jieba_max_surprise_unique') {{
                        field = 'max_surprise';
                    }} else if (sortField === 'jieba_geometric_surprise_unique') {{
                        field = 'log_product_surprise';
                    }} else if (sortField === 'jieba_adjusted_geometric_surprise_unique') {{
                        field = 'log_product_surprise_plus_one';
                    }} else if (sortField === 'jieba_harmonic_surprise_unique') {{
                        field = 'harmonic_surprise';
                    }}
                    const vx = Number(x[field]);
                    const vy = Number(y[field]);
                    if (vx < vy) return -1 * sign;
                    if (vx > vy) return 1 * sign;
                    if (x.jieba_idx < y.jieba_idx) return -1;
                    if (x.jieba_idx > y.jieba_idx) return 1;
                    return 0;
                }});
                topItems = groupedItems.slice(0, topK);

                                stats.textContent = 'AB*=已禁用 | jieba词项总数: ' + phraseItems.length + ' | 过滤词项数: ' + filteredOutCount + ' | 去重词项数: ' + groupedItems.length + ' | 显示: top' + topItems.length + ' | 排序: ' + sortField + ' ' + order + ' | Python停用词规则: ' + (enablePythonStopwordFilter ? '开' : '关');
                phrasesEl.innerHTML = topItems
                  .map(function(item, i) {{
                                            const geomText = Number.isFinite(Number(item.geometric_surprise)) ? Number(item.geometric_surprise).toFixed(12) : 'Infinity';
                                            const adjGeomText = Number.isFinite(Number(item.adjusted_geometric_surprise)) ? Number(item.adjusted_geometric_surprise).toFixed(12) : 'Infinity';

                                            const allColumns = [
                                                ['idx', item.jieba_idx],
                                                ['sum(-log2(p))', Number(item.sum_minus_log2).toFixed(12)],
                                                ['aveg(-log2(p))', Number(item.avg_minus_log2).toFixed(12)],
                                                ['sum(-log2(p))/log2(cnt+1)', Number(item.sum_minus_log2_plus).toFixed(12)],
                                                ['平方惊喜度', Number(item.square_surprise).toFixed(12)],
                                                ['最大惊喜度', Number(item.max_surprise).toFixed(12)],
                                                ['几何惊喜度', geomText],
                                                ['调整几何惊喜度', adjGeomText],
                                                ['调和惊喜度', Number(item.harmonic_surprise).toFixed(12)],
                                                ['count', String(item.count)],
                                            ];

                                            const selectedColumnsMap = {{
                                                jieba_sum_minus_log2_unique: [['idx', item.jieba_idx], ['sum(-log2(p))', Number(item.sum_minus_log2).toFixed(12)], ['count', String(item.count)]],
                                                jieba_avg_minus_log2_unique: [['idx', item.jieba_idx], ['aveg(-log2(p))', Number(item.avg_minus_log2).toFixed(12)], ['count', String(item.count)]],
                                                jieba_sum_minus_log2_unique_plus: [['idx', item.jieba_idx], ['sum(-log2(p))/log2(cnt+1)', Number(item.sum_minus_log2_plus).toFixed(12)], ['count', String(item.count)]],
                                                jieba_square_surprise_unique: [['idx', item.jieba_idx], ['平方惊喜度', Number(item.square_surprise).toFixed(12)], ['count', String(item.count)]],
                                                jieba_max_surprise_unique: [['idx', item.jieba_idx], ['最大惊喜度', Number(item.max_surprise).toFixed(12)], ['count', String(item.count)]],
                                                jieba_geometric_surprise_unique: [['idx', item.jieba_idx], ['几何惊喜度', geomText], ['count', String(item.count)]],
                                                jieba_adjusted_geometric_surprise_unique: [['idx', item.jieba_idx], ['调整几何惊喜度', adjGeomText], ['count', String(item.count)]],
                                                jieba_harmonic_surprise_unique: [['idx', item.jieba_idx], ['调和惊喜度', Number(item.harmonic_surprise).toFixed(12)], ['count', String(item.count)]],
                                            }};

                                            const columns = selectedOnly ? (selectedColumnsMap[sortField] || allColumns) : allColumns;
                                            const info = formatAlignedColumns(columns);
                                            return String(i + 1).padStart(4, ' ') + '. ' + escapeHtml(info + ' | token=' + String(item.text || ''));
                  }})
                                      .join(String.fromCharCode(10));
                return;
            }}

            topItems = phraseItems.slice(0, topK);
                        stats.textContent = 'AB*=已禁用 | jieba词项总数: ' + phraseItems.length + ' | 过滤词项数: ' + filteredOutCount + ' | 显示: top' + topItems.length + ' | 排序: ' + sortField + ' ' + order + ' | Python停用词规则: ' + (enablePythonStopwordFilter ? '开' : '关');
            phrasesEl.innerHTML = topItems
              .map(function(item, i) {{
                                    const allColumns = [
                                        ['idx', item.jieba_idx],
                                        ['p', item.jieba_prob.toFixed(12)],
                                        ['logit', item.jieba_logit.toFixed(12)],
                                    ];
                                    const selectedColumnsMap = {{
                                        jieba_idx: [['idx', item.jieba_idx]],
                                        jieba_prob: [['idx', item.jieba_idx], ['p', item.jieba_prob.toFixed(12)]],
                                        jieba_logit: [['idx', item.jieba_idx], ['logit', item.jieba_logit.toFixed(12)]],
                                    }};
                                    const columns = selectedOnly ? (selectedColumnsMap[sortField] || allColumns) : allColumns;
                                    const info = formatAlignedColumns(columns);
                                    return String(i + 1).padStart(4, ' ') + '. ' + escapeHtml(info + ' | token=' + String(item.text || ''));
              }})
                              .join(String.fromCharCode(10));
        }}

        sortBy.addEventListener('change', recompute);
        sortOrder.addEventListener('change', recompute);
        usePythonStopwords.addEventListener('change', recompute);
                showSelectedMetricOnly.addEventListener('change', recompute);
        topKInput.addEventListener('input', recompute);
        resetBtn.addEventListener('click', () => {{
                        sortBy.value = 'jieba_square_surprise_unique';
            sortOrder.value = 'desc';
            usePythonStopwords.checked = PYTHON_STOPWORD_FILTER_AVAILABLE;
                        showSelectedMetricOnly.checked = true;
            topKInput.value = String(DEFAULT_TOP_K);
            recompute();
        }});

        recompute();
    </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"调试GUI页面已生成: {html_path.resolve()}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
    except Exception as exc:
        print(f"无法自动打开浏览器，请手动打开该文件: {exc}")


def main() -> None:
    from rag.line_profiler_instrument import profile_if_enabled, start_profiler, stop_profiler
    from rag.logprob_keyword_extractor import logprobs_extract

    top_k = 50
    profile_enabled = str(os.getenv("ENABLE_KEYWORD_EXTRACTOR_LINE_PROFILER") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    profile_output_path: Optional[str] = None
    if profile_enabled:
        profile_output_path = start_profiler(
            str((Path("tmp") / "logprobs_extract" / "keyword_extractor.lprof").resolve())
        )
        if profile_output_path:
            print(f"line_profiler 已启用，输出文件: {profile_output_path}")
        else:
            print("line_profiler 不可用，改为仅输出 wall-clock 耗时。")

    profiled_logprobs_extract = profile_if_enabled(logprobs_extract)

    files = [
        "/home/ritanlisa/文档/LID.pdf",
        "/home/ritanlisa/文档/浪潮虚拟化InCloud Sphere 6.5.1运维手册.pdf",
        "/home/ritanlisa/文档/湖超-硬件维护手册20231225.doc",
        "/home/ritanlisa/文档/初步验收与试运行分册-6-硬件维护手册 - 1227.doc",
        "/home/ritanlisa/文档/TBP.pdf",
        ]
    try:
        for file in files:
            if not Path(file).exists():
                print(f"文件不存在，跳过: {file}")
                continue
            # 清理CUDA缓存，确保显存充足
            torch.cuda.empty_cache()

            source_file = Path(file).expanduser().resolve()
            if not source_file.exists() or not source_file.is_file():
                raise FileNotFoundError(f"文件不存在: {source_file}")

            texts_by_doc_name = build_texts_by_doc_name(str(source_file))
            keyword_map = profiled_logprobs_extract(texts_by_doc_name, top_k=top_k)

            output_dir = Path("tmp") / "logprobs_extract"
            print(f"关键词结果: {keyword_map}")
            print(f"输出目录: {output_dir.resolve()}")

            for doc_name in texts_by_doc_name.keys():
                safe_name = _sanitize_filename(doc_name)
                json_path = output_dir / f"{safe_name}.json"
                png_path = output_dir / f"{safe_name}.png"
                print(f"JSON: {json_path.resolve()}")
                print(f"PNG:  {png_path.resolve()}")

                if not json_path.exists():
                    raise FileNotFoundError(f"结果 JSON 未生成: {json_path}")

                payload = _load_logprobs_payload(json_path)
                raw_logprobs = payload.get("logprobs")
                if not isinstance(raw_logprobs, list):
                    raw_logprobs = []
                raw_probs = [
                    max(math.exp(float(item)), 1e-12)
                    for item in raw_logprobs
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                raw_sampled_probs = payload.get("sampled_probs")
                if not isinstance(raw_sampled_probs, list):
                    raw_sampled_probs = []
                sampled_probs = [
                    max(float(item), 1e-12)
                    for item in raw_sampled_probs
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                raw_top_p = payload.get("top_p")
                top_p_value = float(raw_top_p) if isinstance(raw_top_p, (int, float)) else 1.0
                raw_jieba_probs = payload.get("jieba_probs")
                if not isinstance(raw_jieba_probs, list):
                    raw_jieba_probs = []
                jieba_probs = [
                    max(float(item), 1e-300)
                    for item in raw_jieba_probs
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                raw_jieba_minus_log2 = payload.get("jieba_minus_log2_probs")
                if not isinstance(raw_jieba_minus_log2, list):
                    raw_jieba_minus_log2 = []
                jieba_minus_log2_probs = [
                    float(item)
                    for item in raw_jieba_minus_log2
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                raw_softmax_denominators = payload.get("softmax_denominators")
                if not isinstance(raw_softmax_denominators, list):
                    raw_softmax_denominators = []
                softmax_denominators = [
                    max(float(item), 1e-300)
                    for item in raw_softmax_denominators
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                raw_softmax_denominator_log2 = payload.get("softmax_denominator_log2")
                if not isinstance(raw_softmax_denominator_log2, list):
                    raw_softmax_denominator_log2 = []
                softmax_denominator_log2 = [
                    float(item)
                    for item in raw_softmax_denominator_log2
                    if isinstance(item, (int, float)) and math.isfinite(float(item))
                ]
                _plot_logprob_bars(
                    doc_name=str(payload.get("doc_name") or doc_name),
                    probs=raw_probs,
                    minus_log2_probs=[-math.log2(prob) for prob in raw_probs],
                    sampled_probs=sampled_probs,
                    sampled_minus_log2_probs=[-math.log2(prob) for prob in sampled_probs],
                    jieba_probs=jieba_probs,
                    jieba_minus_log2_probs=jieba_minus_log2_probs,
                    softmax_denominators=softmax_denominators,
                    softmax_denominator_log2=softmax_denominator_log2,
                    top_k=top_k,
                    top_p=top_p_value,
                    out_path=png_path,
                )

                if not png_path.exists():
                    raise FileNotFoundError(f"图像未生成: {png_path}")
                show_chart(png_path)
                show_debug_gui_from_payload(payload)
    finally:
        dumped_profile = stop_profiler() if profile_enabled else None
        final_profile_path = dumped_profile or profile_output_path
        if final_profile_path:
            print(f"line_profiler 统计文件: {final_profile_path}")
        elif profile_enabled:
            print("line_profiler 未生成统计文件（可能未安装 line_profiler）。")


if __name__ == "__main__":
    main()
