from __future__ import annotations

import json
import importlib
import logging
import math
import os
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
LOGPROB_FLOOR = math.log(1e-12)
_LOGPROB_BACKEND_CACHE: Optional[
    Tuple[
        str,
        Path,
        Callable[
            [str],
            Tuple[
                List[str],
                List[float],
                List[float],
                int,
                float,
                List[str],
                List[float],
                List[int],
                List[float],
            ],
        ],
    ]
] = None
_JIEBA_STOP_WORDS_CACHE: Optional[FrozenSet[str]] = None


def _suppress_jieba_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module=r"jieba(\..*)?",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"jieba\._compat",
    )


def logprobs_extract(
    texts_by_doc_name: Dict[str, List[str]],
    *,
    top_k: int = 50,
) -> Dict[str, List[str]]:
    if not texts_by_doc_name:
        return {}

    backend_name, backend_model_path, extract_fn = _get_or_create_logprob_backend()

    output_dir = Path("tmp") / "logprobs_extract"
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_map: Dict[str, List[str]] = {}
    for doc_name, texts in texts_by_doc_name.items():
        merged_text = "".join(part for part in texts if part)

        # TODO(step-1): 文档预处理：将输入转成更自然语言、上下文连贯的文档格式。
        normalized_text = merged_text

        if not normalized_text.strip():
            keyword_map[doc_name] = []
            continue

        (
            tokens,
            token_logprobs,
            sampled_probs,
            _,
            _,
            jieba_words,
            jieba_probs,
            token_to_jieba_idx,
            softmax_denominators,
        ) = extract_fn(normalized_text)
        if not token_logprobs:
            keyword_map[doc_name] = []
            continue

        invalid_logprob_count = 0
        sanitized_logprobs: List[float] = []
        for item in token_logprobs:
            value = float(item)
            if not math.isfinite(value):
                value = LOGPROB_FLOOR
                invalid_logprob_count += 1
            sanitized_logprobs.append(value)

        if invalid_logprob_count > 0:
            logger.warning(
                "文档 %s 出现 %d 个非有限 logprob，已按 floor=%s 清洗。",
                doc_name,
                invalid_logprob_count,
                LOGPROB_FLOOR,
            )

        probs = [max(math.exp(lp), 1e-12) for lp in sanitized_logprobs]
        minus_log2_probs = [-math.log2(prob) for prob in probs]
        sampled_probs = [
            max(float(value), 1e-12)
            for value in sampled_probs
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        sampled_minus_log2_probs = [-math.log2(prob) for prob in sampled_probs]

        jieba_count = min(len(jieba_words), len(jieba_probs))
        jieba_words = jieba_words[:jieba_count]
        jieba_probs = jieba_probs[:jieba_count]
        if token_to_jieba_idx:
            token_to_jieba_idx = [
                int(idx) if isinstance(idx, (int, float)) and 0 <= int(idx) < jieba_count else -1
                for idx in token_to_jieba_idx[: len(tokens)]
            ]
            if len(token_to_jieba_idx) < len(tokens):
                token_to_jieba_idx.extend([-1] * (len(tokens) - len(token_to_jieba_idx)))
        else:
            token_to_jieba_idx = [-1] * len(tokens)
        jieba_minus_log2_probs = [
            -math.log2(max(prob, 1e-300))
            for prob in jieba_probs
        ]
        softmax_denominators = [
            float(item)
            for item in softmax_denominators[: len(tokens)]
            if isinstance(item, (int, float)) and math.isfinite(float(item)) and float(item) > 0.0
        ]
        softmax_denominator_log2 = [math.log2(max(item, 1e-300)) for item in softmax_denominators]

        a_prob_upper, b_prob_lower = _estimate_ab_thresholds_from_probs(probs)
        ab_star_phrases: List[str] = []

        jieba_top_terms = _rank_unique_jieba_by_sum_minus_log2(
            jieba_words=jieba_words,
            jieba_probs=jieba_probs,
            top_k=top_k,
        )

        safe_name = _sanitize_filename(doc_name)
        payload_path = output_dir / f"{safe_name}.json"
        with payload_path.open("w", encoding="utf-8") as fout:
            json.dump(
                {
                    "doc_name": doc_name,
                    "backend": backend_name,
                    "model_path": str(backend_model_path),
                    "token_count": len(tokens),
                    "tokens": tokens,
                    "logprobs": sanitized_logprobs,
                    "invalid_logprob_count": invalid_logprob_count,
                    "log2_logprobs": [math.log2(prob) for prob in probs],
                    "minus_log2_logprobs": minus_log2_probs,
                    "sampled_probs": sampled_probs,
                    "sampled_minus_log2_probs": sampled_minus_log2_probs,
                    "jieba_words": jieba_words,
                    "jieba_probs": jieba_probs,
                    "jieba_minus_log2_probs": jieba_minus_log2_probs,
                    "token_to_jieba_idx": token_to_jieba_idx,
                    "softmax_denominators": softmax_denominators,
                    "softmax_denominator_log2": softmax_denominator_log2,
                    "a_prob_upper": a_prob_upper,
                    "b_prob_lower": b_prob_lower,
                    "ab_star_phrases": ab_star_phrases,
                    "ab_star_disabled": True,
                    "top_k": int(top_k),
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        keyword_map[doc_name] = jieba_top_terms

    return keyword_map


def _estimate_ab_thresholds_from_probs(probs: List[float]) -> Tuple[float, float]:
    valid = np.asarray(
        [float(item) for item in probs if isinstance(item, (int, float)) and math.isfinite(float(item))],
        dtype=np.float64,
    )
    if valid.size < 20:
        return 0.2, 0.6

    valid = np.clip(valid, 1e-12, 1.0)
    bins = min(512, max(128, int(math.sqrt(valid.size) * 3)))
    hist, edges = np.histogram(valid, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0

    smooth_window = max(5, (bins // 64) * 2 + 1)
    kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
    smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")

    if smooth.size < 3:
        return 0.2, 0.6

    left_region = smooth[: max(3, smooth.size // 2)]
    right_region = smooth[max(0, smooth.size // 2) :]
    left_peak_idx = int(np.argmax(left_region))
    right_peak_idx = int(np.argmax(right_region) + max(0, smooth.size // 2))

    if right_peak_idx <= left_peak_idx:
        q20, q80 = np.quantile(valid, [0.2, 0.8]).tolist()
        return float(q20), float(q80)

    left_min_idx = right_peak_idx
    for idx in range(left_peak_idx + 1, right_peak_idx):
        prev_v = smooth[idx - 1]
        cur_v = smooth[idx]
        next_v = smooth[idx + 1]
        if cur_v <= prev_v and cur_v <= next_v:
            left_min_idx = idx
            break

    right_min_idx = left_peak_idx
    for idx in range(right_peak_idx - 1, left_peak_idx, -1):
        prev_v = smooth[idx - 1]
        cur_v = smooth[idx]
        next_v = smooth[idx + 1]
        if cur_v <= prev_v and cur_v <= next_v:
            right_min_idx = idx
            break

    a_prob_upper = float(np.clip(centers[left_min_idx], 1e-12, 1.0))
    b_prob_lower = float(np.clip(centers[right_min_idx], 1e-12, 1.0))
    if b_prob_lower < a_prob_upper:
        mid = (a_prob_upper + b_prob_lower) / 2.0
        a_prob_upper = min(a_prob_upper, mid)
        b_prob_lower = max(b_prob_lower, mid)
    return a_prob_upper, b_prob_lower


def _extract_ab_star_phrases(
    *,
    tokens: List[str],
    probs: List[float],
    a_prob_upper: float,
    b_prob_lower: float,
    top_k: int,
) -> List[str]:
    if not tokens or not probs:
        return []

    n = min(len(tokens), len(probs))
    limit = max(1, min(int(top_k), 100))
    candidates: List[Tuple[float, int, str]] = []

    for idx in range(0, n):
        a_prob = float(probs[idx])
        if not math.isfinite(a_prob):
            continue
        if a_prob > a_prob_upper:
            continue

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
            parts.append(token_tail)
            if re.search(r"[。！？.!?]", token_tail):
                break
            tail += 1

        phrase = re.sub(r"\s+", " ", "".join(parts).strip())
        if not phrase:
            fallback_token = str(tokens[idx]).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").strip()
            phrase = fallback_token if fallback_token else f"<WS@{idx}>"
        candidates.append((a_prob, idx, phrase))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [phrase for _, _, phrase in candidates[:limit]]


def _rank_unique_jieba_by_sum_minus_log2(
    *,
    jieba_words: List[str],
    jieba_probs: List[float],
    top_k: int,
) -> List[str]:
    if not jieba_words or not jieba_probs:
        return []

    limit = max(1, min(int(top_k), 100))
    stop_words = _load_jieba_open_source_stop_words()
    token_scores: Dict[str, float] = {}
    first_positions: Dict[str, int] = {}

    for idx, (word, prob) in enumerate(zip(jieba_words, jieba_probs)):
        token = str(word).strip()
        if not token:
            continue
        if _is_noise_token(token):
            continue
        if token.lower() in stop_words:
            continue
        if not isinstance(prob, (int, float)):
            continue
        prob_value = float(prob)
        if not math.isfinite(prob_value):
            continue

        score = -math.log2(max(prob_value, 1e-300))
        token_scores[token] = float(token_scores.get(token, 0.0) + score)
        if token not in first_positions:
            first_positions[token] = idx

    ranked = sorted(
        token_scores.items(),
        key=lambda item: (-item[1], first_positions.get(item[0], 10**9), item[0]),
    )
    return [token for token, _ in ranked[:limit]]


def _load_jieba_open_source_stop_words() -> FrozenSet[str]:
    global _JIEBA_STOP_WORDS_CACHE

    if _JIEBA_STOP_WORDS_CACHE is not None:
        return _JIEBA_STOP_WORDS_CACHE

    stop_words: set[str] = set()
    stop_words_path = Path(__file__).resolve().parent / "resources" / "jieba_stop_words_merged.txt"

    if stop_words_path.is_file():
        try:
            with stop_words_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    token = line.strip()
                    if token:
                        stop_words.add(token.lower())
        except Exception as exc:
            logger.warning("加载 jieba 停用词表失败（%s）：%s", stop_words_path, exc)
    else:
        logger.warning("未找到 jieba 停用词表：%s", stop_words_path)

    if not stop_words:
        # 回退到 jieba 内置默认停用词，避免过滤集为空。
        stop_words.update(
            {
                "the",
                "of",
                "is",
                "and",
                "to",
                "in",
                "that",
                "we",
                "for",
                "an",
                "are",
                "by",
                "be",
                "as",
                "on",
                "with",
                "can",
                "if",
                "from",
                "which",
                "you",
                "it",
                "this",
                "then",
                "at",
                "have",
                "all",
                "not",
                "one",
                "has",
                "or",
            }
        )

    _JIEBA_STOP_WORDS_CACHE = frozenset(stop_words)
    return _JIEBA_STOP_WORDS_CACHE


def _is_noise_token(token: str) -> bool:
    value = str(token).strip()
    if not value:
        return True

    if any(ch.isdigit() for ch in value):
        return True

    # 过滤纯标点/符号/空白的 token，例如：，。、“”…—
    if all(
        unicodedata.category(ch).startswith(("P", "S", "Z"))
        for ch in value
    ):
        return True

    return False


def _aggregate_jieba_word_probs(
    *,
    text: str,
    token_offsets: List[Tuple[int, int]],
    token_texts: List[str],
    token_probs: List[float],
) -> Tuple[List[str], List[float], List[int]]:
    _suppress_jieba_warnings()
    try:
        import jieba
        jieba.setLogLevel(logging.ERROR)
    except Exception as exc:
        raise ImportError(
            "缺少 jieba，无法执行 jieba 分词聚合。请先安装：pip install jieba"
        ) from exc

    if not text:
        return [], [], []
    n = min(len(token_offsets), len(token_texts), len(token_probs))
    if n <= 0:
        return [], [], []

    token_offsets = [
        (int(item[0]), int(item[1]))
        for item in token_offsets[:n]
        if isinstance(item, (tuple, list)) and len(item) >= 2
    ]
    token_texts = [str(item) for item in token_texts[:n]]
    token_probs = [
        max(float(item), 1e-300)
        for item in token_probs[:n]
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]
    n = min(len(token_offsets), len(token_texts), len(token_probs))
    token_offsets = token_offsets[:n]
    token_texts = token_texts[:n]
    token_probs = token_probs[:n]
    if n <= 0:
        return [], [], []

    segments: List[Tuple[str, bool, int, int]] = []
    cursor_text = 0
    source_text = str(text)
    for chunk in re.findall(r"\s+|\S+", source_text):
        if not chunk:
            continue
        start = source_text.find(chunk, cursor_text)
        if start < 0:
            start = cursor_text
        end = start + len(chunk)
        cursor_text = end
        if chunk.isspace():
            segments.append((str(chunk), False, start, end))
            continue

        local_cursor = 0
        for word in jieba.cut(chunk):
            if word:
                word_text = str(word)
                local_start = chunk.find(word_text, local_cursor)
                if local_start < 0:
                    local_start = local_cursor
                local_end = local_start + len(word_text)
                local_cursor = local_end
                segments.append((word_text, True, start + local_start, start + local_end))

    if not segments:
        return [], [], []

    word_probs: List[float] = []
    words: List[str] = []
    token_to_jieba_idx = [-1] * n
    token_cursor = 0

    for segment_text, is_jieba_word, seg_start, seg_end in segments:
        if token_cursor >= n:
            if is_jieba_word:
                words.append(segment_text)
                word_probs.append(1e-300)
            continue

        while token_cursor < n and token_offsets[token_cursor][1] <= seg_start:
            token_cursor += 1

        probe = token_cursor
        matched_token_indices: List[int] = []
        while probe < n and token_offsets[probe][0] < seg_end:
            tok_start, tok_end = token_offsets[probe]
            if tok_end > seg_start and tok_start < seg_end:
                matched_token_indices.append(probe)
            probe += 1

        consumed = [max(float(token_probs[idx]), 1e-300) for idx in matched_token_indices]
        if is_jieba_word:
            jieba_idx = len(words)
            words.append(segment_text)
            if consumed:
                log_sum = sum(math.log(max(prob, 1e-300)) for prob in consumed)
                word_probs.append(max(math.exp(log_sum), 1e-300))
            else:
                word_probs.append(1e-300)
            for tok_idx in matched_token_indices:
                token_to_jieba_idx[tok_idx] = jieba_idx

        while token_cursor < n and token_offsets[token_cursor][1] <= seg_end:
            token_cursor += 1

    if words:
        last_valid = -1
        for idx in range(n):
            mapped = token_to_jieba_idx[idx]
            if 0 <= mapped < len(words):
                last_valid = mapped
            elif last_valid >= 0:
                token_to_jieba_idx[idx] = last_valid

        next_valid = -1
        for idx in range(n - 1, -1, -1):
            mapped = token_to_jieba_idx[idx]
            if 0 <= mapped < len(words):
                next_valid = mapped
            elif next_valid >= 0:
                token_to_jieba_idx[idx] = next_valid

        for idx in range(n):
            if token_to_jieba_idx[idx] < 0:
                token_to_jieba_idx[idx] = 0

        def _normalize_text(value: str) -> str:
            text = str(value)
            text = text.replace("▁", "").replace("Ġ", "")
            text = re.sub(r"^##", "", text)
            text = re.sub(r"\s+", "", text)
            return text

        jieba_norm = [_normalize_text(item) for item in words]
        norm_to_indices: Dict[str, List[int]] = {}
        for idx, norm in enumerate(jieba_norm):
            if not norm:
                continue
            norm_to_indices.setdefault(norm, []).append(idx)

        anchor = 0
        for tok_idx, token_text in enumerate(token_texts[:n]):
            token_norm = _normalize_text(token_text)
            if not token_norm:
                continue
            if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token_norm):
                continue

            mapped = token_to_jieba_idx[tok_idx]
            if 0 <= mapped < len(words):
                mapped_norm = jieba_norm[mapped]
                if (token_norm in mapped_norm) or (mapped_norm and mapped_norm in token_norm):
                    anchor = mapped
                    continue

            candidates = norm_to_indices.get(token_norm, [])
            if candidates:
                best = min(candidates, key=lambda item: abs(item - anchor))
                token_to_jieba_idx[tok_idx] = best
                anchor = best
                continue

            best = -1
            best_dist = 10**9
            left = max(0, anchor - 128)
            right = min(len(words), anchor + 129)
            for jieba_idx in range(left, right):
                norm = jieba_norm[jieba_idx]
                if not norm:
                    continue
                if (token_norm in norm) or (norm in token_norm):
                    dist = abs(jieba_idx - anchor)
                    if dist < best_dist:
                        best_dist = dist
                        best = jieba_idx
            if best >= 0:
                token_to_jieba_idx[tok_idx] = best
                anchor = best

    return words, word_probs, token_to_jieba_idx


def _get_or_create_logprob_backend(
) -> Tuple[
    str,
    Path,
    Callable[[str], Tuple[List[str], List[float], List[float], int, float, List[str], List[float], List[int], List[float]]],
]:
    global _LOGPROB_BACKEND_CACHE
    if _LOGPROB_BACKEND_CACHE is not None:
        return _LOGPROB_BACKEND_CACHE

    hf_model_path = _resolve_bitnet_hf_path()
    _LOGPROB_BACKEND_CACHE = _build_logprob_backend(hf_model_path=hf_model_path)
    return _LOGPROB_BACKEND_CACHE


def _resolve_bitnet_hf_path() -> Optional[Path]:
    env_path = str(os.getenv("BITNET_HF_PATH") or "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate

    model_dir = Path("models") / "bitnet-b1.58-2B-4T"
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        return model_dir
    return None


def _build_logprob_backend(
    *,
    hf_model_path: Optional[Path],
) -> Tuple[
    str,
    Path,
    Callable[[str], Tuple[List[str], List[float], List[float], int, float, List[str], List[float], List[int], List[float]]],
]:
    if hf_model_path is not None:
        extractor = _build_transformers_runtime(hf_model_path)
        return "transformers", hf_model_path, extractor

    raise FileNotFoundError(
        "未找到可用 bitNet HF 模型目录：models/bitnet-b1.58-2B-4T（或设置 BITNET_HF_PATH）。"
    )


def _build_transformers_runtime(
    model_path: Path,
) -> Callable[[str], Tuple[List[str], List[float], List[float], int, float, List[str], List[float], List[int], List[float]]]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise ImportError(
            "缺少 transformers/torch，无法使用 HF bitNet 后端。请安装：pip install transformers torch"
        ) from exc

    from transformers import AutoTokenizer, AutoModelForCausalLM, TopKLogitsWarper, TopPLogitsWarper
    generation_config = None

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            fix_mistral_regex=True,
            **tokenizer_kwargs,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)

    try:
        if not bool(torch.cuda.is_available()):
            raise RuntimeError(
                "BitNet 低比特后端当前要求 CUDA 设备。检测到无可用 GPU，已中止加载（不做回退）。"
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            device_map={"": "cuda:0"},
            local_files_only=True,
        )
        generation_config = getattr(model, "generation_config", None)
    except OSError as exc:
        message = str(exc)
        if "configuration_bitnet.py" in message or "modeling_bitnet.py" in message:
            raise RuntimeError(
                "检测到本地 bitNet HF 模型缺少自定义代码文件(configuration_bitnet.py/modeling_bitnet.py)。"
                "\n请执行以下任一方案："
                "\n1) 从 HuggingFace 模型仓库补全这两个 .py 文件到模型目录；"
                "\n2) 克隆并安装 microsoft/BitNet，再将对应配置与建模文件放到模型目录；"
                "\n3) 重新运行本脚本，确认使用的是 models/bitnet-b1.58-2B-4T。"
            ) from exc
        raise

    device = "cuda:0"
    model.eval()

    model_device = _detect_model_device(model)
    if model_device != "cuda":
        raise RuntimeError(
            "检测到可用 CUDA 设备，但 BitNet 模型仍在 CPU 上。"
            "请检查 device_map、accelerate 与 PyTorch CUDA 环境。"
        )

    if not _is_bitnet_lowbit_model(model):
        raise RuntimeError(
            "模型未进入 BitNet 低比特线性层路径（未检测到 AutoBitLinear/BitLinear），已中止（不做回退）。"
        )

    quant_cfg = getattr(getattr(model, "config", None), "quantization_config", None)
    quant_method = str(getattr(quant_cfg, "quant_method", "")).lower() if quant_cfg is not None else ""
    if quant_method != "quantizationmethod.bitnet":
        print(f"警告：检测到模型量化方法为 '{quant_method}'，而非预期的 'quantizationmethod.bitnet'。请确认模型是否正确量化并兼容当前提取逻辑。")
        raise RuntimeError("当前加载模型未声明 bitnet 原生量化配置，已中止（不做回退）。")

    model_max_len = int(getattr(tokenizer, "model_max_length", 2048) or 2048)
    if model_max_len <= 0 or model_max_len > 8192:
        model_max_len = 4096
    if model_max_len < 16:
        model_max_len = 512

    top_p = float(getattr(generation_config, "top_p", 1.0) or 1.0) if generation_config is not None else 1.0
    top_k = int(getattr(generation_config, "top_k", 0) or 0) if generation_config is not None else 0
    special_token_ids = {int(item) for item in list(getattr(tokenizer, "all_special_ids", []) or [])}

    def _compute_token_offsets_for_text(text: str, target_token_ids: List[int]) -> List[Tuple[int, int]]:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            ref_ids = [int(item) for item in list(encoded.get("input_ids") or [])]
            ref_offsets = [
                (int(item[0]), int(item[1]))
                for item in list(encoded.get("offset_mapping") or [])
                if isinstance(item, (tuple, list)) and len(item) >= 2
            ]
            if len(ref_ids) != len(ref_offsets):
                return [(0, 0)] * len(target_token_ids)

            result: List[Tuple[int, int]] = []
            pointer = 0
            for tid in target_token_ids:
                found = -1
                upper = min(len(ref_ids), pointer + 256)
                for probe in range(pointer, upper):
                    if ref_ids[probe] == int(tid):
                        found = probe
                        break
                if found < 0:
                    for probe in range(0, len(ref_ids)):
                        if ref_ids[probe] == int(tid):
                            found = probe
                            break
                if found < 0:
                    result.append((0, 0))
                else:
                    result.append(ref_offsets[found])
                    pointer = found + 1
            return result
        except Exception:
            return [(0, 0)] * len(target_token_ids)

    def _extract(text: str) -> Tuple[List[str], List[float], List[float], int, float, List[str], List[float], List[int], List[float]]:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < 2:
            return [], [], [], top_k, top_p, [], [], [], []

        all_tokens: List[str] = []
        all_logprobs: List[float] = []
        all_sampled_probs: List[float] = []
        all_token_ids: List[int] = []
        all_softmax_denominators: List[float] = []
        cursor = 0
        step = max(2, model_max_len - 1)

        while cursor < len(token_ids) - 1:
            chunk_ids = token_ids[cursor : cursor + model_max_len]
            if len(chunk_ids) < 2:
                break

            input_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_tensor)
                logits = outputs.logits[:, :-1, :]
                target_tensor = input_tensor[:, 1:]
                warpers = []
                if top_k > 0:
                    warpers.append(TopKLogitsWarper(top_k=top_k))
                if 0.0 < top_p < 1.0:
                    warpers.append(TopPLogitsWarper(top_p=top_p))

                seq_len = int(target_tensor.shape[1])
                chunk_logprobs: List[float] = []
                sampled_chunk: List[float] = []
                chunk_softmax_denominators: List[float] = []

                for pos in range(seq_len):
                    scores = logits[:, pos, :]
                    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                    target_pos = target_tensor[:, pos]

                    chosen_logit = scores.gather(-1, target_pos.unsqueeze(-1)).squeeze(-1)
                    norm_pos = torch.logsumexp(scores, dim=-1)
                    softmax_denominator_pos = torch.exp(norm_pos)
                    logprob_pos = chosen_logit - norm_pos
                    logprob_pos = torch.nan_to_num(logprob_pos, nan=LOGPROB_FLOOR, posinf=0.0, neginf=LOGPROB_FLOOR)
                    logprob_value = float(logprob_pos[0].detach().cpu().item())
                    chunk_logprobs.append(logprob_value)
                    chunk_softmax_denominators.append(float(softmax_denominator_pos[0].detach().cpu().item()))

                    if warpers:
                        warped_scores = scores
                        for warper in warpers:
                            warped_scores = warper(input_tensor[:, : pos + 1], warped_scores)
                        probs_after_warp = torch.softmax(warped_scores, dim=-1)
                        chosen = probs_after_warp.gather(-1, target_pos.unsqueeze(-1)).squeeze(-1)
                        sampled_chunk.append(float(chosen[0].detach().cpu().item()))
                    else:
                        sampled_chunk.append(max(math.exp(logprob_value), 1e-12))

            del outputs, logits, target_tensor
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            chunk_target_ids = input_tensor[0, 1:].detach().cpu().tolist()
            chunk_tokens = [
                tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
                for tid in chunk_target_ids
            ]

            for tid, tok, lp, sp, denom in zip(chunk_target_ids, chunk_tokens, chunk_logprobs, sampled_chunk, chunk_softmax_denominators):
                tid_int = int(tid)
                if tid_int in special_token_ids:
                    continue
                all_tokens.append(str(tok))
                all_token_ids.append(tid_int)
                all_logprobs.append(float(lp))
                all_sampled_probs.append(float(sp))
                all_softmax_denominators.append(max(float(denom), 1e-300))
            del input_tensor
            cursor += step

        probs_from_logprobs = [max(math.exp(item), 1e-300) for item in all_logprobs]
        token_offsets = _compute_token_offsets_for_text(text, all_token_ids)
        jieba_words, jieba_probs, token_to_jieba_idx = _aggregate_jieba_word_probs(
            text=text,
            token_offsets=token_offsets,
            token_texts=all_tokens,
            token_probs=probs_from_logprobs,
        )

        return (
            all_tokens,
            all_logprobs,
            all_sampled_probs,
            top_k,
            top_p,
            jieba_words,
            jieba_probs,
            token_to_jieba_idx,
            all_softmax_denominators,
        )

    return _extract


def _is_bitnet_lowbit_model(model: Any) -> bool:
    try:
        for module in model.modules():
            cls_name = str(module.__class__.__name__)
            if cls_name in {"AutoBitLinear", "BitLinear"}:
                return True
    except Exception:
        return False
    return False


def _detect_model_device(model: Any) -> str:
    try:
        first_param = next(model.parameters())
        device_obj = getattr(first_param, "device", None)
        if device_obj is None:
            return "cpu"
        if hasattr(device_obj, "type"):
            return str(device_obj.type)
        return str(device_obj)
    except Exception:
        return "cpu"

def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", str(name).strip())
    return cleaned or "document"

