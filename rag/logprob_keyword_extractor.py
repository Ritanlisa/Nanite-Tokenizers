from __future__ import annotations

import json
import importlib
import logging
import math
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from rag.line_profiler_instrument import profile_if_enabled

logger = logging.getLogger(__name__)
LOGPROB_FLOOR = math.log(1e-12)
TRANSITION_MAX_NEG_LOG2_PROB = 0.4
TRANSITION_MAX_PHRASE_LEN = 24
CONNECT_INCLUDE_CHARS = {"·", "・", "･", "-", "'", "’", "."}
CONNECT_EXCEPTION_CHARS = {" ", "\n", "\t", "\r", "<br>"}
BPE_MIN_NEW_TERM_FREQ = 2


@dataclass(frozen=True)
class DictionarySegmentationContext:
    text: str
    token_offsets: List[Tuple[int, int]]
    token_texts: List[str]
    token_probs: List[float]
    top_next_tokens: List[str]
    top_next_probs: List[float]


DictionarySegmenterFn = Callable[
    [DictionarySegmentationContext],
    Tuple[List[str], List[float], List[int]],
]
DEFAULT_DICTIONARY_SEGMENTER = "token-bpe-positive-pmi"
_DICTIONARY_SEGMENTERS: Dict[str, DictionarySegmenterFn] = {}
_ACTIVE_DICTIONARY_SEGMENTER = DEFAULT_DICTIONARY_SEGMENTER
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
_STOP_WORDS_CACHE: Optional[Set[str]] = None


def register_dictionary_segmenter(name: str, fn: DictionarySegmenterFn) -> None:
    segmenter_name = str(name or "").strip().lower()
    if not segmenter_name:
        raise ValueError("分词策略名称不能为空")
    _DICTIONARY_SEGMENTERS[segmenter_name] = fn


def set_active_dictionary_segmenter(name: str) -> None:
    global _ACTIVE_DICTIONARY_SEGMENTER
    segmenter_name = str(name or "").strip().lower()
    if segmenter_name not in _DICTIONARY_SEGMENTERS:
        available = ", ".join(sorted(_DICTIONARY_SEGMENTERS.keys())) or "<none>"
        raise ValueError(f"未知分词策略: {segmenter_name}，可用策略: {available}")
    _ACTIVE_DICTIONARY_SEGMENTER = segmenter_name


def get_active_dictionary_segmenter() -> str:
    return _ACTIVE_DICTIONARY_SEGMENTER


def _segment_dictionary_tokens(
    *,
    text: str,
    token_offsets: List[Tuple[int, int]],
    token_texts: List[str],
    token_probs: List[float],
    top_next_tokens: List[str],
    top_next_probs: List[float],
) -> Tuple[List[str], List[float], List[int]]:
    # 根据当前激活策略做词典分词，未注册时自动回退到默认实现。
    method = get_active_dictionary_segmenter()
    fn = _DICTIONARY_SEGMENTERS.get(method)
    if fn is None:
        fallback = _DICTIONARY_SEGMENTERS.get(DEFAULT_DICTIONARY_SEGMENTER)
        if fallback is None:
            raise RuntimeError("无可用分词策略，请先注册至少一个策略")
        logger.warning("分词策略 %s 未注册，回退到 %s", method, DEFAULT_DICTIONARY_SEGMENTER)
        fn = fallback
    context = _normalize_segmentation_context(
        DictionarySegmentationContext(
            text=str(text or ""),
            token_offsets=token_offsets,
            token_texts=token_texts,
            token_probs=token_probs,
            top_next_tokens=top_next_tokens,
            top_next_probs=top_next_probs,
        )
    )
    return fn(context)


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


def _is_pure_numeric_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    normalized = re.sub(r"[,._\-+:/\\，。．、：；]+", "", text)
    if not normalized:
        return False
    return bool(re.fullmatch(r"\d+", normalized))


def _is_noise_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return True
    if text in CONNECT_INCLUDE_CHARS:
        return False
    if _is_pure_numeric_token(text):
        return True
    if len(text) <= 1 and not ("\u4e00" <= text <= "\u9fff") and text not in CONNECT_INCLUDE_CHARS:
        return True
    # 仅由标点/连接符组成的项视为噪声。
    non_space = [ch for ch in text if not ch.isspace()]
    if non_space and all(unicodedata.category(ch).startswith("P") and ch not in CONNECT_INCLUDE_CHARS for ch in non_space):
        return True
    return False


def _load_jieba_open_source_stop_words() -> Set[str]:
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is not None:
        return set(_STOP_WORDS_CACHE)

    builtins = {
        "的", "了", "和", "是", "在", "及", "与", "或", "为", "对", "将", "可", "而", "就",
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    }
    path = str(os.getenv("KEYWORD_STOPWORDS_PATH") or "").strip()
    loaded: Set[str] = set()
    if path:
        candidate = Path(path)
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    word = str(line).strip().lower()
                    if word and not word.startswith("#"):
                        loaded.add(word)
            except Exception:
                pass

    _STOP_WORDS_CACHE = {*(word.lower() for word in builtins), *loaded}
    return set(_STOP_WORDS_CACHE)


def _is_stopword_filter_enabled() -> bool:
    raw = str(os.getenv("KEYWORD_ENABLE_STOPWORDS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _utf8_char_length(text: str) -> int:
    # Python 字符串按 Unicode 码点计数，和 UTF-8 文本字符数语义一致。
    return len(str(text or ""))


def _normalize_document_text_for_keyword_extraction(text: str) -> str:
    # todo: 清洗文本
    return text


def _filter_keyword_terms(terms: List[str], *, top_k: int, minlength: int) -> List[str]:
    if not terms:
        return []
    min_chars = max(1, int(minlength))
    stop_words = _load_jieba_open_source_stop_words()
    result: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        token = str(term or "").strip()
        if not token:
            continue
        if _utf8_char_length(token) < min_chars:
            continue
        lowered = token.lower()
        if _is_noise_token(token) or lowered in stop_words:
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max(1, int(top_k)):
            break
    return result


def logprobs_extract(
    texts_by_doc_name: Dict[str, List[str]],
    *,
    top_k: int = 12,
    minlength: int = 2,
) -> Dict[str, List[str]]:
    if not texts_by_doc_name:
        return {}

    backend_name, backend_model_path, extract_fn = _get_or_create_logprob_backend()

    output_dir = Path("tmp") / "logprobs_extract"
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_map: Dict[str, List[str]] = {}
    for doc_name, texts in texts_by_doc_name.items():
        merged_text = "".join(part for part in texts if part)

        # 文档预处理：修复词内换行，降低同一词被拆成多个词条的概率。
        normalized_text = _normalize_document_text_for_keyword_extraction(merged_text)

        if not normalized_text.strip():
            keyword_map[doc_name] = []
            continue

        (
            tokens,
            token_logprobs,
            sampled_probs,
            _,
            _,
            dictionary_words,
            dictionary_probs,
            token_to_dict_idx,
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

        dictionary_count = min(len(dictionary_words), len(dictionary_probs))
        dictionary_words = dictionary_words[:dictionary_count]
        dictionary_probs = dictionary_probs[:dictionary_count]
        if token_to_dict_idx:
            token_to_dict_idx = [
                int(idx) if isinstance(idx, (int, float)) and 0 <= int(idx) < dictionary_count else -1
                for idx in token_to_dict_idx[: len(tokens)]
            ]
            if len(token_to_dict_idx) < len(tokens):
                token_to_dict_idx.extend([-1] * (len(tokens) - len(token_to_dict_idx)))
        else:
            token_to_dict_idx = [-1] * len(tokens)
        dictionary_minus_log2_probs = [
            -math.log2(max(prob, 1e-300))
            for prob in dictionary_probs
        ]
        softmax_denominators = [
            float(item)
            for item in softmax_denominators[: len(tokens)]
            if isinstance(item, (int, float)) and math.isfinite(float(item)) and float(item) > 0.0
        ]
        softmax_denominator_log2 = [math.log2(max(item, 1e-300)) for item in softmax_denominators]

        a_prob_upper, b_prob_lower = _estimate_ab_thresholds_from_probs(probs)
        ab_star_phrases: List[str] = []

        rank_limit = max(int(top_k), min(300, int(top_k) * 6))
        dictionary_top_terms = _rank_unique_terms_by_sum_minus_log2(
            words=dictionary_words,
            probs=dictionary_probs,
            top_k=rank_limit,
        )
        min_chars = max(1, int(minlength))
        if _is_stopword_filter_enabled():
            dictionary_top_terms = _filter_keyword_terms(dictionary_top_terms, top_k=top_k, minlength=min_chars)
        else:
            filtered_terms: List[str] = []
            seen_terms: Set[str] = set()
            for term in dictionary_top_terms:
                token = str(term or "").strip()
                if not token:
                    continue
                if _utf8_char_length(token) < min_chars:
                    continue
                if token in seen_terms:
                    continue
                seen_terms.add(token)
                filtered_terms.append(token)
                if len(filtered_terms) >= max(1, int(top_k)):
                    break
            dictionary_top_terms = filtered_terms

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
                    "dictionary_words": dictionary_words,
                    "dictionary_probs": dictionary_probs,
                    "dictionary_minus_log2_probs": dictionary_minus_log2_probs,
                    "token_to_dict_idx": token_to_dict_idx,
                    "dictionary_method": get_active_dictionary_segmenter(),
                    "transition_max_neg_log2_prob": TRANSITION_MAX_NEG_LOG2_PROB,
                    "jieba_words": dictionary_words,
                    "jieba_probs": dictionary_probs,
                    "jieba_minus_log2_probs": dictionary_minus_log2_probs,
                    "token_to_jieba_idx": token_to_dict_idx,
                    "softmax_denominators": softmax_denominators,
                    "softmax_denominator_log2": softmax_denominator_log2,
                    "a_prob_upper": a_prob_upper,
                    "b_prob_lower": b_prob_lower,
                    "ab_star_phrases": ab_star_phrases,
                    "ab_star_disabled": True,
                    "top_k": int(top_k),
                    "minlength": int(min_chars),
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        keyword_map[doc_name] = dictionary_top_terms

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


def _rank_unique_terms_by_sum_minus_log2(
    *,
    words: List[str],
    probs: List[float],
    top_k: int,
) -> List[str]:
    if not words or not probs:
        return []

    limit = max(1, min(int(top_k), 100))
    token_scores: Dict[str, float] = {}
    first_positions: Dict[str, int] = {}

    normalized_words = [str(item).strip() for item in words]
    higher_terms = {item for item in normalized_words if len(item) >= 2}
    covered_subword_positions = [False] * len(normalized_words)
    if higher_terms:
        max_window = 24
        for start in range(len(normalized_words)):
            first = normalized_words[start]
            if not first:
                continue
            merged = first
            upper = min(len(normalized_words), start + max_window)
            for end in range(start + 1, upper):
                part = normalized_words[end]
                if not part:
                    break
                merged = f"{merged}{part}"
                if merged in higher_terms and len(merged) > len(first):
                    for idx in range(start, end + 1):
                        covered_subword_positions[idx] = True

    for idx, (word, prob) in enumerate(zip(words, probs)):
        token = str(word).strip()
        if not token:
            continue
        if not isinstance(prob, (int, float)):
            continue
        prob_value = float(prob)
        if not math.isfinite(prob_value):
            continue

        if idx < len(covered_subword_positions) and covered_subword_positions[idx]:
            continue

        # 与 debug 前端默认口径保持一致：按平方惊喜度（surprise^2）累计排序。
        surprise = -math.log2(max(prob_value, 1e-300))
        score = surprise * surprise
        token_scores[token] = float(token_scores.get(token, 0.0) + score)
        if token not in first_positions:
            first_positions[token] = idx

    ranked = sorted(
        token_scores.items(),
        key=lambda item: (-item[1], first_positions.get(item[0], 10**9), item[0]),
    )
    return [token for token, _ in ranked[:limit]]


def _normalize_segmentation_context(
    context: DictionarySegmentationContext,
) -> DictionarySegmentationContext:
    # 统一清洗输入上下文，避免非法概率值和越界偏移影响后续分词。
    n = min(len(context.token_texts), len(context.token_probs))
    if n <= 0:
        return DictionarySegmentationContext(
            text=str(context.text or ""),
            token_offsets=[],
            token_texts=[],
            token_probs=[],
            top_next_tokens=[],
            top_next_probs=[],
        )

    token_texts = [str(item) for item in context.token_texts[:n]]
    token_probs = [
        max(float(item), 1e-300)
        for item in context.token_probs[:n]
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]
    n = min(len(token_texts), len(token_probs))

    token_offsets: List[Tuple[int, int]] = []
    for item in context.token_offsets[:n]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            token_offsets.append((int(item[0]), int(item[1])))
        else:
            token_offsets.append((0, 0))
    if len(token_offsets) < n:
        token_offsets.extend([(0, 0)] * (n - len(token_offsets)))

    normalized_top_next_tokens = ["" for _ in range(n)]
    normalized_top_next_probs = [0.0 for _ in range(n)]
    for idx in range(min(n, len(context.top_next_tokens), len(context.top_next_probs))):
        normalized_top_next_tokens[idx] = str(context.top_next_tokens[idx])
        raw_prob = context.top_next_probs[idx]
        if isinstance(raw_prob, (int, float)) and math.isfinite(float(raw_prob)):
            normalized_top_next_probs[idx] = max(float(raw_prob), 0.0)

    return DictionarySegmentationContext(
        text=str(context.text or ""),
        token_offsets=token_offsets[:n],
        token_texts=token_texts[:n],
        token_probs=token_probs[:n],
        top_next_tokens=normalized_top_next_tokens[:n],
        top_next_probs=normalized_top_next_probs[:n],
    )


def _is_break_char(char: str) -> bool:
    if not char:
        return True
    if char in CONNECT_INCLUDE_CHARS:
        return False
    # 放过普通空格（含 Unicode Space_Separator），但换行/制表等控制空白仍作为断点。
    if char in {"\n", "\r", "\t", "\v", "\f"}:
        return True
    category = unicodedata.category(char)
    if category in {"Zl", "Zp"}:
        return True
    return category.startswith("P")


def _contains_break_char(token: str) -> bool:
    text = str(token or "")
    if not text:
        return True
    for char in text:
        if _is_break_char(char):
            return True
    return False


@dataclass
class _PhraseUnit:
    text: str
    start_idx: int
    end_idx: int
    token_indices: Tuple[int, ...]
    token_count: int
    reciprocal_sum: float


def _merge_phrase_units(
    *,
    left: _PhraseUnit,
    right: _PhraseUnit,
) -> _PhraseUnit:
    return _PhraseUnit(
        text=f"{left.text}{right.text}",
        start_idx=left.start_idx,
        end_idx=right.end_idx,
        token_indices=tuple([*left.token_indices, *right.token_indices]),
        token_count=int(left.token_count + right.token_count),
        reciprocal_sum=float(left.reciprocal_sum + right.reciprocal_sum),
    )


def _is_exception_token_for_bpe(token: str) -> bool:
    return _is_token_composed_by_markers(token, CONNECT_EXCEPTION_CHARS)


def _is_include_token_for_bpe(token: str) -> bool:
    return _is_token_composed_by_markers(token, CONNECT_INCLUDE_CHARS)


def _is_token_composed_by_markers(token: str, markers: Set[str]) -> bool:
    text = str(token or "")
    marker_list = [str(item) for item in markers if str(item)]
    if not text or not marker_list:
        return False

    # DP 判断 token 是否可完全由 marker 拼接而成，支持多字符 marker。
    n = len(text)
    reachable = [False] * (n + 1)
    reachable[0] = True
    for idx in range(n):
        if not reachable[idx]:
            continue
        for marker in marker_list:
            if text.startswith(marker, idx):
                reachable[idx + len(marker)] = True
    return bool(reachable[n])


def _strip_known_connect_markers_for_bpe(token: str) -> str:
    text = str(token or "")
    if not text:
        return ""

    markers = sorted(
        {*(str(item) for item in CONNECT_INCLUDE_CHARS if str(item)), *(str(item) for item in CONNECT_EXCEPTION_CHARS if str(item))},
        key=len,
        reverse=True,
    )
    if not markers:
        return text

    parts: List[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        matched = False
        for marker in markers:
            if text.startswith(marker, idx):
                idx += len(marker)
                matched = True
                break
        if matched:
            continue
        parts.append(text[idx])
        idx += 1
    return "".join(parts)


def _has_disallowed_punctuation_for_bpe(token: str) -> bool:
    text = _strip_known_connect_markers_for_bpe(token)
    if not text:
        return False
    for char in text:
        if unicodedata.category(char).startswith("P"):
            return True
    return False


def _classify_token_for_bpe(token: str) -> str:
    # BPE 仅允许 CONNECT_INCLUDE_CHARS 参与组合；CONNECT_EXCEPTION_CHARS 视为空字符桥接。
    text = str(token or "")
    if not text:
        return "break"
    if _is_exception_token_for_bpe(text):
        return "exception"
    if _is_include_token_for_bpe(text):
        return "content"
    if _has_disallowed_punctuation_for_bpe(text):
        return "break"
    if _contains_break_char(text):
        return "break"
    return "content"

@profile_if_enabled
def _collect_positive_pmi_pair_keys(
    segments: List[List[_PhraseUnit]],
    token_probs: List[float],
    *,
    min_pair_count: int,
    console_debug: bool = False,
) -> Dict[Tuple[str, str], Tuple[float, float, int, int, str, str]]:
    _ = token_probs
    prob_floor = 1e-300
    token_to_id: Dict[str, int] = {}
    id_to_token: List[str] = []
    unigram_counts_by_id: List[int] = []
    bigram_counts_by_id: Dict[Tuple[int, int], int] = {}
    bigram_prob_mass_by_id: Dict[Tuple[int, int], float] = {}
    total_tokens = 0
    total_bigrams = 0
    min_pair_count_int = int(min_pair_count)
    bpe_min_new_term_freq_int = int(BPE_MIN_NEW_TERM_FREQ)

    token_id_get = token_to_id.get
    bigram_get = bigram_counts_by_id.get
    mass_get = bigram_prob_mass_by_id.get
    unigram_counts_append = unigram_counts_by_id.append
    id_to_token_append = id_to_token.append

    for segment in segments:
        if not segment:
            continue
        segment_len = len(segment)
        total_tokens += segment_len

        # 单次遍历同时完成 unigram 与 bigram 统计，使用整数 ID 降低字符串哈希开销。
        prev_token_id = -1
        prev_prob = prob_floor
        for unit in segment:
            token_text = unit.text
            token_id = token_id_get(token_text, -1)
            if token_id < 0:
                token_id = len(id_to_token)
                token_to_id[token_text] = token_id
                id_to_token_append(token_text)
                unigram_counts_append(0)
            unigram_counts_by_id[token_id] = unigram_counts_by_id[token_id] + 1

            token_count = unit.token_count
            reciprocal_sum = unit.reciprocal_sum
            if token_count <= 0 or reciprocal_sum <= 0.0:
                current_prob = prob_floor
            else:
                current_prob = token_count / reciprocal_sum
                if current_prob < prob_floor:
                    current_prob = prob_floor

            if prev_token_id >= 0:
                pair_ids = (prev_token_id, token_id)
                bigram_counts_by_id[pair_ids] = bigram_get(pair_ids, 0) + 1
                pair_confidence = math.sqrt(prev_prob * current_prob)
                bigram_prob_mass_by_id[pair_ids] = mass_get(pair_ids, 0.0) + pair_confidence
                total_bigrams += 1

            prev_token_id = token_id
            prev_prob = current_prob

    if total_tokens <= 0 or total_bigrams <= 0:
        return {}

    inv_total_tokens = 1.0 / float(total_tokens)
    inv_total_bigrams = 1.0 / float(total_bigrams)

    # 先收集候选，再用 NumPy 向量化批量计算 PMI 与 merge_score。
    candidate_left_ids: List[int] = []
    candidate_right_ids: List[int] = []
    candidate_pair_counts: List[int] = []
    candidate_left_counts: List[int] = []
    candidate_right_counts: List[int] = []
    candidate_avg_probs: List[float] = []
    candidate_pair_lens: List[int] = []

    mass_get = bigram_prob_mass_by_id.get
    for pair_ids, pair_count in bigram_counts_by_id.items():
        # 硬性要求：BPE 新合并词在原文中的可观测出现次数至少为 2。
        # 对于相邻 pair 合并，新词出现次数等价于该 pair 的相邻出现次数。
        if pair_count < bpe_min_new_term_freq_int:
            continue
        if pair_count < min_pair_count_int:
            continue
        left_id, right_id = pair_ids
        if left_id < 0 or right_id < 0:
            continue
        if left_id >= len(unigram_counts_by_id) or right_id >= len(unigram_counts_by_id):
            continue
        left_count = unigram_counts_by_id[left_id]
        right_count = unigram_counts_by_id[right_id]
        if pair_count <= 0 or left_count <= 0 or right_count <= 0:
            continue

        left_text = id_to_token[left_id]
        right_text = id_to_token[right_id]

        candidate_left_ids.append(left_id)
        candidate_right_ids.append(right_id)
        candidate_pair_counts.append(pair_count)
        candidate_left_counts.append(left_count)
        candidate_right_counts.append(right_count)
        candidate_avg_probs.append(mass_get(pair_ids, 0.0) / pair_count)
        candidate_pair_lens.append(len(left_text) + len(right_text))

    if not candidate_pair_counts:
        return {}

    pair_counts_arr = np.asarray(candidate_pair_counts, dtype=np.float64)
    left_counts_arr = np.asarray(candidate_left_counts, dtype=np.float64)
    right_counts_arr = np.asarray(candidate_right_counts, dtype=np.float64)
    avg_probs_arr = np.asarray(candidate_avg_probs, dtype=np.float64)

    p_ab = pair_counts_arr * inv_total_bigrams
    p_a = left_counts_arr * inv_total_tokens
    p_b = right_counts_arr * inv_total_tokens

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi_arr = np.log(p_ab / (p_a * p_b))
        freq_factor_arr = np.log1p(pair_counts_arr)
        prob_factor_arr = 1.0 / (1.0 + np.maximum(-np.log2(np.maximum(avg_probs_arr, 1e-300)), 0.0))
        merge_score_arr = pmi_arr * freq_factor_arr * prob_factor_arr

    valid_mask = np.isfinite(pmi_arr) & (pmi_arr > 0.0) & np.isfinite(merge_score_arr)
    if not bool(np.any(valid_mask)):
        return {}

    candidate_pair_counts_arr = np.asarray(candidate_pair_counts, dtype=np.int64)
    candidate_pair_lens_arr = np.asarray(candidate_pair_lens, dtype=np.int64)
    candidate_left_ids_arr = np.asarray(candidate_left_ids, dtype=np.int64)
    candidate_right_ids_arr = np.asarray(candidate_right_ids, dtype=np.int64)
    id_to_token_arr = np.asarray(id_to_token, dtype=np.str_)
    candidate_left_text_arr = id_to_token_arr[candidate_left_ids_arr]
    candidate_right_text_arr = id_to_token_arr[candidate_right_ids_arr]

    selected_indices = np.flatnonzero(valid_mask)
    if selected_indices.size <= 0:
        return {}

    pair_keys: Dict[Tuple[str, str], Tuple[float, float, int, int, str, str]] = {}
    for idx in selected_indices.tolist():
        left = str(candidate_left_text_arr[idx])
        right = str(candidate_right_text_arr[idx])
        pair_keys[(left, right)] = (
            float(merge_score_arr[idx]),
            float(pmi_arr[idx]),
            int(candidate_pair_counts_arr[idx]),
            int(candidate_pair_lens_arr[idx]),
            left,
            right,
        )

    if console_debug and pair_keys:
        best_pair, best_key = max(pair_keys.items(), key=lambda item: item[1])
        print(
            f"Current Batch Best Pair: {best_pair}, Score Detail: {best_key}, BatchSize={len(pair_keys)}\n"
        )
    return pair_keys


def _apply_batch_pair_merge_once(
    segments: List[List[_PhraseUnit]],
    pair_keys: Dict[Tuple[str, str], Tuple[float, float, int, int, str, str]],
) -> int:
    merged_total = 0
    if not pair_keys:
        return merged_total

    for seg_idx, segment in enumerate(segments):
        segment_len = len(segment)
        if segment_len < 2:
            continue

        # 快速跳过：该 segment 不含可合并 pair 时无需重建列表。
        first_match = -1
        for probe in range(0, segment_len - 1):
            pair_probe = (segment[probe].text, segment[probe + 1].text)
            if pair_probe in pair_keys:
                first_match = probe
                break
        if first_match < 0:
            continue

        new_segment: List[_PhraseUnit] = segment[:first_match]
        append_unit = new_segment.append
        cursor = first_match
        while cursor < segment_len:
            if cursor + 1 >= segment_len:
                append_unit(segment[cursor])
                cursor += 1
                continue

            pair_here = (segment[cursor].text, segment[cursor + 1].text)
            key_here = pair_keys.get(pair_here)
            if key_here is None:
                append_unit(segment[cursor])
                cursor += 1
                continue

            # 局部冲突消解：若与右侧重叠 pair 冲突，优先保留评分更高者。
            if cursor + 2 < segment_len:
                pair_next = (segment[cursor + 1].text, segment[cursor + 2].text)
                key_next = pair_keys.get(pair_next)
                if key_next is not None and key_next > key_here:
                    append_unit(segment[cursor])
                    cursor += 1
                    continue

            if cursor + 1 < segment_len:
                append_unit(
                    _merge_phrase_units(left=segment[cursor], right=segment[cursor + 1])
                )
                merged_total += 1
                cursor += 2
                continue

            append_unit(segment[cursor])
            cursor += 1

        segments[seg_idx] = new_segment

    return merged_total


@profile_if_enabled
def _segment_bpe_positive_pmi(context: DictionarySegmentationContext) -> Tuple[List[str], List[float], List[int]]:
    # 在 BitNet token 序列上执行 BatchBPE：每轮收集所有正 PMI 候选并批量合并。
    token_texts = context.token_texts
    token_probs = context.token_probs
    n = min(len(token_texts), len(token_probs))
    if n <= 0:
        return [], [], []

    segments: List[List[_PhraseUnit]] = []
    current: List[_PhraseUnit] = []
    for idx in range(n):
        token = str(token_texts[idx])
        token_kind = _classify_token_for_bpe(token)
        if token_kind == "break":
            if current:
                segments.append(current)
                current = []
            continue
        if token_kind == "exception":
            # 例外字符视为空字符：不入词组，但允许跨越它统计/合并相邻内容 token。
            continue
        token_prob = max(float(token_probs[idx]), 1e-300)
        current.append(
            _PhraseUnit(
                text=token,
                start_idx=idx,
                end_idx=idx,
                token_indices=(idx,),
                token_count=1,
                reciprocal_sum=1.0 / token_prob,
            )
        )
    if current:
        segments.append(current)

    if not segments:
        return [], [], [-1] * n

    # 持续批量合并，直到不存在可用正 PMI pair。
    while True:
        pair_keys = _collect_positive_pmi_pair_keys(
            segments,
            token_probs,
            min_pair_count=10,
            console_debug=True,
        )
        if not pair_keys:
            break
        merged = _apply_batch_pair_merge_once(segments, pair_keys)
        if merged <= 0:
            break

    phrase_words: List[str] = []
    phrase_probs: List[float] = []
    token_to_phrase_idx = [-1] * n

    for segment in segments:
        for unit in segment:
            phrase_text = str(unit.text).strip()
            if not phrase_text:
                continue
            if unit.start_idx < 0 or unit.end_idx >= n or unit.start_idx > unit.end_idx:
                continue

            if not unit.token_indices:
                continue
            log_sum = 0.0
            for token_idx in unit.token_indices:
                if 0 <= int(token_idx) < n:
                    log_sum += math.log(max(float(token_probs[token_idx]), 1e-300))
            phrase_prob = max(math.exp(log_sum), 1e-300)

            phrase_idx = len(phrase_words)
            phrase_words.append(phrase_text)
            phrase_probs.append(float(phrase_prob))
            for token_idx in unit.token_indices:
                if 0 <= int(token_idx) < n:
                    token_to_phrase_idx[int(token_idx)] = phrase_idx

    return phrase_words, phrase_probs, token_to_phrase_idx


def _build_transition_phrase_patterns(
    context: DictionarySegmentationContext,
) -> List[List[str]]:
    # 用“当前 token A -> 所有后继 AB 的最后出现位置”构造候选短语模板。
    # 是否继续扩展由该 AB 最后出现位置上 B 的信息量阈值决定。
    token_texts = context.token_texts
    token_probs = context.token_probs
    n = min(len(token_texts), len(token_probs))
    if n <= 0:
        return []

    # successor_last_index_by_token[A][B] = AB 最后一次相邻出现时 B 的索引。
    successor_last_index_by_token: Dict[str, Dict[str, int]] = {}
    for idx in range(1, n):
        left = str(token_texts[idx - 1])
        right = str(token_texts[idx])
        successor_last_index_by_token.setdefault(left, {})[right] = idx

    unique_tokens: List[str] = []
    seen_tokens: Set[str] = set()
    for token in token_texts:
        if token not in seen_tokens:
            seen_tokens.add(token)
            unique_tokens.append(token)

    phrase_patterns: List[List[str]] = []
    for token in unique_tokens:
        if _contains_break_char(token):
            continue

        phrase_tokens = [str(token)]
        visited_pairs: Set[Tuple[str, str]] = set()

        while len(phrase_tokens) < TRANSITION_MAX_PHRASE_LEN:
            anchor_token = str(phrase_tokens[-1])
            successors = successor_last_index_by_token.get(anchor_token, {})
            if not successors:
                break

            selected_token: Optional[str] = None
            selected_second_index = -1
            selected_info = float("inf")

            for candidate_token, second_index in successors.items():
                if second_index <= 0 or second_index >= n:
                    continue
                if _contains_break_char(candidate_token):
                    continue

                pair = (anchor_token, str(candidate_token))
                if pair in visited_pairs:
                    continue

                candidate_prob = float(token_probs[second_index])
                if (not math.isfinite(candidate_prob)) or candidate_prob <= 0.0:
                    continue

                info_content = -math.log2(max(candidate_prob, 1e-300))
                if info_content >= float(TRANSITION_MAX_NEG_LOG2_PROB):
                    continue

                if (
                    info_content < selected_info
                    or (
                        math.isclose(info_content, selected_info)
                        and int(second_index) > int(selected_second_index)
                    )
                ):
                    selected_token = str(candidate_token)
                    selected_second_index = int(second_index)
                    selected_info = float(info_content)

            if not selected_token:
                break

            visited_pairs.add((anchor_token, selected_token))
            phrase_tokens.append(selected_token)

        phrase_patterns.append(phrase_tokens)

    phrase_pattern_set: Set[Tuple[str, ...]] = set()
    dedup_patterns: List[List[str]] = []
    for item in phrase_patterns + [[token] for token in unique_tokens if not _contains_break_char(token)]:
        key = tuple(item)
        if not key or key in phrase_pattern_set:
            continue
        phrase_pattern_set.add(key)
        dedup_patterns.append(list(item))
    return dedup_patterns


def _harmonic_mean(values: List[float], floor: float = 1e-300) -> float:
    valid = [max(float(value), floor) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not valid:
        return floor
    reciprocal_sum = sum(1.0 / value for value in valid)
    if reciprocal_sum <= 0.0:
        return floor
    return max(len(valid) / reciprocal_sum, floor)


@profile_if_enabled
def _materialize_phrase_patterns(
    context: DictionarySegmentationContext,
    phrase_patterns: List[List[str]],
    *,
    probability_mode: str,
) -> Tuple[List[str], List[float], List[int]]:
    # 将模板映射回 token 序列，生成短语、短语概率及 token 到短语索引映射。
    token_texts = context.token_texts
    token_probs = context.token_probs
    n = min(len(token_texts), len(token_probs))
    if n <= 0:
        return [], [], []

    patterns_by_first: Dict[str, List[List[str]]] = {}
    for pattern in phrase_patterns:
        first = str(pattern[0])
        patterns_by_first.setdefault(first, []).append(pattern)
    for first in patterns_by_first.keys():
        patterns_by_first[first].sort(key=lambda item: len(item), reverse=True)

    phrase_words: List[str] = []
    phrase_probs: List[float] = []
    token_to_phrase_idx = [-1] * n

    cursor = 0
    while cursor < n:
        current = str(token_texts[cursor])
        if _contains_break_char(current):
            cursor += 1
            continue

        candidates = patterns_by_first.get(current, [])
        matched: Optional[List[str]] = None

        for pattern in candidates:
            length = len(pattern)
            if length <= 0 or cursor + length > n:
                continue
            ok = True
            for offset, expected in enumerate(pattern):
                if str(token_texts[cursor + offset]) != str(expected):
                    ok = False
                    break
            if ok:
                matched = pattern
                break

        if matched is None:
            matched = [current]

        seg_len = len(matched)
        phrase_text = "".join(str(item) for item in matched).strip()
        if not phrase_text:
            phrase_text = str(current)

        segment_probs = [max(float(token_probs[idx]), 1e-300) for idx in range(cursor, min(cursor + seg_len, n))]
        if probability_mode == "harmonic":
            phrase_prob = _harmonic_mean(segment_probs)
        else:
            log_sum = sum(math.log(prob) for prob in segment_probs)
            phrase_prob = max(math.exp(log_sum), 1e-300)

        phrase_idx = len(phrase_words)
        phrase_words.append(phrase_text)
        phrase_probs.append(float(phrase_prob))
        for idx in range(cursor, min(cursor + seg_len, n)):
            token_to_phrase_idx[idx] = phrase_idx

        cursor += max(1, seg_len)

    if phrase_words:
        last_valid = -1
        for idx in range(n):
            if _contains_break_char(str(token_texts[idx])):
                token_to_phrase_idx[idx] = -1
                last_valid = -1
                continue
            mapped = token_to_phrase_idx[idx]
            if 0 <= mapped < len(phrase_words):
                last_valid = mapped
            elif last_valid >= 0:
                token_to_phrase_idx[idx] = last_valid

        next_valid = -1
        for idx in range(n - 1, -1, -1):
            if _contains_break_char(str(token_texts[idx])):
                token_to_phrase_idx[idx] = -1
                next_valid = -1
                continue
            mapped = token_to_phrase_idx[idx]
            if 0 <= mapped < len(phrase_words):
                next_valid = mapped
            elif next_valid >= 0:
                token_to_phrase_idx[idx] = next_valid

        for idx in range(n):
            if _contains_break_char(str(token_texts[idx])):
                token_to_phrase_idx[idx] = -1
                continue
            if token_to_phrase_idx[idx] < 0:
                token_to_phrase_idx[idx] = 0

    return phrase_words, phrase_probs, token_to_phrase_idx


@profile_if_enabled
def _segment_transition_chain(context: DictionarySegmentationContext) -> Tuple[List[str], List[float], List[int]]:
    # 默认策略：按转移链拼接短语，概率使用连乘。
    phrase_patterns = _build_transition_phrase_patterns(context)
    return _materialize_phrase_patterns(context, phrase_patterns, probability_mode="product")


@profile_if_enabled
def _segment_transition(context: DictionarySegmentationContext) -> Tuple[List[str], List[float], List[int]]:
    # 新策略：基于 AB 最后出现位置 + B 信息量阈值，按转移链拼接短语，概率使用连乘。
    phrase_patterns = _build_transition_phrase_patterns(context)
    return _materialize_phrase_patterns(context, phrase_patterns, probability_mode="product")


@profile_if_enabled
def _segment_transition_chain_hmean(context: DictionarySegmentationContext) -> Tuple[List[str], List[float], List[int]]:
    # 备选策略：与默认策略相同，但短语概率改为调和平均。
    phrase_patterns = _build_transition_phrase_patterns(context)
    return _materialize_phrase_patterns(context, phrase_patterns, probability_mode="harmonic")


@profile_if_enabled
def _segment_with_jieba(context: DictionarySegmentationContext) -> Tuple[List[str], List[float], List[int]]:
    # 兼容 jieba 分词：按字符区间对齐 token，并聚合为词级概率。
    _suppress_jieba_warnings()
    try:
        import jieba

        jieba.setLogLevel(logging.ERROR)
    except Exception as exc:
        raise ImportError(
            "缺少 jieba，无法执行 jieba 分词聚合。请先安装：pip install jieba"
        ) from exc

    text = str(context.text or "")
    token_offsets = context.token_offsets
    token_texts = context.token_texts
    token_probs = context.token_probs
    n = min(len(token_offsets), len(token_texts), len(token_probs))
    if not text or n <= 0:
        return [], [], []

    segments: List[Tuple[str, bool, int, int]] = []
    cursor_text = 0
    for chunk in re.findall(r"\s+|\S+", text):
        if not chunk:
            continue
        start = text.find(chunk, cursor_text)
        if start < 0:
            start = cursor_text
        end = start + len(chunk)
        cursor_text = end
        if chunk.isspace():
            segments.append((str(chunk), False, start, end))
            continue

        local_cursor = 0
        for word in jieba.cut(chunk):
            if not word:
                continue
            word_text = str(word)
            local_start = chunk.find(word_text, local_cursor)
            if local_start < 0:
                local_start = local_cursor
            local_end = local_start + len(word_text)
            local_cursor = local_end
            segments.append((word_text, True, start + local_start, start + local_end))

    word_probs: List[float] = []
    words: List[str] = []
    token_to_word_idx = [-1] * n
    token_cursor = 0

    for segment_text, is_word, seg_start, seg_end in segments:
        if token_cursor >= n:
            if is_word and not _contains_break_char(segment_text):
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

        if is_word and not _contains_break_char(segment_text):
            word_idx = len(words)
            words.append(segment_text)
            if matched_token_indices:
                log_sum = sum(math.log(max(float(token_probs[idx]), 1e-300)) for idx in matched_token_indices)
                word_probs.append(max(math.exp(log_sum), 1e-300))
            else:
                word_probs.append(1e-300)
            for tok_idx in matched_token_indices:
                token_to_word_idx[tok_idx] = word_idx

        while token_cursor < n and token_offsets[token_cursor][1] <= seg_end:
            token_cursor += 1

    if words:
        last_valid = -1
        for idx in range(n):
            mapped = token_to_word_idx[idx]
            if 0 <= mapped < len(words):
                last_valid = mapped
            elif last_valid >= 0:
                token_to_word_idx[idx] = last_valid

        next_valid = -1
        for idx in range(n - 1, -1, -1):
            mapped = token_to_word_idx[idx]
            if 0 <= mapped < len(words):
                next_valid = mapped
            elif next_valid >= 0:
                token_to_word_idx[idx] = next_valid

        for idx in range(n):
            if token_to_word_idx[idx] < 0:
                token_to_word_idx[idx] = 0

    return words, word_probs, token_to_word_idx


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

    def _decode_token_id_span(token_id_span: List[int]) -> str:
        if not token_id_span:
            return ""
        try:
            text = tokenizer.decode(
                [int(tid) for tid in token_id_span],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            return str(text)
        except Exception:
            return ""

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

    @profile_if_enabled
    def _extract(text: str) -> Tuple[List[str], List[float], List[float], int, float, List[str], List[float], List[int], List[float]]:
        # 核心提取流程：分块推理、计算 token 概率，再聚合到词典短语。
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < 2:
            return [], [], [], top_k, top_p, [], [], [], []

        all_tokens: List[str] = []
        all_logprobs: List[float] = []
        all_sampled_probs: List[float] = []
        all_token_ids: List[int] = []
        all_softmax_denominators: List[float] = []
        all_pred_top_tokens: List[str] = []
        all_pred_top_probs: List[float] = []
        cursor = 0
        step = max(2, model_max_len - 1)

        while cursor < len(token_ids) - 1:
            chunk_ids = token_ids[cursor : cursor + model_max_len]
            if len(chunk_ids) < 2:
                break

            input_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_tensor) # 33.4% Time
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
                chunk_pred_top_ids: List[int] = []
                chunk_pred_top_probs: List[float] = []

                for pos in range(seq_len):
                    scores = logits[:, pos, :]
                    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                    target_pos = target_tensor[:, pos]

                    chosen_logit = scores.gather(-1, target_pos.unsqueeze(-1)).squeeze(-1)
                    norm_pos = torch.logsumexp(scores, dim=-1)
                    softmax_denominator_pos = torch.exp(norm_pos)
                    top_logit, top_idx = torch.max(scores, dim=-1)
                    top_prob = torch.exp(top_logit - norm_pos)
                    logprob_pos = chosen_logit - norm_pos
                    logprob_pos = torch.nan_to_num(logprob_pos, nan=LOGPROB_FLOOR, posinf=0.0, neginf=LOGPROB_FLOOR)
                    logprob_value = float(logprob_pos[0].detach().cpu().item()) # 28.5% Time
                    chunk_logprobs.append(logprob_value)
                    chunk_softmax_denominators.append(float(softmax_denominator_pos[0].detach().cpu().item()))
                    chunk_pred_top_ids.append(int(top_idx[0].detach().cpu().item()))
                    chunk_pred_top_probs.append(float(top_prob[0].detach().cpu().item()))

                    if warpers:
                        warped_scores = scores
                        for warper in warpers:
                            warped_scores = warper(input_tensor[:, : pos + 1], warped_scores) # 13.4% Time
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
            chunk_pred_top_tokens = [
                tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
                for tid in chunk_pred_top_ids
            ]

            for tid, tok, lp, sp, denom, pred_tok, pred_prob in zip(
                chunk_target_ids,
                chunk_tokens,
                chunk_logprobs,
                sampled_chunk,
                chunk_softmax_denominators,
                chunk_pred_top_tokens,
                chunk_pred_top_probs,
            ):
                tid_int = int(tid)
                if tid_int in special_token_ids:
                    continue
                all_tokens.append(str(tok))
                all_token_ids.append(tid_int)
                all_logprobs.append(float(lp))
                all_sampled_probs.append(float(sp))
                all_softmax_denominators.append(max(float(denom), 1e-300))
                all_pred_top_tokens.append(str(pred_tok))
                all_pred_top_probs.append(max(float(pred_prob), 0.0))
            del input_tensor
            cursor += step

        probs_from_logprobs = [max(math.exp(item), 1e-300) for item in all_logprobs]
        token_offsets = _compute_token_offsets_for_text(text, all_token_ids)
        _ = token_offsets

        next_top_tokens = [""] * len(all_tokens)
        next_top_probs = [0.0] * len(all_tokens)
        for idx in range(0, len(all_tokens) - 1):
            next_top_tokens[idx] = str(all_pred_top_tokens[idx + 1])
            next_top_probs[idx] = max(float(all_pred_top_probs[idx + 1]), 0.0)

        phrase_words, phrase_probs, token_to_phrase_idx = _segment_dictionary_tokens(
            text=text,
            token_offsets=token_offsets,
            token_texts=all_tokens,
            token_probs=probs_from_logprobs,
            top_next_tokens=next_top_tokens,
            top_next_probs=next_top_probs,
        )

        # 避免 BPE 子词直接字符串拼接造成乱码：按每个词对应 token_id 序列统一解码。
        if phrase_words and token_to_phrase_idx and all_token_ids:
            phrase_token_ids: List[List[int]] = [[] for _ in range(len(phrase_words))]
            for tok_pos, phrase_idx in enumerate(token_to_phrase_idx[: len(all_token_ids)]):
                if not isinstance(phrase_idx, int):
                    continue
                if 0 <= phrase_idx < len(phrase_token_ids):
                    phrase_token_ids[phrase_idx].append(int(all_token_ids[tok_pos]))

            decoded_phrase_words: List[str] = []
            for idx, original in enumerate(phrase_words):
                merged = _decode_token_id_span(phrase_token_ids[idx]) if idx < len(phrase_token_ids) else ""
                merged = str(merged or "").strip()
                decoded_phrase_words.append(merged if merged else str(original))
            phrase_words = decoded_phrase_words

        return (
            all_tokens,
            all_logprobs,
            all_sampled_probs,
            top_k,
            top_p,
            phrase_words,
            phrase_probs,
            token_to_phrase_idx,
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


def _initialize_dictionary_segmenters() -> None:
    register_dictionary_segmenter("segment-transition", _segment_transition)
    register_dictionary_segmenter("token-transition-chain", _segment_transition_chain)
    register_dictionary_segmenter("token-transition-chain-hmean", _segment_transition_chain_hmean)
    register_dictionary_segmenter("token-bpe-positive-pmi", _segment_bpe_positive_pmi)
    register_dictionary_segmenter("jieba", _segment_with_jieba)
    env_method = str(os.getenv("KEYWORD_SEGMENTER_METHOD") or "").strip().lower()
    if env_method:
        try:
            set_active_dictionary_segmenter(env_method)
        except ValueError:
            logger.warning(
                "环境变量 KEYWORD_SEGMENTER_METHOD=%s 无效，使用默认策略 %s",
                env_method,
                DEFAULT_DICTIONARY_SEGMENTER,
            )


_initialize_dictionary_segmenters()

def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", str(name).strip())
    return cleaned or "document"

