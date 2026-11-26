import re
from typing import Any


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_ngrams(text: str, n: int = 13) -> set[str]:
    norm = _normalize_text(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def _char_trigram_jaccard(a: str, b: str) -> float:
    ga = _extract_ngrams(a, n=3)
    gb = _extract_ngrams(b, n=3)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def check_sample_contamination(
    eval_prompt: str,
    train_corpus: list[str],
    ngram_size: int = 13,
    semantic_threshold: float = 0.68,
) -> dict[str, Any]:
    """
    双层防污染检测：
    Layer 1: 13-gram 精确子串重叠检测（捕获直接背题/原题泄露）
    Layer 2: 3-gram Jaccard 近似改写检测（捕获同义词替换、语序微调的改写泄露）
    """
    if not train_corpus:
        return {"is_contaminated": False, "method": "none", "score": 0.0, "matched_snippet": None}

    eval_ngrams = _extract_ngrams(eval_prompt, n=ngram_size)
    best_sim = 0.0
    best_snippet = None

    for train_text in train_corpus:
        # Layer 1: 13-gram 重叠
        train_ngrams = _extract_ngrams(train_text, n=ngram_size)
        if eval_ngrams and (eval_ngrams & train_ngrams):
            return {
                "is_contaminated": True,
                "method": f"{ngram_size}-gram_exact_overlap",
                "score": 1.0,
                "matched_snippet": train_text[:180],
            }

        # Layer 2: 改写/扰动相似度检测
        sim = _char_trigram_jaccard(eval_prompt, train_text)
        if sim > best_sim:
            best_sim = sim
            best_snippet = train_text[:180]

    if best_sim >= semantic_threshold:
        return {
            "is_contaminated": True,
            "method": "paraphrase_similarity_leak",
            "score": round(best_sim, 4),
            "matched_snippet": best_snippet,
        }

    return {
        "is_contaminated": False,
        "method": "clean",
        "score": round(best_sim, 4),
        "matched_snippet": None,
    }