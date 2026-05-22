from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional
import importlib
import re


CJK_CHAR_CLASS = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
CJK_RE = re.compile(f"[{CJK_CHAR_CLASS}]")
CJK_SEQUENCE_RE = re.compile(f"[{CJK_CHAR_CLASS}]+")
TOKEN_RE = re.compile(
    f"[{CJK_CHAR_CLASS}]"
    r"|[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*"
    r"|\d+(?:\.\d+)?"
    r"|[^\s]"
)
SEARCH_TOKEN_RE = re.compile(
    f"[{CJK_CHAR_CLASS}]"
    r"|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"
)
EN_NUM_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
SENTENCE_BOUNDARIES = set("\u3002\uff01\uff1f\uff1b.!?;\n")
TRAILING_PUNCTUATION = set("\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001,.!?;:%)]}")


@dataclass(frozen=True)
class SplitPiece:
    text: str
    split_reason: str


def estimated_tokens(text: str) -> int:
    """Approximate model-token length for mixed Chinese/English text."""

    return len(TOKEN_RE.findall(text or ""))


def tokenize_for_search(
    text: str,
    tokenizer: str = "jieba_search",
    domain_terms_path: str = "",
    enable_cjk_bigrams: bool = True,
) -> List[str]:
    """Tokenize mixed Chinese/English text for BM25.

    English and numeric spans are preserved as lower-case terms. Chinese text is
    segmented with jieba search mode when available, augmented with optional
    domain terms and CJK bigrams. If jieba is unavailable, the function falls
    back to character-level CJK tokens so BM25 remains usable.
    """

    text = normalize_space(text)
    if not text:
        return []

    normalized_tokenizer = (tokenizer or "jieba_search").strip().lower()
    tokens: List[str] = []

    tokens.extend(token.lower() for token in EN_NUM_TOKEN_RE.findall(text))
    tokens.extend(domain_term_tokens(text, domain_terms_path))

    if normalized_tokenizer in {"jieba", "jieba_search"}:
        jieba_module = load_jieba()
        if jieba_module is not None:
            tokens.extend(normalize_search_token(token) for token in jieba_module.cut_for_search(text))
        else:
            tokens.extend(CJK_RE.findall(text))
    elif normalized_tokenizer in {"char", "character", "cjk_char"}:
        tokens.extend(CJK_RE.findall(text))
    else:
        tokens.extend(token.lower() for token in SEARCH_TOKEN_RE.findall(text))

    if enable_cjk_bigrams:
        tokens.extend(cjk_bigrams(text))

    return [token for token in tokens if token]


def load_jieba():
    try:
        return importlib.import_module("jieba")
    except ImportError:
        return None


def normalize_search_token(token: str) -> str:
    token = (token or "").strip().lower()
    if not token:
        return ""
    if contains_cjk(token):
        return token
    match = EN_NUM_TOKEN_RE.fullmatch(token)
    return match.group(0).lower() if match else ""


def domain_term_tokens(text: str, domain_terms_path: str) -> List[str]:
    terms = load_domain_terms(domain_terms_path)
    if not terms:
        return []
    tokens: List[str] = []
    lowered_text = text.lower()
    for term in terms:
        if contains_cjk(term):
            count = text.count(term)
            tokens.extend([term] * count)
        else:
            normalized = term.lower()
            count = lowered_text.count(normalized)
            tokens.extend([normalized] * count)
    return tokens


@lru_cache(maxsize=16)
def load_domain_terms(domain_terms_path: str) -> tuple[str, ...]:
    if not domain_terms_path:
        return ()
    path = Path(domain_terms_path)
    if not path.exists():
        return ()
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term.lower() if not contains_cjk(term) else term)
    return tuple(dict.fromkeys(terms))


def cjk_bigrams(text: str) -> List[str]:
    bigrams: List[str] = []
    for sequence in CJK_SEQUENCE_RE.findall(text or ""):
        if len(sequence) < 2:
            continue
        bigrams.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return bigrams


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def cjk_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    return len(CJK_RE.findall(compact)) / len(compact)


def split_by_estimated_tokens(text: str, max_tokens: int, overlap: int = 0) -> List[str]:
    return [piece.text for piece in split_by_estimated_tokens_with_metadata(text, max_tokens, overlap)]


def split_by_estimated_tokens_with_metadata(text: str, max_tokens: int, overlap: int = 0) -> List[SplitPiece]:
    text = normalize_space(text)
    if not text:
        return []
    if estimated_tokens(text) <= max_tokens:
        return [SplitPiece(text, "within_limit")]

    units = boundary_units(text, max_tokens)
    chunks: List[SplitPiece] = []
    current = ""
    current_reasons: List[str] = []

    for unit in units:
        candidate = join_text(current, unit.text) if current else unit.text
        if current and estimated_tokens(candidate) > max_tokens:
            chunks.append(SplitPiece(current.strip(), combine_split_reasons(current_reasons)))
            overlap_text = trailing_overlap_text(current, overlap) if overlap > 0 else ""
            current = join_text(overlap_text, unit.text) if overlap_text else unit.text
            current_reasons = ["overlap", unit.split_reason] if overlap_text else [unit.split_reason]
            if estimated_tokens(current) > max_tokens and estimated_tokens(unit.text) <= max_tokens:
                current = unit.text
                current_reasons = [unit.split_reason]
        else:
            current = candidate
            current_reasons.append(unit.split_reason)

    if current.strip():
        chunks.append(SplitPiece(current.strip(), combine_split_reasons(current_reasons)))
    return [chunk for chunk in chunks if chunk.text.strip()]


def boundary_units(text: str, max_tokens: int) -> List[SplitPiece]:
    units: List[SplitPiece] = []
    paragraphs = paragraph_units(text)
    for paragraph in paragraphs:
        if estimated_tokens(paragraph) <= max_tokens:
            units.append(SplitPiece(paragraph, "paragraph_boundary"))
            continue
        for sentence in semantic_units(paragraph):
            if estimated_tokens(sentence) <= max_tokens:
                units.append(SplitPiece(sentence, "sentence_boundary"))
            else:
                units.extend(SplitPiece(piece, "hard_limit") for piece in split_long_text(sentence, max_tokens))
    return units


def paragraph_units(text: str) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text or "") if paragraph.strip()]
    return paragraphs or [text.strip()]


def semantic_units(text: str) -> List[str]:
    units: List[str] = []
    buffer: List[str] = []
    for char in text:
        buffer.append(char)
        if char in SENTENCE_BOUNDARIES:
            unit = "".join(buffer).strip()
            if unit:
                units.append(unit)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        units.append(tail)
    return units or [text.strip()]


def expand_long_units(units: Iterable[str], max_tokens: int) -> List[str]:
    expanded: List[str] = []
    for unit in units:
        if estimated_tokens(unit) <= max_tokens:
            expanded.append(unit)
        else:
            expanded.extend(split_long_text(unit, max_tokens))
    return expanded


def split_long_text(text: str, max_tokens: int) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return []
    chunks = []
    for index in range(0, len(tokens), max_tokens):
        chunks.append(join_token_like_text(tokens[index : index + max_tokens]))
    return chunks


def trailing_overlap_text(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    sentences = semantic_units(text)
    output = ""
    for sentence in reversed(sentences):
        candidate = join_text(sentence, output) if output else sentence
        if estimated_tokens(candidate) > token_budget:
            break
        output = candidate
    return output or trailing_token_text(text, token_budget)


def trailing_token_text(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return ""
    return join_token_like_text(tokens[-token_budget:])


def combine_split_reasons(reasons: List[str]) -> str:
    filtered = [reason for reason in reasons if reason != "overlap"]
    if not filtered:
        return "within_limit"
    if "hard_limit" in filtered:
        return "hard_limit"
    if "sentence_boundary" in filtered:
        return "sentence_boundary"
    if "paragraph_boundary" in filtered:
        return "paragraph_boundary"
    return filtered[-1]


def join_token_like_text(tokens: List[str]) -> str:
    output = ""
    for token in tokens:
        if not output:
            output = token
            continue
        if is_cjk_token(token) or is_cjk_token(output[-1]) or token in TRAILING_PUNCTUATION:
            output += token
        elif output[-1] in "([{/":
            output += token
        else:
            output += " " + token
    return output.strip()


def join_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left[-1] in SENTENCE_BOUNDARIES or right[0] in TRAILING_PUNCTUATION:
        return left + right
    if contains_cjk(left[-1]) or contains_cjk(right[0]):
        return left + right
    return left + " " + right


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_cjk_token(text: str) -> bool:
    return len(text) == 1 and contains_cjk(text)
