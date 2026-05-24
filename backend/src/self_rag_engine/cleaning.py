from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple
import re
import unicodedata


REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references(?:\s+and\s+notes)?|bibliography|参考文献|参考资料|works\s+cited|literature\s+cited)\s*[:：]?\s*$",
    re.IGNORECASE,
)

REFERENCE_SECTION_RE = re.compile(
    r"(^|[>\n\r]\s*)(references(?:\s+and\s+notes)?|bibliography|参考文献|参考资料|works\s+cited|literature\s+cited)\s*[:：]?\s*$",
    re.IGNORECASE,
)

BIBLIOGRAPHY_MARKER_RE = re.compile(
    r"(\bdoi\s*:|\bDOI\s*:|\bet\s+al\.?|"
    r"\[[JMC]\]|,\s*\d{4}\s*,\s*\d+\s*\(\s*\d+\s*\)\s*:\s*\d+[-–]\d+|"
    r"\bVol\.\s*\d+|\bNo\.\s*\d+)",
    re.IGNORECASE,
)

PUBLICATION_HEADER_RE = re.compile(
    r"^\s*(vol\.\s*\d+.*no\.\s*\d+.*|.*第\s*\d+\s*卷.*第\s*\d+\s*期.*|"
    r".*\bissn\b.*|.*\bjournal\s+of\b.*\bvol\.\b.*)\s*$",
    re.IGNORECASE,
)

FRONT_MATTER_ONLY_RE = re.compile(
    r"(received\s+\d+|accepted\s+\d+|published\s+\d+|corresponding\s+author|"
    r"©|copyright|creative\s+commons|open\s+access|email\s*:|e-mail\s*:)",
    re.IGNORECASE,
)

DATE_ONLY_RE = re.compile(
    r"^\s*(\d{4}\s*年\s*\d{1,2}\s*月|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\s*$",
    re.IGNORECASE,
)

SUPPLEMENT_LICENSE_RE = re.compile(
    r"(supplement\s+(published|doi)|parent\s+article\s+doi|creative\s+commons\s+attribution|"
    r"format\s+provided\s+by\s+the\s+authors\s+and\s+unedited|further\s+distribution\s+of\s+this\s+work)",
    re.IGNORECASE,
)

NOISE_LINE_RE = re.compile(
    r"^\s*("
    r"\d+"
    r"|page\s+\d+(\s+of\s+\d+)?"
    r"|check\s*for\s*updates"
    r"|checkfor"
    r"|©\s*.+"
    r"|copyright\s+.+"
    r"|downloaded\s+from\s+.+"
    r"|https?://\S+"
    r"|www\.\S+"
    r"|doi:\s*\S+"
    r")\s*$",
    re.IGNORECASE,
)

PUBLICATION_TITLE_NOISE_RE = re.compile(
    r"^\s*(check\s*for\s*updates|checkfor|updates|letter|article|research\s+article|paper|optics\s+letters)\s*$",
    re.IGNORECASE,
)

PLATFORM_METADATA_RE = re.compile(
    r"(citation:|view\s+online:|view\s+table\s+of\s+contents:|published\s+by\s+the|"
    r"to\s+cite\s+this\s+article:|view\s+the\s+article\s+online\s+for\s+updates)",
    re.IGNORECASE,
)

RECOMMENDATION_HEADING_RE = re.compile(
    r"^\s*(articles\s+you\s+may\s+be\s+interested\s+in|you\s+may\s+also\s+like|related\s+content)\s*$",
    re.IGNORECASE,
)

ACADEMIC_HEADING_RE = re.compile(
    r"^\s*(((\d+|[ivxlcdm]+)\.?|[一二三四五六七八九十]+[、.]?)\s*)?"
    r"(abstract|introduction|background|related\s+work|methods?|materials\s+and\s+methods|"
    r"experiments?|experimental\s+setup|results?|discussion|conclusions?|acknowledg(e)?ments?|funding|"
    r"摘要|引言|前言|相关工作|材料与方法|方法|实验结果|实验|结果与讨论|结果|讨论|结论|致谢|基金|资助)"
    r"\s*$",
    re.IGNORECASE,
)


def clean_blocks(blocks: Iterable[Any], config: Any, source_type: str = "paper") -> Tuple[List[Any], Dict[str, Any]]:
    """Annotate common paper noise before parent/child chunking."""

    block_list = list(blocks)
    if not getattr(config, "enable_document_cleaning", True):
        return block_list, {"enabled": False, "input_blocks": len(block_list), "output_blocks": len(block_list)}
    return annotate_blocks(block_list, config, source_type)


def annotate_blocks(blocks: Iterable[Any], config: Any, source_type: str = "paper") -> Tuple[List[Any], Dict[str, Any]]:
    block_list = list(blocks)
    is_note = source_type in {"note", "text"}

    repeated_lines = (
        find_repeated_page_lines(block_list, getattr(config, "repeated_header_min_pages", 2))
        if getattr(config, "clean_repeated_headers", True)
        else set()
    )

    annotated: List[Any] = []
    noise_counts = defaultdict(int)
    role_counts = defaultdict(int)
    references_started = False
    reference_level = 0
    recommendation_page = None

    for block in block_list:
        # Code blocks are preserved verbatim regardless of source type
        # (no whitespace normalization, no hyphen joining, no noise regex).
        if getattr(block, "block_type", "") == "code":
            raw = getattr(block, "text", "") or ""
            setattr(block, "text", raw)
            setattr(block, "text_for_display", raw)
            setattr(block, "text_for_embedding", raw)
            setattr(block, "role", "body")
            setattr(block, "noise_reason", "")
            role_counts["body"] += 1
            annotated.append(block)
            continue
        # Notes / plain text get light cleaning only: whitespace normalization,
        # empty-block marking, and NO paper-layout noise rules or reference swallowing.
        if is_note:
            raw = re.sub(r"[ \t]+", " ", getattr(block, "text", "") or "").strip()
            note_role = "body" if raw else "noise"
            setattr(block, "text", raw)
            setattr(block, "text_for_display", raw)
            setattr(block, "text_for_embedding", raw)
            setattr(block, "role", note_role)
            setattr(block, "noise_reason", "" if raw else "empty")
            if note_role == "noise":
                noise_counts["empty"] += 1
            role_counts[note_role] += 1
            annotated.append(block)
            continue

        text = normalize_scientific_text(getattr(block, "text", "") or "")
        role = getattr(block, "role", "body") or "body"
        noise_reason = getattr(block, "noise_reason", "") or ""

        exclude_refs = getattr(config, "clean_exclude_references", True)
        is_heading = getattr(block, "block_type", "") in {"title", "section_header"}
        block_level = getattr(block, "level", 0) or 1
        # References section ends at the next heading whose level is no deeper than the
        # reference heading (e.g. an Appendix sibling). Assumes the reference heading's
        # level (from heading_level) is typically 2; if it were misdetected as 1 (title),
        # a level-2 appendix could still be swallowed — a known boundary.
        if references_started and exclude_refs and is_heading and not is_reference_heading(block) and block_level <= reference_level:
            references_started = False
            reference_level = 0

        if is_reference_heading(block) and exclude_refs:
            references_started = True
            reference_level = block_level
            role = "reference"
        elif references_started and exclude_refs:
            role = "reference"
        elif is_reference_section_path(getattr(block, "section_path", "")):
            role = "reference"
        elif not text.strip():
            role = "noise"
            noise_reason = "empty"
        if references_started and getattr(config, "clean_exclude_references", True):
            pass
        elif is_recommendation_heading(block):
            recommendation_page = getattr(block, "page_start", None)
            role = "noise"
            noise_reason = "recommendation_noise"
        elif recommendation_page is not None:
            page = getattr(block, "page_start", None)
            if page == recommendation_page and not is_academic_heading(text, block):
                role = "noise"
                noise_reason = "recommendation_noise"
            else:
                recommendation_page = None
        elif is_platform_metadata_block(block):
            role = "noise"
            noise_reason = "platform_metadata"
        elif is_publication_noise_block(block):
            role = "noise"
            noise_reason = "publication_noise"
        elif is_publication_header_block(block):
            role = "noise"
            noise_reason = "publication_header"
        elif is_bibliography_dominated_text(text):
            if references_started or is_reference_section_path(getattr(block, "section_path", "")):
                role = "reference"
                noise_reason = ""

        new_text, line_removed = clean_text_lines(text, repeated_lines)
        if line_removed:
            noise_counts["repeated_or_noise_lines"] += line_removed
        if not new_text.strip() and role not in {"asset", "reference", "furniture"}:
            role = "noise"
            noise_reason = noise_reason or "empty_after_line_clean"
        if role == "body":
            block_type = "section_header" if is_academic_heading(new_text, block) else getattr(block, "block_type", "")
            level = 2 if block_type == "section_header" and getattr(block, "level", 0) == 0 else getattr(block, "level", 0)
            setattr(block, "block_type", block_type)
            setattr(block, "level", level)
        setattr(block, "text", new_text.strip())
        setattr(block, "text_for_display", new_text.strip())
        setattr(block, "text_for_embedding", remove_citation_markers(new_text.strip()))
        setattr(block, "role", role)
        setattr(block, "noise_reason", noise_reason)
        if role == "reference" and not getattr(block, "section_path", ""):
            setattr(block, "section_path", "References")
        if role == "noise" and noise_reason:
            noise_counts[noise_reason] += 1
        role_counts[role] += 1
        annotated.append(block)

    report = {
        "enabled": True,
        "input_blocks": len(block_list),
        "output_blocks": len(annotated),
        "removed_blocks": 0,
        "removed_counts": dict(noise_counts),
        "role_counts": dict(role_counts),
        "noise_counts": dict(noise_counts),
        "repeated_lines": sorted(repeated_lines),
    }
    return annotated, report


def filter_blocks_for_main_chunks(blocks: Iterable[Any]) -> List[Any]:
    return [block for block in blocks if getattr(block, "role", "body") not in {"reference", "furniture", "noise"}]


def normalize_scientific_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_citation_markers(text: str) -> str:
    text = re.sub(r"\s*\[[0-9,\-–—\s]+\]", "", text or "")
    text = re.sub(r"\s*\([A-Z][A-Za-z]+ et al\.,?\s+\d{4}[a-z]?\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def find_repeated_page_lines(blocks: List[Any], min_pages: int) -> set[str]:
    line_pages: Dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        page = getattr(block, "page_start", None)
        if page is None:
            continue
        for line in split_candidate_lines(getattr(block, "text", "") or ""):
            normalized = normalize_line(line)
            if is_repeated_line_candidate(normalized):
                line_pages[normalized].add(int(page))
    return {line for line, pages in line_pages.items() if len(pages) >= min_pages}


def clean_text_lines(text: str, repeated_lines: set[str]) -> tuple[str, int]:
    output = []
    removed = 0
    for line in text.splitlines() or [text]:
        normalized = normalize_line(line)
        if not normalized:
            continue
        if normalized in repeated_lines or NOISE_LINE_RE.match(normalized):
            removed += 1
            continue
        output.append(line.strip())
    cleaned = "\n".join(output)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), removed


def is_reference_heading(block: Any) -> bool:
    text = normalize_line(getattr(block, "text", "") or "")
    block_type = getattr(block, "block_type", "")
    if block_type in {"title", "section_header"} and REFERENCE_HEADING_RE.match(text):
        return True
    return bool(REFERENCE_HEADING_RE.match(text))


def is_reference_section_path(section_path: str) -> bool:
    text = normalize_line(section_path or "")
    if not text:
        return False
    for part in re.split(r">\s*", text):
        if REFERENCE_HEADING_RE.match(part.strip()):
            return True
    return bool(REFERENCE_SECTION_RE.search(text) or REFERENCE_HEADING_RE.match(text))


def is_publication_noise_block(block: Any) -> bool:
    text = normalize_line(getattr(block, "text", "") or "")
    if PUBLICATION_TITLE_NOISE_RE.match(text):
        return True
    return False


def is_publication_header_block(block: Any) -> bool:
    text = normalize_line(getattr(block, "text", "") or "")
    return is_publication_header_text(text)


def is_publication_header_text(text: str) -> bool:
    normalized = normalize_line(text)
    if not normalized:
        return False
    if PUBLICATION_HEADER_RE.match(normalized):
        return True
    if len(normalized) <= 140 and "第" in normalized and "卷" in normalized and "期" in normalized:
        return True
    return False


def is_bibliography_dominated_text(text: str) -> bool:
    normalized = normalize_line(text)
    if not normalized:
        return False
    if len(normalized) < 80:
        return False
    markers = BIBLIOGRAPHY_MARKER_RE.findall(normalized)
    if len(markers) >= 4:
        return True
    lines = split_candidate_lines(text)
    if len(lines) >= 3:
        citation_lines = sum(1 for line in lines if BIBLIOGRAPHY_MARKER_RE.search(normalize_line(line)))
        if citation_lines / len(lines) >= 0.6:
            return True
    return False


def is_front_matter_only_text(text: str) -> bool:
    normalized = normalize_line(text)
    if not normalized:
        return False
    if is_publication_header_text(normalized):
        return True
    if DATE_ONLY_RE.match(normalized):
        return True
    if is_supplement_license_text(normalized):
        return True
    if len(normalized) <= 260 and FRONT_MATTER_ONLY_RE.search(normalized):
        words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", normalized)
        if len(words) <= 35:
            return True
    return False


def is_supplement_license_text(text: str) -> bool:
    normalized = normalize_line(text)
    if not normalized:
        return False
    if not SUPPLEMENT_LICENSE_RE.search(normalized):
        return False
    # Drop supplement cover/license pages, but keep real supplement sections
    # such as architecture and training-data descriptions.
    content_markers = re.search(
        r"(architecture|implementation|training\s+data|experiment|method|results?|discussion|conclusion|"
        r"网络结构|训练|实验|方法|结果|讨论|结论)",
        normalized,
        re.IGNORECASE,
    )
    if content_markers:
        return False
    return True


def should_drop_retrieval_text(text: str, section_path: str = "", config: Any = None) -> tuple[bool, str]:
    """Return whether a rendered parent chunk should be excluded from indexes."""

    if config is not None and not getattr(config, "enable_document_cleaning", True):
        return False, ""
    if not normalize_line(text):
        return True, "empty"
    exclude_references = getattr(config, "clean_exclude_references", True) if config is not None else True
    if exclude_references:
        if is_reference_section_path(section_path):
            return True, "reference_section"
        if is_bibliography_dominated_text(text):
            return True, "bibliography_like"
    if is_front_matter_only_text(text):
        return True, "front_matter_or_header"
    return False, ""


def is_platform_metadata_block(block: Any) -> bool:
    text = normalize_line(getattr(block, "text", "") or "")
    return bool(PLATFORM_METADATA_RE.search(text))


def is_recommendation_heading(block: Any) -> bool:
    text = normalize_line(getattr(block, "text", "") or "")
    return bool(RECOMMENDATION_HEADING_RE.match(text))


def is_academic_heading(text: str, block: Any) -> bool:
    if getattr(block, "block_type", "") in {"title", "section_header"}:
        return False
    normalized = normalize_line(text)
    if len(normalized) > 80:
        return False
    return bool(ACADEMIC_HEADING_RE.match(normalized))


def split_candidate_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text.strip()] if text.strip() else []


def is_repeated_line_candidate(normalized: str) -> bool:
    if not normalized:
        return False
    if len(normalized) > 180:
        return False
    if len(normalized.split()) > 18:
        return False
    return True


def normalize_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
