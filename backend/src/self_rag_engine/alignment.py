from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from .cleaning import is_reference_heading
from .config import SelfRagConfig
from .document_parser import Block
from .grobid_parser import GrobidDocument, GrobidSection


def align_docling_grobid(
    body_blocks: List[Block],
    furniture_blocks: List[Block],
    grobid_doc: Optional[GrobidDocument],
    config: SelfRagConfig,
) -> tuple[List[Block], Dict[str, Any]]:
    heading_map = build_heading_map(grobid_doc)
    current_section_path = "Untitled"
    current_section_level = 1
    matched_headings = 0
    unmatched_docling_headings: List[str] = []
    abstract_marked = False
    aligned: List[Block] = []

    for block in body_blocks:
        if block.role == "asset":
            block.section_path = block.section_path or current_section_path
            block.section_level = block.section_level or current_section_level
            aligned.append(block)
            continue

        if block.block_type in {"title", "section_header"}:
            key = normalize_heading(block.text)
            matched = heading_map.get(key)
            if matched:
                current_section_path = matched.section_path
                current_section_level = matched.level
                block.grobid_ref = matched.section_id
                matched_headings += 1
            elif is_reference_heading(block):
                current_section_path = "References"
                current_section_level = block.level or 1
            else:
                current_section_path = block.text.strip() or current_section_path
                current_section_level = block.level or 1
                if key and len(unmatched_docling_headings) < 20:
                    unmatched_docling_headings.append(block.text.strip())
        elif grobid_doc and grobid_doc.abstract and not abstract_marked and likely_abstract(block, grobid_doc.abstract, aligned):
            current_section_path = "Abstract"
            current_section_level = 1
            abstract_marked = True

        block.section_path = block.section_path or current_section_path
        block.section_level = block.section_level or current_section_level
        if is_reference_heading(block) or is_reference_section(current_section_path):
            block.role = "reference"
            block.section_path = "References"
        aligned.append(block)

    for block in furniture_blocks:
        block.role = "furniture"
        block.section_path = block.section_path or "Furniture"
        block.section_level = block.section_level or 0
        aligned.append(block)

    report = {
        "grobid_available": bool(grobid_doc),
        "matched_headings": matched_headings,
        "unmatched_docling_headings": unmatched_docling_headings,
        "sections_from_grobid": len(getattr(grobid_doc, "sections", []) or []),
        "furniture_blocks": len(furniture_blocks),
        "reference_records": len(getattr(grobid_doc, "references", []) or []),
    }
    return aligned, report


def build_heading_map(grobid_doc: Optional[GrobidDocument]) -> Dict[str, GrobidSection]:
    if not grobid_doc:
        return {}
    mapping: Dict[str, GrobidSection] = {}
    for section in grobid_doc.sections:
        key = normalize_heading(section.heading)
        if key and key not in mapping:
            mapping[key] = section
    return mapping


def normalize_heading(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"^\s*(\d+(\.\d+)*|[ivxlcdm]+)\.?\s+", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_reference_section(section_path: str) -> bool:
    return any(normalize_heading(part) in {"references", "bibliography", "参考文献", "参考资料"} for part in section_path.split(">"))


def likely_abstract(block: Block, abstract: str, previous_blocks: List[Block]) -> bool:
    if len(previous_blocks) > 10 or block.block_type not in {"paragraph", "list_item"}:
        return False
    block_tokens = token_set(block.text)
    abstract_tokens = token_set(abstract)
    if not block_tokens or not abstract_tokens:
        return False
    return len(block_tokens & abstract_tokens) / max(len(block_tokens), 1) >= 0.4


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", (text or "").lower()))
