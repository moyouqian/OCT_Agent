from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import re

from .config import SelfRagConfig
from .cleaning import should_drop_retrieval_text
from .document_parser import Block, ParsedDocument, stable_id
from .text_utils import SplitPiece, estimated_tokens, split_by_estimated_tokens_with_metadata


@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    parent_index: int
    title: str
    source_path: str
    source_type: str
    file_type: str
    section_path: str
    section_level: int
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    block_ids: List[str] = field(default_factory=list)


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    child_index: int
    title: str
    source_path: str
    source_type: str
    file_type: str
    section_path: str
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    embedding_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetRecord:
    asset_id: str
    parent_id: Optional[str]
    doc_id: str
    asset_type: str
    asset_index: int
    label: str
    caption: str
    page_start: Optional[int]
    page_end: Optional[int]
    bbox: List[Dict[str, Any]] = field(default_factory=list)
    file_path: str = ""
    text_repr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_parent_chunks(parsed: ParsedDocument, config: SelfRagConfig) -> tuple[List[ParentChunk], List[AssetRecord]]:
    if any(getattr(block, "section_path", "") for block in parsed.blocks):
        sections = sectionize_by_block_metadata(parsed.blocks)
    else:
        sections = sectionize(parsed.blocks)
    parents: List[ParentChunk] = []
    assets: List[AssetRecord] = []
    asset_index = 0

    for section in sections:
        text_blocks = [
            block
            for block in section["blocks"]
            if block.block_type not in {"figure", "table", "equation"}
            and getattr(block, "role", "body") not in {"reference", "furniture", "noise"}
        ]
        asset_blocks = [
            block
            for block in section["blocks"]
            if block.block_type in {"figure", "table", "equation"} or getattr(block, "role", "") == "asset"
        ]
        section_text = render_section_text(section["path"], text_blocks)
        if not section_text.strip() and not asset_blocks:
            continue

        pieces = split_parent_text(section_text, config.parent_chunk_max_tokens)
        if not pieces and asset_blocks:
            pieces = [SplitPiece(section["path"], "asset_only")]

        section_parent_ids: List[str] = []
        for piece in pieces:
            if not asset_blocks and is_heading_only_piece(piece.text, section["path"]):
                continue
            should_drop, _ = should_drop_retrieval_text(piece.text, section["path"], config)
            if should_drop:
                continue
            parent_index = len(parents)
            page_start, page_end = page_range(section["blocks"])
            parent_id = stable_id("parent", parsed.doc_id, parent_index, section["path"], piece.text)
            content_hash = content_hash_text(piece.text)
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=parsed.doc_id,
                parent_index=parent_index,
                title=parsed.title,
                source_path=parsed.source_path,
                source_type=parsed.source_type,
                file_type=parsed.file_type,
                section_path=section["path"] or parsed.title,
                section_level=section["level"],
                page_start=page_start,
                page_end=page_end,
                text=piece.text,
                content_hash=content_hash,
                block_ids=[block.block_id for block in section["blocks"]],
                metadata={
                    "content_types": sorted({block.block_type for block in section["blocks"]}),
                    "roles": sorted({getattr(block, "role", "body") for block in section["blocks"]}),
                    "citation_ids": sorted(
                        {
                            citation_id
                            for block in section["blocks"]
                            for citation_id in (getattr(block, "citation_ids", None) or [])
                        }
                    ),
                    "alignment": parsed.metadata.get("alignment", {}),
                    "estimated_tokens": estimated_tokens(piece.text),
                    "split_reason": piece.split_reason,
                    "parser": parsed.parser,
                },
            )
            parents.append(parent)
            section_parent_ids.append(parent.parent_id)

        parent_for_assets = section_parent_ids[-1] if section_parent_ids else None
        for block in asset_blocks:
            asset_index += 1
            label = asset_label(block, asset_index)
            assets.append(
                AssetRecord(
                    asset_id=stable_id("asset", parsed.doc_id, block.block_id, block.block_type, label),
                    parent_id=parent_for_assets,
                    doc_id=parsed.doc_id,
                    asset_type=block.block_type,
                    asset_index=asset_index,
                    label=label,
                    caption=block.caption or block.text,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    bbox=block.bbox_list,
                    file_path=block.asset_file_path,
                    text_repr=block.text if block.block_type in {"table", "equation"} else "",
                    metadata={"docling_ref": block.docling_ref} if block.docling_ref else {},
                )
            )

    return parents, assets


def build_child_chunks(parent: ParentChunk, config: SelfRagConfig) -> List[ChildChunk]:
    pieces = split_child_text(parent.text, config.child_chunk_size, config.child_chunk_overlap)
    children = []
    for index, piece in enumerate(pieces):
        embedding_text = contextualize_child(parent, piece.text)
        child_id = stable_id("child", parent.parent_id, index, piece.text)
        metadata = {
            "child_id": child_id,
            "parent_id": parent.parent_id,
            "doc_id": parent.doc_id,
            "source_path": parent.source_path,
            "source_type": parent.source_type,
            "file_type": parent.file_type,
            "title": parent.title,
            "section_path": parent.section_path,
            "page_start": parent.page_start,
            "page_end": parent.page_end,
            "child_index": index,
            "parent_index": parent.parent_index,
            "content_type": "text",
            "chunk_level": "child",
            "estimated_tokens": estimated_tokens(piece.text),
            "split_reason": piece.split_reason,
        }
        children.append(
            ChildChunk(
                child_id=child_id,
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                child_index=index,
                title=parent.title,
                source_path=parent.source_path,
                source_type=parent.source_type,
                file_type=parent.file_type,
                section_path=parent.section_path,
                page_start=parent.page_start,
                page_end=parent.page_end,
                text=piece.text,
                embedding_text=embedding_text,
                metadata=metadata,
            )
        )
    return children


def sectionize(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current_path: List[tuple[int, str]] = []
    current: Optional[Dict[str, Any]] = None

    def start_section(path: str, level: int) -> Dict[str, Any]:
        section = {"path": path, "level": level, "blocks": []}
        sections.append(section)
        return section

    for block in blocks:
        if block.block_type in {"title", "section_header"}:
            level = block.level or (1 if block.block_type == "title" else 2)
            current_path = [(lvl, text) for lvl, text in current_path if lvl < level]
            current_path.append((level, block.text.strip()))
            path = " > ".join(text for _, text in current_path if text)
            current = start_section(path, level)
            current["blocks"].append(block)
            continue
        if current is None:
            current = start_section("Untitled", 1)
        current["blocks"].append(block)
    return sections


def sectionize_by_block_metadata(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for block in blocks:
        role = getattr(block, "role", "body")
        if role in {"noise", "reference", "furniture"}:
            continue
        section_path = getattr(block, "section_path", "") or "Untitled"
        section_level = getattr(block, "section_level", 0) or getattr(block, "level", 0) or 1
        if current is None or current["path"] != section_path:
            current = {"path": section_path, "level": section_level, "blocks": []}
            sections.append(current)
        current["blocks"].append(block)
    return sections


def render_section_text(section_path: str, blocks: List[Block]) -> str:
    parts = []
    if section_path and section_path != "Untitled":
        parts.append(section_path)
    for block in blocks:
        if getattr(block, "role", "body") in {"noise", "reference", "furniture"}:
            continue
        text = (getattr(block, "text_for_display", "") or block.text).strip()
        if not text:
            continue
        if block.block_type == "section_header" and text in section_path:
            continue
        parts.append(text)
    return "\n\n".join(dict.fromkeys(parts)).strip()


def split_parent_text(text: str, max_tokens: int) -> List[SplitPiece]:
    return split_by_estimated_tokens_with_metadata(text, max_tokens, overlap=0)


def is_heading_only_piece(text: str, section_path: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", (text or "")).strip()
    normalized_section = re.sub(r"\s+", " ", (section_path or "")).strip()
    return bool(normalized_text and normalized_text == normalized_section)


def split_child_text(text: str, chunk_size: int, overlap: int) -> List[SplitPiece]:
    return split_by_estimated_tokens_with_metadata(text, chunk_size, overlap=overlap)


def contextualize_child(parent: ParentChunk, child_text: str) -> str:
    page = page_label(parent.page_start, parent.page_end)
    header = [
        f"Title: {parent.title}",
        f"Source type: {parent.source_type}",
        f"Section: {parent.section_path}",
    ]
    if page:
        header.append(f"Page: {page}")
    header.append("Content:")
    return "\n".join(header) + f"\n{child_text.strip()}"


def page_range(blocks: Iterable[Block]) -> tuple[Optional[int], Optional[int]]:
    pages = []
    for block in blocks:
        if block.page_start is not None:
            pages.append(block.page_start)
        if block.page_end is not None:
            pages.append(block.page_end)
    if not pages:
        return None, None
    return min(pages), max(pages)


def page_label(page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"


def content_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asset_label(block: Block, index: int) -> str:
    if block.caption:
        match = re.search(r"\b(Figure|Fig\.|Table|Equation|Eq\.)\s*\d+[A-Za-z]?", block.caption, re.I)
        if match:
            return match.group(0)
    prefix = {"figure": "Figure", "table": "Table", "equation": "Equation"}.get(block.block_type, "Asset")
    return f"{prefix} {index}"
