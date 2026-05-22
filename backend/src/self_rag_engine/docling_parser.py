from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .document_parser import (
    Block,
    docling_block_type,
    heading_level,
    item_caption,
    item_text,
    markdown_blocks,
    provenance,
)


def get_self_ref(obj: Any) -> str:
    for attr in ("self_ref", "cref", "ref"):
        value = getattr(obj, attr, None)
        if value:
            return str(value)
    if isinstance(obj, dict):
        for key in ("self_ref", "$ref", "cref", "ref"):
            value = obj.get(key)
            if value:
                return str(value)
    return ""


def build_ref_index(doc: Any) -> Dict[str, Any]:
    index: Dict[str, Any] = {}
    for attr in ("texts", "tables", "pictures", "groups"):
        for item in getattr(doc, attr, None) or []:
            ref = get_self_ref(item)
            if ref:
                index[ref] = item
    return index


def resolve_ref(node: Any, ref_index: Dict[str, Any]) -> Any:
    ref = get_self_ref(node)
    if ref and ref in ref_index and ref_index[ref] is not node:
        return ref_index[ref]
    return node


def walk_docling_tree(node: Any, ref_index: Dict[str, Any]) -> Iterable[Any]:
    resolved = resolve_ref(node, ref_index)
    children = getattr(resolved, "children", None)
    if children is None and isinstance(resolved, dict):
        children = resolved.get("children")
    if children:
        for child in children:
            yield from walk_docling_tree(child, ref_index)
        return
    yield resolved


def extract_docling_blocks(doc: Any, fallback_markdown: str = "") -> tuple[List[Block], List[Block], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "fallback": False,
        "warnings": [],
        "body_blocks": 0,
        "furniture_blocks": 0,
    }
    ref_index = build_ref_index(doc)
    try:
        body_items = list(walk_docling_tree(getattr(doc, "body"), ref_index))
        if not body_items:
            raise ValueError("empty Docling body tree")
        body_blocks = blocks_from_items(body_items, tree="body")
    except Exception:
        report["fallback"] = True
        report["warnings"].append("Docling body traversal failed; used legacy item order fallback.")
        body_blocks = legacy_blocks(doc, fallback_markdown)

    furniture_blocks: List[Block] = []
    furniture = getattr(doc, "furniture", None)
    if furniture is not None:
        try:
            furniture_blocks = blocks_from_items(list(walk_docling_tree(furniture, ref_index)), tree="furniture")
        except Exception as exc:
            report["warnings"].append(f"Docling furniture traversal failed: {exc}")

    report["body_blocks"] = len(body_blocks)
    report["furniture_blocks"] = len(furniture_blocks)
    return body_blocks, furniture_blocks, report


def legacy_blocks(doc: Any, fallback_markdown: str = "") -> List[Block]:
    items = []
    for attr in ("texts", "tables", "pictures"):
        items.extend(getattr(doc, attr, None) or [])
    if not items and fallback_markdown:
        blocks = markdown_blocks(fallback_markdown)
        for block in blocks:
            block.source_parser = "docling_markdown_fallback"
        return blocks
    return blocks_from_items(items, tree="body")


def blocks_from_items(items: Iterable[Any], tree: str) -> List[Block]:
    blocks: List[Block] = []
    seen_refs = set()
    for item in items:
        ref = get_self_ref(item)
        if ref and ref in seen_refs:
            continue
        if ref:
            seen_refs.add(ref)
        text = item_text(item).strip()
        label = str(getattr(item, "label", "") or getattr(item, "name", "") or "").lower()
        block_type = docling_block_type(label, item)
        page_start, page_end, bboxes = provenance(item)
        caption = item_caption(item)
        if not text and caption:
            text = caption
        if not text and block_type not in {"figure", "table", "equation"}:
            continue
        role = "furniture" if tree == "furniture" else ("asset" if block_type in {"figure", "table", "equation"} else "body")
        blocks.append(
            Block(
                block_id=f"b{len(blocks)}" if tree == "body" else f"f{len(blocks)}",
                block_type=block_type,
                text=text,
                level=heading_level(block_type, text),
                page_start=page_start,
                page_end=page_end,
                docling_ref=ref,
                bbox_list=bboxes,
                caption=caption,
                role=role,
                source_parser="docling",
                raw_payload={"label": label, "tree": tree},
            )
        )
    return blocks
