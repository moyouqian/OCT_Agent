from __future__ import annotations

from typing import Any, Dict, List

from .types import Document


def build_citations(documents: List[Document]) -> List[Dict[str, Any]]:
    citations = []
    for index, document in enumerate(documents, start=1):
        meta = document.get("meta") or document.get("metadata") or {}
        citation_id = meta.get("citation_id") or f"S{index}"
        if not meta.get("parent_id"):
            continue
        citations.append(
            {
                "citation_id": citation_id,
                "title": meta.get("title", ""),
                "page": format_page(meta.get("page_start"), meta.get("page_end")),
                "section": meta.get("section_path", ""),
                "source_path": meta.get("source_path") or meta.get("source", ""),
                "related_assets": meta.get("related_assets", []),
            }
        )
    return citations


def format_page(page_start: Any, page_end: Any) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"
