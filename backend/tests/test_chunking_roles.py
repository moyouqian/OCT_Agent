from self_rag_engine.chunking import build_parent_chunks, merge_small_sections
from self_rag_engine.config import SelfRagConfig
from self_rag_engine.document_parser import Block, ParsedDocument


def _sec(path, level, *texts):
    blocks = [
        Block(block_id=f"x{i}", block_type="paragraph", text=t, section_path=path, section_level=level)
        for i, t in enumerate(texts)
    ]
    return {"path": path, "level": level, "blocks": blocks}


def test_merge_child_into_parent():
    cfg = SelfRagConfig(parent_chunk_min_tokens=50, parent_chunk_max_tokens=500)
    sections = [_sec("Methods", 1, "short"), _sec("Methods > Setup", 2, "also short")]
    merged = merge_small_sections(sections, cfg)
    assert len(merged) == 1
    assert merged[0]["path"] == "Methods"


def test_named_siblings_not_merged():
    cfg = SelfRagConfig(parent_chunk_min_tokens=50, parent_chunk_max_tokens=500)
    sections = [_sec("Title > Abstract", 2, "short a"), _sec("Title > Methods", 2, "short b")]
    merged = merge_small_sections(sections, cfg)
    assert len(merged) == 2


def test_untitled_fragment_merged_into_previous():
    cfg = SelfRagConfig(parent_chunk_min_tokens=50, parent_chunk_max_tokens=500)
    sections = [_sec("Intro", 1, "real content"), _sec("Untitled", 1, "stray")]
    merged = merge_small_sections(sections, cfg)
    assert len(merged) == 1
    assert merged[0]["path"] == "Intro"


def test_parent_chunking_uses_roles_and_section_path_metadata():
    parsed = ParsedDocument(
        doc_id="doc_test",
        source_path="paper.pdf",
        source_type="paper",
        file_type="pdf",
        title="Paper",
        content_hash="hash",
        parser="test",
        blocks=[
            Block(
                block_id="b1",
                block_type="paragraph",
                text="Useful introduction text.",
                section_path="Introduction",
                section_level=1,
                role="body",
            ),
            Block(
                block_id="b2",
                block_type="paragraph",
                text="page footer",
                section_path="Introduction",
                role="furniture",
            ),
            Block(
                block_id="b3",
                block_type="paragraph",
                text="[1] Reference item.",
                section_path="References",
                role="reference",
            ),
            Block(block_id="b4", block_type="paragraph", text="", section_path="Introduction", role="noise"),
            Block(
                block_id="b5",
                block_type="table",
                text="Table values",
                caption="Table 1. Values",
                section_path="Introduction",
                role="asset",
            ),
        ],
    )

    parents, assets = build_parent_chunks(parsed, SelfRagConfig(parent_chunk_max_tokens=200))

    assert len(parents) == 1
    assert parents[0].section_path == "Introduction"
    assert "Useful introduction text." in parents[0].text
    assert "Reference item" not in parents[0].text
    assert "page footer" not in parents[0].text
    assert len(assets) == 1
    assert assets[0].label == "Table 1"
