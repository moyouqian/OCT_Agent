from self_rag_engine.cleaning import clean_blocks, should_drop_retrieval_text
from self_rag_engine.config import SelfRagConfig
from self_rag_engine.document_parser import Block


def test_clean_blocks_marks_references_section():
    blocks = [
        Block(block_id="b0", block_type="section_header", text="Methods", level=2),
        Block(block_id="b1", block_type="paragraph", text="Useful method text."),
        Block(block_id="b2", block_type="section_header", text="References", level=2),
        Block(block_id="b3", block_type="paragraph", text="[1] Some paper."),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["body", "body", "reference", "reference"]
    assert report["role_counts"]["reference"] == 2


def test_clean_blocks_marks_chinese_references_section_with_colon():
    blocks = [
        Block(block_id="b0", block_type="section_header", text="5 总结与展望", level=2),
        Block(block_id="b1", block_type="paragraph", text="未来可进一步提高系统速度。"),
        Block(block_id="b2", block_type="section_header", text="参考文献：", level=2),
        Block(block_id="b3", block_type="paragraph", text="[1] HUANG D, SWANSON E A. Optical coherence tomography [J]. Science, 1991."),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["body", "body", "reference", "reference"]
    assert report["role_counts"]["reference"] == 2


def test_clean_blocks_marks_references_and_notes_section():
    blocks = [
        Block(block_id="b0", block_type="section_header", text="Discussion", level=2),
        Block(block_id="b1", block_type="paragraph", text="Useful discussion text."),
        Block(block_id="b2", block_type="section_header", text="REFERENCES AND NOTES", level=2),
        Block(block_id="b3", block_type="paragraph", text="Q. Chen, V. Koltun, in Proceedings of ICCV, 2017."),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["body", "body", "reference", "reference"]
    assert report["role_counts"]["reference"] == 2


def test_references_section_terminates_at_next_sibling_heading():
    blocks = [
        Block(block_id="b0", block_type="section_header", text="References", level=2),
        Block(block_id="b1", block_type="paragraph", text="[1] Some paper, 2019."),
        Block(block_id="b2", block_type="section_header", text="Appendix A", level=2),
        Block(block_id="b3", block_type="paragraph", text="Supplementary derivation kept as body."),
    ]
    cleaned, _ = clean_blocks(blocks, SelfRagConfig())
    assert [b.role for b in cleaned] == ["reference", "reference", "body", "body"]


def test_clean_blocks_marks_repeated_page_headers_and_page_numbers_as_noise():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="Journal Header", page_start=1, page_end=1),
        Block(block_id="b1", block_type="paragraph", text="1", page_start=1, page_end=1),
        Block(block_id="b2", block_type="paragraph", text="Actual page one content.", page_start=1, page_end=1),
        Block(block_id="b3", block_type="paragraph", text="Journal Header", page_start=2, page_end=2),
        Block(block_id="b4", block_type="paragraph", text="Actual page two content.", page_start=2, page_end=2),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["noise", "noise", "body", "noise", "body"]
    assert "Journal Header" in report["repeated_lines"]


def test_clean_blocks_marks_publication_front_matter_noise():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="Checkfor", page_start=1),
        Block(block_id="b1", block_type="paragraph", text="updates", page_start=1),
        Block(block_id="b2", block_type="title", text="Optics Letters", page_start=1),
        Block(block_id="b3", block_type="title", text="Deep-learning-based approach", page_start=1),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["noise", "noise", "noise", "body"]
    assert report["noise_counts"]["publication_noise"] == 3


def test_clean_blocks_marks_publication_volume_header_as_noise():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="Vol. 38 No. 6 November 2021 第 38 卷 第 6 期 2021 年 11 月"),
        Block(block_id="b1", block_type="paragraph", text="相衬光学相干层析可以用于无损检测。"),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["noise", "body"]
    assert report["noise_counts"]["publication_header"] == 1


def test_parent_chunk_filter_drops_reference_section_path():
    text = (
        "5 总结与展望 > 参考文献：\n"
        "HUANG D, SWANSON E A, LIN C P, et al. Optical coherence tomography [J]. "
        "Science, 1991, 254(5035): 1178-1181."
    )

    should_drop, reason = should_drop_retrieval_text(text, "5 总结与展望 > 参考文献：", SelfRagConfig())

    assert should_drop is True
    assert reason == "reference_section"


def test_parent_chunk_filter_drops_bibliography_like_text_without_heading():
    text = (
        "35-38.\n"
        "[1] HUANG D, SWANSON E A, LIN C P, et al. Optical coherence tomography [J]. "
        "Science, 1991, 254(5035): 1178-1181.\n"
        "[2] FUJIMOTO J G, PITRIS C, BOPPART S A, et al. Optical coherence tomography [J]. "
        "Neoplasia, 2000, 2(1-2): 9-25.\n"
        "[3] SU R, KIRILLIN M, CHANG E W, et al. Perspectives of OCT [J]."
    )

    should_drop, reason = should_drop_retrieval_text(text, "35-38.", SelfRagConfig())

    assert should_drop is True
    assert reason == "bibliography_like"


def test_parent_chunk_filter_drops_date_only_front_matter():
    should_drop, reason = should_drop_retrieval_text("2021 年 11 月", "Untitled", SelfRagConfig())

    assert should_drop is True
    assert reason == "front_matter_or_header"


def test_parent_chunk_filter_drops_supplement_license_front_matter():
    text = (
        "Deep-learning-based approach for strain estimation in phase-sensitive optical coherence elastography: supplement "
        "BO DONG, NAIXING HUANG, YULEI BAI, AND SHENGLI XIE. "
        "This supplement published with Optica Publishing Group on 23 November 2021 by The Authors under the terms "
        "of the Creative Commons Attribution 4.0 License in the format provided by the authors and unedited. "
        "Supplement DOI: https://doi.org/10.6084/m9.figshare.16955140 Parent Article DOI: https://doi.org/10.1364/OL.446403"
    )

    should_drop, reason = should_drop_retrieval_text(
        text,
        "Deep-learning-based approach for strain estimation in phase-sensitive optical coherence elastography: supplement",
        SelfRagConfig(),
    )

    assert should_drop is True
    assert reason == "front_matter_or_header"


def test_clean_blocks_promotes_academic_heading_paragraphs():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="1. Introduction", page_start=1),
        Block(block_id="b1", block_type="paragraph", text="Useful body.", page_start=1),
    ]

    cleaned, _ = clean_blocks(blocks, SelfRagConfig())

    assert cleaned[0].block_type == "section_header"
    assert cleaned[0].level == 2


def test_clean_blocks_promotes_chinese_academic_heading():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="三、方法", page_start=1),
        Block(block_id="b1", block_type="paragraph", text="本文采用深度学习方法。", page_start=1),
    ]
    cleaned, _ = clean_blocks(blocks, SelfRagConfig())
    assert cleaned[0].block_type == "section_header"
    assert cleaned[0].level == 2


def test_clean_blocks_marks_platform_metadata_and_recommendations_as_noise():
    blocks = [
        Block(block_id="b0", block_type="paragraph", text="Citation: 科学通报 65, 2094", page_start=1),
        Block(block_id="b1", block_type="section_header", text="Articles you may be interested in", page_start=1),
        Block(block_id="b2", block_type="section_header", text="Some unrelated recommended article", page_start=1),
        Block(block_id="b3", block_type="section_header", text="1 光学相干层析 (OCT) 成像原理", page_start=2),
        Block(block_id="b4", block_type="paragraph", text="正文内容。", page_start=2),
    ]

    cleaned, report = clean_blocks(blocks, SelfRagConfig())

    assert [block.role for block in cleaned] == ["noise", "noise", "noise", "body", "body"]
    assert report["noise_counts"]["platform_metadata"] == 1
    assert report["noise_counts"]["recommendation_noise"] == 2


def test_code_block_preserved_verbatim():
    code = "def f():\n    x = 1-\n    return x"
    blocks = [Block(block_id="b0", block_type="code", text=code)]
    cleaned, _ = clean_blocks(blocks, SelfRagConfig(), source_type="note")
    assert cleaned[0].text == code
    assert cleaned[0].role == "body"


def test_note_mode_does_not_swallow_after_references_heading():
    blocks = [
        Block(block_id="b0", block_type="section_header", text="参考资料", level=2),
        Block(block_id="b1", block_type="paragraph", text="https://example.com 笔记链接"),
        Block(block_id="b2", block_type="section_header", text="后续笔记", level=2),
        Block(block_id="b3", block_type="paragraph", text="这段必须保留。"),
    ]
    cleaned, _ = clean_blocks(blocks, SelfRagConfig(), source_type="note")
    assert all(b.role != "reference" for b in cleaned)
    assert cleaned[-1].text == "这段必须保留。"
