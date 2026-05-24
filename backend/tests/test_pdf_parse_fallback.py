from pathlib import Path

from self_rag_engine.config import SelfRagConfig
from self_rag_engine.document_parser import Block, parse_pdf_docling_grobid


class FakeDoc:
    pass


def test_parse_pdf_docling_grobid_falls_back_when_grobid_fails(monkeypatch, tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    config = SelfRagConfig(parsed_dir=str(tmp_path / "parsed"))

    import self_rag_engine.document_parser as document_parser
    import self_rag_engine.docling_parser as docling_parser
    import self_rag_engine.grobid_parser as grobid_parser

    monkeypatch.setattr(
        document_parser,
        "parse_docling_pdf_payload",
        lambda file_path, config: ("doc_test", "hash", FakeDoc(), "", type("Ocr", (), {"as_dict": lambda self: {}})()),
    )
    monkeypatch.setattr(
        docling_parser,
        "extract_docling_blocks",
        lambda doc, fallback_markdown="": (
            [Block(block_id="b1", block_type="title", text="Paper Title", section_path="Paper Title")],
            [],
            {"fallback": False},
        ),
    )
    monkeypatch.setattr(grobid_parser, "parse_pdf_with_grobid", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    parsed = parse_pdf_docling_grobid(source, config)

    assert parsed.parser == "docling"
    assert parsed.metadata["grobid"]["status"] == "failed"
    assert parsed.title == "Paper Title"
