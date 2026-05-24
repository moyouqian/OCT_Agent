import pytest

from self_rag_engine.config import SelfRagConfig
from self_rag_engine import document_parser
from self_rag_engine.document_parser import OcrDecision, build_pdf_pipeline_options, decide_docling_ocr


class FakeOcrOptions:
    def __init__(self, lang=None, force_full_page_ocr=False):
        self.lang = lang or []
        self.force_full_page_ocr = force_full_page_ocr


class FakePdfPipelineOptions:
    def __init__(self):
        self.do_ocr = None
        self.images_scale = None
        self.generate_page_images = None
        self.generate_picture_images = None
        self.ocr_options = None


def build_options(mode: str, ocr_decision=None):
    config = SelfRagConfig(
        docling_enable_ocr=mode,
        docling_image_scale=1.25,
        docling_generate_page_images=True,
        docling_generate_picture_images=True,
    )
    return build_pdf_pipeline_options(config, FakePdfPipelineOptions, FakeOcrOptions, ocr_decision)


def test_docling_ocr_auto_without_file_keeps_region_ocr_enabled():
    options = build_options("auto")

    assert options.do_ocr is True
    assert options.ocr_options.force_full_page_ocr is False
    assert options.ocr_options.lang == ["chinese", "english"]
    assert options.images_scale == 1.25
    assert options.generate_page_images is True
    assert options.generate_picture_images is True


def test_docling_ocr_auto_disables_ocr_when_text_layer_is_present(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    config = SelfRagConfig(docling_enable_ocr="auto", docling_auto_ocr_min_text_chars=300)

    monkeypatch.setattr(
        document_parser,
        "inspect_pdf_text_layer",
        lambda file_path, sample_pages: {
            "total_pages": 8,
            "sampled_pages": 5,
            "text_chars": 1200,
            "error": "",
        },
    )

    decision = decide_docling_ocr(pdf, config)
    options = build_options("auto", decision)

    assert decision.enabled is False
    assert decision.reason == "text layer detected (1200 chars)"
    assert options.do_ocr is False
    assert options.ocr_options.force_full_page_ocr is False


def test_docling_ocr_auto_enables_ocr_when_text_layer_is_sparse(monkeypatch, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    config = SelfRagConfig(docling_enable_ocr="auto", docling_auto_ocr_min_text_chars=300)

    monkeypatch.setattr(
        document_parser,
        "inspect_pdf_text_layer",
        lambda file_path, sample_pages: {
            "total_pages": 8,
            "sampled_pages": 5,
            "text_chars": 25,
            "error": "",
        },
    )

    decision = decide_docling_ocr(pdf, config)
    options = build_options("auto", decision)

    assert decision.enabled is True
    assert decision.force_full_page is False
    assert "sparse text layer" in decision.reason
    assert options.do_ocr is True
    assert options.ocr_options.force_full_page_ocr is False


def test_docling_ocr_true_forces_full_page_ocr():
    options = build_options("true")

    assert options.do_ocr is True
    assert options.ocr_options.force_full_page_ocr is True


def test_docling_ocr_false_disables_ocr():
    options = build_options("false")

    assert options.do_ocr is False
    assert options.ocr_options.force_full_page_ocr is False


def test_docling_ocr_rejects_unknown_mode():
    config = SelfRagConfig(docling_enable_ocr="sometimes")

    with pytest.raises(ValueError):
        build_pdf_pipeline_options(config, FakePdfPipelineOptions, FakeOcrOptions)
