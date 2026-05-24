from self_rag_engine import text_utils
from self_rag_engine.text_utils import (
    estimated_tokens,
    split_by_estimated_tokens,
    split_by_estimated_tokens_with_metadata,
    tokenize_for_search,
)


def test_estimated_tokens_counts_chinese_without_spaces():
    text = "\u5149\u5b66\u76f8\u5e72\u5c42\u6790\u53ef\u4ee5\u7528\u4e8e\u65e0\u635f\u68c0\u6d4b\u3002"

    assert estimated_tokens(text) >= 10


def test_split_by_estimated_tokens_splits_chinese_on_sentence_boundaries():
    text = (
        "\u7b2c\u4e00\u53e5\u7528\u4e8e\u4ecb\u7ecd\u7814\u7a76\u80cc\u666f\u3002"
        "\u7b2c\u4e8c\u53e5\u63cf\u8ff0\u5b9e\u9a8c\u65b9\u6cd5\u3002"
        "\u7b2c\u4e09\u53e5\u7ed9\u51fa\u7ed3\u679c\u3002"
    )

    chunks = split_by_estimated_tokens(text, max_tokens=14, overlap=0)

    assert len(chunks) >= 2
    assert chunks[0].endswith("\u3002")


def test_tokenize_for_search_supports_domain_terms_and_bigrams(tmp_path):
    domain_terms = tmp_path / "domain_terms.txt"
    domain_terms.write_text(
        "\u5149\u5b66\u76f8\u5e72\u5c42\u6790\n\u65e0\u635f\u68c0\u6d4b\n",
        encoding="utf-8",
    )

    tokens = tokenize_for_search(
        "\u5149\u5b66\u76f8\u5e72\u5c42\u6790\u5728\u65e0\u635f\u68c0\u6d4b\u9886\u57df\u7684\u5e94\u7528",
        domain_terms_path=str(domain_terms),
    )

    assert "\u5149\u5b66\u76f8\u5e72\u5c42\u6790" in tokens
    assert "\u65e0\u635f\u68c0\u6d4b" in tokens
    assert "\u76f8\u5e72" in tokens
    assert "\u5c42\u6790" in tokens


def test_tokenize_for_search_supports_chinese_and_english_terms():
    tokens = tokenize_for_search("\u76f8\u4f4d\u654f\u611f OCT strain-estimation PhS-OCE")

    assert "oct" in tokens
    assert "strain-estimation" in tokens
    assert "phs-oce" in tokens
    assert "\u76f8\u4f4d" in tokens


def test_tokenize_for_search_falls_back_when_jieba_unavailable(monkeypatch):
    monkeypatch.setattr(text_utils, "load_jieba", lambda: None)

    tokens = tokenize_for_search("\u5149\u5b66\u76f8\u5e72\u5c42\u6790", enable_cjk_bigrams=False)

    assert "\u5149" in tokens
    assert "\u5b66" in tokens


def test_split_metadata_marks_hard_limit_only_for_overlong_sentence():
    text = "\u5b9e\u9a8c" * 40

    pieces = split_by_estimated_tokens_with_metadata(text, max_tokens=10, overlap=0)

    assert len(pieces) > 1
    assert {piece.split_reason for piece in pieces} == {"hard_limit"}


def test_overlap_does_not_create_empty_or_oversized_chunks():
    text = (
        "\u7b2c\u4e00\u53e5\u7528\u4e8e\u4ecb\u7ecd\u7814\u7a76\u80cc\u666f\u3002"
        "\u7b2c\u4e8c\u53e5\u63cf\u8ff0\u5b9e\u9a8c\u65b9\u6cd5\u3002"
        "\u7b2c\u4e09\u53e5\u7ed9\u51fa\u7ed3\u679c\u3002"
    )

    chunks = split_by_estimated_tokens(text, max_tokens=14, overlap=8)

    assert all(chunk.strip() for chunk in chunks)
    assert all(estimated_tokens(chunk) <= 14 for chunk in chunks)
