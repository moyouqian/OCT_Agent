from self_rag_engine.ingestion import mojibake_score


def test_mojibake_detects_cid_and_pua_and_replacement():
    assert mojibake_score("正常文本 (cid:123)(cid:45) 更多") > 0
    assert mojibake_score("text  here") > 0
    assert mojibake_score("bad �� text") > 0


def test_mojibake_clean_text_scores_zero():
    assert mojibake_score("这是正常的中文与 English 混排文本。") == 0
