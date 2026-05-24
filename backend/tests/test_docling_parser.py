from self_rag_engine.docling_parser import extract_docling_blocks


class FakeItem:
    def __init__(self, ref, text, label="text"):
        self.self_ref = ref
        self.text = text
        self.label = label
        self.prov = []


class FakeRef:
    def __init__(self, ref):
        self.ref = ref


class FakeNode:
    def __init__(self, children):
        self.children = children


class FakeDoc:
    def __init__(self):
        self.texts = [FakeItem("#/texts/0", "first paragraph"), FakeItem("#/texts/1", "second paragraph")]
        self.tables = [FakeItem("#/tables/0", "table text", label="table")]
        self.pictures = []
        self.groups = []
        self.body = FakeNode([FakeRef("#/texts/0"), FakeRef("#/tables/0"), FakeRef("#/texts/1")])
        self.furniture = FakeNode([])


def test_extract_docling_blocks_uses_body_reading_order():
    body_blocks, furniture_blocks, report = extract_docling_blocks(FakeDoc())

    assert [block.text for block in body_blocks] == ["first paragraph", "table text", "second paragraph"]
    assert body_blocks[1].block_type == "table"
    assert body_blocks[1].role == "asset"
    assert furniture_blocks == []
    assert report["fallback"] is False
