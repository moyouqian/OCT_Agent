from self_rag_engine.grobid_parser import parse_grobid_tei


def test_parse_grobid_tei_extracts_metadata_sections_and_references():
    tei = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title>A Test Paper</title>
            <author><persName><forename>Ada</forename><surname>Lovelace</surname></persName></author>
          </titleStmt>
          <publicationStmt><date when="2024-01-01"/></publicationStmt>
        </fileDesc>
        <profileDesc><abstract><p>This is the abstract.</p></abstract></profileDesc>
      </teiHeader>
      <text>
        <body>
          <div><head>Introduction</head><p>Body text <ref type="bibr" target="#b0">[1]</ref>.</p></div>
        </body>
        <back>
          <listBibl>
            <biblStruct xml:id="b0">
              <analytic><title>Reference Title</title><author><persName><forename>Grace</forename><surname>Hopper</surname></persName></author></analytic>
              <monogr><imprint><date when="2020"/></imprint></monogr>
              <idno type="DOI">10.123/test</idno>
              <note type="raw_reference">Hopper, G. Reference Title.</note>
            </biblStruct>
          </listBibl>
        </back>
      </text>
    </TEI>
    """

    parsed = parse_grobid_tei(tei, doc_id="doc_test")

    assert parsed.title == "A Test Paper"
    assert parsed.abstract == "This is the abstract."
    assert parsed.sections[0].heading == "Introduction"
    assert parsed.sections[0].paragraphs[0].citation_ids == ["b0"]
    assert parsed.references[0].title == "Reference Title"
    assert parsed.references[0].year == "2020"
    assert parsed.references[0].doi == "10.123/test"
