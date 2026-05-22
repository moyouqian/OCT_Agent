from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import re
import xml.etree.ElementTree as ET

from .config import SelfRagConfig
from .document_parser import ReferenceRecord, stable_id


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class GrobidParagraph:
    paragraph_id: str
    text: str
    citation_ids: List[str] = field(default_factory=list)
    raw_xml_id: str = ""


@dataclass
class GrobidSection:
    section_id: str
    section_path: str
    level: int
    heading: str
    paragraphs: List[GrobidParagraph] = field(default_factory=list)


@dataclass
class GrobidDocument:
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    abstract: str = ""
    sections: List[GrobidSection] = field(default_factory=list)
    references: List[ReferenceRecord] = field(default_factory=list)
    raw_tei_path: str = ""
    warnings: List[str] = field(default_factory=list)


def parse_pdf_with_grobid(file_path: Path, doc_id: str, config: SelfRagConfig) -> GrobidDocument:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests to use the optional GROBID parser.") from exc

    url = f"{config.grobid_url.rstrip('/')}/api/processFulltextDocument"
    data = [("includeRawCitations", "1" if config.grobid_include_raw_citations else "0")]
    if config.grobid_use_coordinates:
        data.extend([("teiCoordinates", name) for name in ("ref", "figure", "biblStruct", "formula")])
    with file_path.open("rb") as handle:
        response = requests.post(
            url,
            data=data,
            files={"input": (file_path.name, handle, "application/pdf")},
            timeout=config.grobid_timeout,
        )
    response.raise_for_status()
    parsed_dir = Path(config.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    raw_tei_path = parsed_dir / f"{doc_id}.grobid.tei.xml"
    raw_tei_path.write_text(response.text, encoding="utf-8")
    return parse_grobid_tei(response.text, doc_id=doc_id, raw_tei_path=str(raw_tei_path))


def parse_grobid_tei(tei_xml: str, doc_id: str, raw_tei_path: str = "") -> GrobidDocument:
    root = ET.fromstring(tei_xml)
    doc = GrobidDocument(raw_tei_path=raw_tei_path)
    doc.title = first_text(root, ".//tei:teiHeader//tei:titleStmt/tei:title") or first_text(
        root, ".//tei:title[@type='main']"
    )
    doc.authors = parse_authors(root)
    doc.year = parse_year(root)
    doc.doi = first_text(root, ".//tei:idno[@type='DOI']")
    abstract = root.find(".//tei:profileDesc/tei:abstract", NS)
    doc.abstract = normalize_text(" ".join(abstract.itertext())) if abstract is not None else ""
    doc.sections = parse_sections(root, doc_id)
    doc.references = parse_references(root, doc_id)
    return doc


def parse_authors(root: ET.Element) -> List[str]:
    authors: List[str] = []
    for author in root.findall(".//tei:teiHeader//tei:author", NS):
        parts = [normalize_text(" ".join(node.itertext())) for node in author.findall(".//tei:persName/*", NS)]
        name = normalize_text(" ".join(part for part in parts if part))
        if not name:
            name = normalize_text(" ".join(author.itertext()))
        if name and name not in authors:
            authors.append(name)
    return authors


def parse_year(root: ET.Element) -> str:
    for expr in (
        ".//tei:publicationStmt//tei:date",
        ".//tei:sourceDesc//tei:biblStruct//tei:date",
        ".//tei:date",
    ):
        node = root.find(expr, NS)
        if node is None:
            continue
        value = node.attrib.get("when") or node.attrib.get("year") or normalize_text(" ".join(node.itertext()))
        match = re.search(r"\d{4}", value or "")
        if match:
            return match.group(0)
    return ""


def parse_sections(root: ET.Element, doc_id: str) -> List[GrobidSection]:
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []
    sections: List[GrobidSection] = []

    def visit(div: ET.Element, path: List[str], level: int) -> None:
        heading = first_child_text(div, "head") or f"Section {len(sections) + 1}"
        section_path = " > ".join(path + [heading])
        paragraphs: List[GrobidParagraph] = []
        for index, para in enumerate(div.findall("./tei:p", NS), start=1):
            text = normalize_text(" ".join(para.itertext()))
            if not text:
                continue
            citation_ids = []
            for ref in para.findall(".//tei:ref[@type='bibr']", NS):
                target = (ref.attrib.get("target") or "").lstrip("#")
                if target:
                    citation_ids.append(target)
            paragraphs.append(
                GrobidParagraph(
                    paragraph_id=stable_id("grobid_para", doc_id, section_path, index, text),
                    text=text,
                    citation_ids=list(dict.fromkeys(citation_ids)),
                    raw_xml_id=para.attrib.get("{http://www.w3.org/XML/1998/namespace}id", ""),
                )
            )
        sections.append(
            GrobidSection(
                section_id=stable_id("grobid_section", doc_id, section_path),
                section_path=section_path,
                level=level,
                heading=heading,
                paragraphs=paragraphs,
            )
        )
        for child in div.findall("./tei:div", NS):
            visit(child, path + [heading], level + 1)

    for div in body.findall("./tei:div", NS):
        visit(div, [], 1)
    return sections


def parse_references(root: ET.Element, doc_id: str) -> List[ReferenceRecord]:
    records: List[ReferenceRecord] = []
    for index, bibl in enumerate(root.findall(".//tei:text/tei:back//tei:listBibl/tei:biblStruct", NS), start=1):
        label = bibl.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "") or str(index)
        raw = first_text_from(bibl, ".//tei:note[@type='raw_reference']") or normalize_text(" ".join(bibl.itertext()))
        title = first_text_from(bibl, ".//tei:title")
        doi = first_text_from(bibl, ".//tei:idno[@type='DOI']")
        year = ""
        date = bibl.find(".//tei:date", NS)
        if date is not None:
            year = re.search(r"\d{4}", date.attrib.get("when", "") or date.attrib.get("year", "") or "") or ""
            year = year.group(0) if year else ""
        authors = []
        for author in bibl.findall(".//tei:author", NS):
            name = normalize_text(" ".join(author.itertext()))
            if name:
                authors.append(name)
        records.append(
            ReferenceRecord(
                reference_id=stable_id("ref", doc_id, label, raw),
                doc_id=doc_id,
                label=label,
                raw_text=raw,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                metadata={"source": "grobid"},
            )
        )
    return records


def first_text(root: ET.Element, expr: str) -> str:
    node = root.find(expr, NS)
    return normalize_text(" ".join(node.itertext())) if node is not None else ""


def first_text_from(root: ET.Element, expr: str) -> str:
    return first_text(root, expr)


def first_child_text(root: ET.Element, local_name: str) -> str:
    node = root.find(f"./tei:{local_name}", NS)
    return normalize_text(" ".join(node.itertext())) if node is not None else ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
