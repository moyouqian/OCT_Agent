from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import re

from .config import SelfRagConfig
from .cleaning import clean_blocks


@dataclass
class ReferenceRecord:
    reference_id: str
    doc_id: str
    label: str = ""
    raw_text: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    block_id: str
    block_type: str
    text: str
    level: int = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    docling_ref: str = ""
    bbox_list: List[Dict[str, Any]] = field(default_factory=list)
    caption: str = ""
    asset_file_path: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    grobid_ref: str = ""
    section_path: str = ""
    section_level: int = 0
    role: str = "body"
    text_for_display: str = ""
    text_for_embedding: str = ""
    citation_ids: List[str] = field(default_factory=list)
    asset_ids: List[str] = field(default_factory=list)
    noise_reason: str = ""
    source_parser: str = ""


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    source_type: str
    file_type: str
    title: str
    content_hash: str
    parser: str
    blocks: List[Block]
    authors: List[str] = field(default_factory=list)
    year: str = ""
    parser_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[ReferenceRecord] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)


@dataclass
class OcrDecision:
    mode: str
    enabled: bool
    force_full_page: bool
    reason: str = ""
    sampled_pages: int = 0
    extracted_chars: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "force_full_page": self.force_full_page,
            "reason": self.reason,
            "sampled_pages": self.sampled_pages,
            "extracted_chars": self.extracted_chars,
        }


def parse_document(file_path: str, config: SelfRagConfig) -> ParsedDocument:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, config)
    if ext == ".md":
        return parse_markdown_file(path, config)
    if ext == ".txt":
        return parse_plain_text_file(path, config)
    if ext == ".docx":
        return parse_docx_file(path, config)
    raise ValueError(f"Unsupported file type: {ext}")


def parse_pdf_with_docling(file_path: Path, config: SelfRagConfig) -> ParsedDocument:
    return parse_pdf_docling_only(file_path, config)


def parse_pdf(file_path: Path, config: SelfRagConfig) -> ParsedDocument:
    if config.pdf_parser.lower() != "docling":
        raise ValueError("Only the Docling PDF parser is supported for structured PDF ingestion.")
    mode = getattr(config, "pdf_parse_mode", "docling_grobid").strip().lower()
    if mode == "docling":
        return parse_pdf_docling_only(file_path, config)
    if mode == "docling_grobid":
        return parse_pdf_docling_grobid(file_path, config)
    raise ValueError("SELF_RAG_PDF_PARSE_MODE must be docling or docling_grobid.")


def parse_pdf_docling_only(file_path: Path, config: SelfRagConfig) -> ParsedDocument:
    if config.pdf_parser.lower() != "docling":
        raise ValueError("Only the Docling PDF parser is supported for structured PDF ingestion.")

    doc_id, content_hash, doc, markdown, ocr_decision = parse_docling_pdf_payload(file_path, config)
    from .alignment import align_docling_grobid
    from .docling_parser import extract_docling_blocks

    body_blocks, furniture_blocks, docling_report = extract_docling_blocks(doc, fallback_markdown=markdown)
    aligned_blocks, alignment_report = align_docling_grobid(body_blocks, furniture_blocks, None, config)
    blocks, cleaning_report = clean_blocks(aligned_blocks, config)
    title = extract_title(blocks, fallback=file_path.stem)
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        source_type="paper",
        file_type="pdf",
        title=title,
        content_hash=content_hash,
        parser="docling",
        parser_version=docling_version(),
        blocks=blocks,
        metadata={
            "cleaning": cleaning_report,
            "ocr": ocr_decision.as_dict(),
            "docling": docling_report,
            "alignment": alignment_report,
            "grobid": {"enabled": False, "status": "disabled"},
        },
    )


def parse_pdf_docling_grobid(file_path: Path, config: SelfRagConfig) -> ParsedDocument:
    doc_id, content_hash, doc, markdown, ocr_decision = parse_docling_pdf_payload(file_path, config)
    from .alignment import align_docling_grobid
    from .docling_parser import extract_docling_blocks
    from .grobid_parser import parse_pdf_with_grobid

    body_blocks, furniture_blocks, docling_report = extract_docling_blocks(doc, fallback_markdown=markdown)
    grobid_doc = None
    grobid_error = ""
    grobid_status = "disabled"
    parse_warnings: List[str] = []
    if getattr(config, "grobid_enabled", True):
        try:
            grobid_doc = parse_pdf_with_grobid(file_path, doc_id, config)
            grobid_status = "ok"
        except Exception as exc:
            grobid_error = str(exc)
            parse_warnings.append(f"GROBID parsing failed: {grobid_error}")
            grobid_status = "failed"

    aligned_blocks, alignment_report = align_docling_grobid(body_blocks, furniture_blocks, grobid_doc, config)
    blocks, cleaning_report = clean_blocks(aligned_blocks, config)
    fallback_title = extract_title(blocks, fallback=file_path.stem)
    title = getattr(grobid_doc, "title", "") or fallback_title
    references = list(getattr(grobid_doc, "references", []) or [])
    parser = "docling+grobid" if grobid_status == "ok" else "docling"
    grobid_meta = {
        "enabled": bool(getattr(config, "grobid_enabled", True)),
        "status": grobid_status,
        "url": getattr(config, "grobid_url", ""),
        "raw_tei_path": getattr(grobid_doc, "raw_tei_path", "") if grobid_doc else "",
    }
    if grobid_error:
        grobid_meta["error"] = grobid_error
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        source_type="paper",
        file_type="pdf",
        title=title,
        content_hash=content_hash,
        parser=parser,
        parser_version=docling_version(),
        blocks=blocks,
        authors=list(getattr(grobid_doc, "authors", []) or []),
        year=getattr(grobid_doc, "year", "") if grobid_doc else "",
        metadata={
            "cleaning": cleaning_report,
            "ocr": ocr_decision.as_dict(),
            "docling": docling_report,
            "alignment": alignment_report,
            "grobid": grobid_meta,
        },
        references=references,
        parse_warnings=parse_warnings,
    )


def parse_docling_pdf_payload(file_path: Path, config: SelfRagConfig) -> tuple[str, str, Any, str, OcrDecision]:
    content_hash = hash_file(file_path)
    doc_id = stable_id("doc", str(file_path.resolve()), content_hash)
    parsed_dir = Path(config.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    ocr_decision = decide_docling_ocr(file_path, config)
    converter = build_docling_converter(config, ocr_decision=ocr_decision)
    result = converter.convert(str(file_path))
    doc = result.document

    raw_dict = export_docling_dict(doc)
    if raw_dict:
        (parsed_dir / f"{doc_id}.docling.json").write_text(
            json.dumps(raw_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    markdown = export_docling_markdown(doc)
    if markdown:
        (parsed_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    return doc_id, content_hash, doc, markdown, ocr_decision


def build_docling_converter(config: SelfRagConfig, ocr_decision: Optional[OcrDecision] = None):
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("Install the ingestion extra with Docling to parse PDF files.") from exc

    pipeline_options = build_pdf_pipeline_options(config, PdfPipelineOptions, RapidOcrOptions, ocr_decision)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def build_pdf_pipeline_options(
    config: SelfRagConfig,
    options_cls=None,
    ocr_options_cls=None,
    ocr_decision: Optional[OcrDecision] = None,
):
    if options_cls is None or ocr_options_cls is None:
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        except ImportError as exc:
            raise RuntimeError("Install the ingestion extra with Docling to parse PDF files.") from exc
        options_cls = options_cls or PdfPipelineOptions
        ocr_options_cls = ocr_options_cls or RapidOcrOptions

    ocr_decision = ocr_decision or decide_docling_ocr(None, config)

    pipeline_options = options_cls()
    pipeline_options.do_ocr = ocr_decision.enabled
    pipeline_options.images_scale = config.docling_image_scale
    pipeline_options.generate_page_images = config.docling_generate_page_images
    pipeline_options.generate_picture_images = config.docling_generate_picture_images
    pipeline_options.ocr_options = ocr_options_cls(
        lang=["chinese", "english"],
        force_full_page_ocr=ocr_decision.force_full_page,
    )
    return pipeline_options


def decide_docling_ocr(file_path: Optional[Path], config: SelfRagConfig) -> OcrDecision:
    mode = config.docling_enable_ocr.strip().lower()
    if mode not in {"auto", "true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError("SELF_RAG_DOCLING_ENABLE_OCR must be auto, true, or false.")

    if mode in {"false", "0", "no", "off"}:
        return OcrDecision(mode="false", enabled=False, force_full_page=False, reason="disabled by config")
    if mode in {"true", "1", "yes", "on"}:
        return OcrDecision(mode="true", enabled=True, force_full_page=True, reason="forced by config")
    if file_path is None:
        return OcrDecision(mode="auto", enabled=True, force_full_page=False, reason="no file to inspect")

    stats = inspect_pdf_text_layer(
        file_path,
        sample_pages=max(1, config.docling_auto_ocr_sample_pages),
    )
    min_chars = max(0, config.docling_auto_ocr_min_text_chars)
    if stats["error"]:
        return OcrDecision(
            mode="auto",
            enabled=True,
            force_full_page=False,
            reason=f"text layer inspection failed: {stats['error']}",
            sampled_pages=stats["sampled_pages"],
            extracted_chars=stats["text_chars"],
        )
    if stats["text_chars"] >= min_chars:
        return OcrDecision(
            mode="auto",
            enabled=False,
            force_full_page=False,
            reason=f"text layer detected ({stats['text_chars']} chars)",
            sampled_pages=stats["sampled_pages"],
            extracted_chars=stats["text_chars"],
        )
    return OcrDecision(
        mode="auto",
        enabled=True,
        force_full_page=False,
        reason=f"sparse text layer ({stats['text_chars']} chars < {min_chars})",
        sampled_pages=stats["sampled_pages"],
        extracted_chars=stats["text_chars"],
    )


def inspect_pdf_text_layer(file_path: Path, sample_pages: int = 5, reader: Any = None) -> Dict[str, Any]:
    try:
        if reader is None:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        page_indexes = sampled_page_indexes(total_pages, sample_pages)
        text_chars = 0
        for page_index in page_indexes:
            text = reader.pages[page_index].extract_text() or ""
            text_chars += len(normalize_pdf_text_for_ocr_probe(text))
        return {
            "total_pages": total_pages,
            "sampled_pages": len(page_indexes),
            "text_chars": text_chars,
            "error": "",
        }
    except Exception as exc:
        return {
            "total_pages": 0,
            "sampled_pages": 0,
            "text_chars": 0,
            "error": str(exc),
        }


def sampled_page_indexes(total_pages: int, sample_pages: int) -> List[int]:
    if total_pages <= 0 or sample_pages <= 0:
        return []
    if total_pages <= sample_pages:
        return list(range(total_pages))
    if sample_pages == 1:
        return [0]
    indexes = {
        round(i * (total_pages - 1) / (sample_pages - 1))
        for i in range(sample_pages)
    }
    return sorted(indexes)


def normalize_pdf_text_for_ocr_probe(text: str) -> str:
    return re.sub(r"\s+", "", text)


def parse_markdown_file(file_path: Path, config: Optional[SelfRagConfig] = None) -> ParsedDocument:
    config = config or SelfRagConfig()
    text = file_path.read_text(encoding="utf-8").strip()
    content_hash = hash_file(file_path)
    doc_id = stable_id("doc", str(file_path.resolve()), content_hash)
    body, metadata = split_frontmatter(text)
    raw_blocks = markdown_blocks(body)
    blocks, cleaning_report = clean_blocks(raw_blocks, config)
    title = metadata.get("title") or extract_title(blocks, fallback=file_path.stem)
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        source_type="note",
        file_type="md",
        title=title,
        content_hash=content_hash,
        parser="markdown",
        blocks=blocks,
        metadata={**metadata, "cleaning": cleaning_report},
    )


def parse_plain_text_file(file_path: Path, config: Optional[SelfRagConfig] = None) -> ParsedDocument:
    config = config or SelfRagConfig()
    text = file_path.read_text(encoding="utf-8").strip()
    content_hash = hash_file(file_path)
    doc_id = stable_id("doc", str(file_path.resolve()), content_hash)
    raw_blocks = text_blocks(text)
    blocks, cleaning_report = clean_blocks(raw_blocks, config)
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        source_type="text",
        file_type="txt",
        title=file_path.stem,
        content_hash=content_hash,
        parser="plain_text",
        blocks=blocks,
        metadata={"cleaning": cleaning_report},
    )


def parse_docx_file(file_path: Path, config: Optional[SelfRagConfig] = None) -> ParsedDocument:
    config = config or SelfRagConfig()
    from .ingestion import extract_docx_text

    text = extract_docx_text(str(file_path))
    content_hash = hash_file(file_path)
    doc_id = stable_id("doc", str(file_path.resolve()), content_hash)
    raw_blocks = markdown_blocks(text)
    blocks, cleaning_report = clean_blocks(raw_blocks, config)
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        source_type="text",
        file_type="docx",
        title=extract_title(blocks, fallback=file_path.stem),
        content_hash=content_hash,
        parser="docx",
        blocks=blocks,
        metadata={"cleaning": cleaning_report},
    )


def docling_blocks(doc: Any, fallback_markdown: str = "") -> List[Block]:
    blocks: List[Block] = []
    items = []
    for attr in ("texts", "tables", "pictures"):
        values = getattr(doc, attr, None) or []
        items.extend(values)

    if not items and fallback_markdown:
        return markdown_blocks(fallback_markdown)

    for index, item in enumerate(items):
        text = item_text(item).strip()
        label = str(getattr(item, "label", "") or getattr(item, "name", "") or "").lower()
        block_type = docling_block_type(label, item)
        page_start, page_end, bboxes = provenance(item)
        caption = item_caption(item)
        if not text and caption:
            text = caption
        if not text and block_type not in {"figure", "table", "equation"}:
            continue
        blocks.append(
            Block(
                block_id=f"b{index}",
                block_type=block_type,
                text=text,
                level=heading_level(block_type, text),
                page_start=page_start,
                page_end=page_end,
                docling_ref=str(getattr(item, "self_ref", "") or ""),
                bbox_list=bboxes,
                caption=caption,
                raw_payload={"label": label},
            )
        )
    return blocks


def markdown_blocks(text: str) -> List[Block]:
    blocks: List[Block] = []
    lines = text.splitlines()
    buffer: List[str] = []
    block_index = 0
    in_code = False

    def flush_paragraph() -> None:
        nonlocal block_index
        paragraph = "\n".join(buffer).strip()
        buffer.clear()
        if paragraph:
            blocks.append(Block(block_id=f"b{block_index}", block_type="paragraph", text=paragraph))
            block_index += 1

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            buffer.append(line)
            in_code = not in_code
            if not in_code:
                paragraph = "\n".join(buffer).strip()
                buffer.clear()
                blocks.append(Block(block_id=f"b{block_index}", block_type="code", text=paragraph))
                block_index += 1
            continue
        if in_code:
            buffer.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            blocks.append(
                Block(
                    block_id=f"b{block_index}",
                    block_type="title" if level == 1 and not blocks else "section_header",
                    text=heading.group(2).strip(),
                    level=level,
                )
            )
            block_index += 1
            continue
        if not stripped:
            flush_paragraph()
            continue
        buffer.append(line)
    flush_paragraph()
    return blocks


def text_blocks(text: str) -> List[Block]:
    blocks = []
    for index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        paragraph = paragraph.strip()
        if paragraph:
            blocks.append(Block(block_id=f"b{index}", block_type="paragraph", text=paragraph))
    return blocks


def split_frontmatter(text: str) -> tuple[str, Dict[str, Any]]:
    if not text.startswith("---"):
        return text, {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, flags=re.DOTALL)
    if not match:
        return text, {}
    metadata: Dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return match.group(2), metadata


def export_docling_dict(doc: Any) -> Dict[str, Any]:
    for method in ("export_to_dict", "model_dump", "dict"):
        func = getattr(doc, method, None)
        if callable(func):
            try:
                return func()
            except TypeError:
                return func(mode="json")
    return {}


def export_docling_markdown(doc: Any) -> str:
    func = getattr(doc, "export_to_markdown", None)
    if callable(func):
        return func()
    return ""


def item_text(item: Any) -> str:
    for attr in ("text", "orig", "content"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value
    export = getattr(item, "export_to_markdown", None)
    if callable(export):
        try:
            return export()
        except Exception:
            return ""
    return ""


def item_caption(item: Any) -> str:
    captions = getattr(item, "captions", None) or []
    parts = []
    for caption in captions:
        text = item_text(caption).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def docling_block_type(label: str, item: Any) -> str:
    class_name = item.__class__.__name__.lower()
    marker = f"{label} {class_name}"
    if "section_header" in marker or "heading" in marker:
        return "section_header"
    if "title" in marker:
        return "title"
    if "table" in marker:
        return "table"
    if "picture" in marker or "figure" in marker:
        return "figure"
    if "formula" in marker or "equation" in marker:
        return "equation"
    if "list" in marker:
        return "list_item"
    return "paragraph"


def heading_level(block_type: str, text: str) -> int:
    if block_type == "title":
        return 1
    if block_type == "section_header":
        match = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
        if match:
            return min(match.group(1).count(".") + 1, 6)
        return 2
    return 0


def provenance(item: Any) -> tuple[Optional[int], Optional[int], List[Dict[str, Any]]]:
    prov = getattr(item, "prov", None) or getattr(item, "provenance", None) or []
    pages = []
    bboxes = []
    for entry in prov:
        page_no = getattr(entry, "page_no", None)
        if page_no is not None:
            pages.append(int(page_no))
        bbox = getattr(entry, "bbox", None)
        if bbox is not None:
            if hasattr(bbox, "model_dump"):
                bboxes.append(bbox.model_dump(mode="json"))
            elif hasattr(bbox, "dict"):
                bboxes.append(bbox.dict())
            else:
                bboxes.append({"value": str(bbox)})
    if not pages:
        return None, None, bboxes
    return min(pages), max(pages), bboxes


def extract_title(blocks: List[Block], fallback: str) -> str:
    for block in blocks:
        if block.block_type == "title" and block.text.strip():
            return block.text.strip()
    for block in blocks:
        if block.block_type == "section_header" and block.text.strip() and not is_common_section_heading(block.text):
            return block.text.strip()
    for block in blocks:
        if block.block_type == "section_header" and block.level <= 1 and block.text.strip():
            return block.text.strip()
    return fallback


def is_common_section_heading(text: str) -> bool:
    normalized = re.sub(r"^\s*(\d+(\.\d+)*|[ivxlcdm]+|[a-z])\.?\s+", "", text.strip().lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized).strip()
    return normalized in {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "materials and methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgement",
        "acknowledgements",
        "参考文献",
    }


def hash_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "\n".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def docling_version() -> str:
    try:
        import docling

        return getattr(docling, "__version__", "")
    except Exception:
        return ""
