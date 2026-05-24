import os
from dataclasses import dataclass, field
from pathlib import Path


SELF_RAG_PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SELF_RAG_PACKAGE_DIR.parents[1]
DEFAULT_SELF_RAG_DATA_DIR = BACKEND_DIR / "data" / "self_rag"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _data_path(*parts: str) -> str:
    root = Path(os.getenv("SELF_RAG_DATA_DIR", str(DEFAULT_SELF_RAG_DATA_DIR)))
    return str(root.joinpath(*parts))


def _env_or_data_path(name: str, *default_parts: str) -> str:
    raw = os.getenv(name)
    if raw is not None:
        path = Path(raw)
        return str(path if path.is_absolute() else Path(os.getenv("SELF_RAG_DATA_DIR", str(DEFAULT_SELF_RAG_DATA_DIR))) / path)
    return _data_path(*default_parts)


def _env_or_existing_path(name: str, default_path: str) -> str:
    raw = os.getenv(name)
    if raw is not None:
        path = Path(raw)
        return str(path if path.is_absolute() else Path(os.getenv("SELF_RAG_DATA_DIR", str(DEFAULT_SELF_RAG_DATA_DIR))) / path)
    return default_path if os.path.exists(default_path) else ""


@dataclass
class SelfRagConfig:
    """Configuration for the independent Self-RAG graph."""

    chat_backend: str = field(default_factory=lambda: os.getenv("SELF_RAG_CHAT_BACKEND", "auto"))
    embedding_backend: str = field(default_factory=lambda: os.getenv("SELF_RAG_EMBEDDING_BACKEND", "auto"))
    openai_base_url: str = field(
        default_factory=lambda: (
            os.getenv("SELF_RAG_OPENAI_BASE_URL")
            or os.getenv("SILICONFLOW_API_BASE")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL", "")
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: (
            os.getenv("SELF_RAG_OPENAI_API_KEY")
            or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
    )
    chat_model: str = field(
        default_factory=lambda: (
            os.getenv("SELF_RAG_CHAT_MODEL")
            or os.getenv("SILICONFLOW_API_MODEL")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("MODEL", "")
        )
    )
    embedding_model: str = field(default_factory=lambda: os.getenv("SELF_RAG_EMBEDDING_MODEL", ""))
    local_model_path: str = field(
        default_factory=lambda: os.getenv(
            "SELF_RAG_LOCAL_MODEL_PATH",
            "./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        )
    )
    chroma_dir: str = field(default_factory=lambda: _env_or_data_path("SELF_RAG_CHROMA_DIR", "chroma_store"))
    collection_name: str = field(default_factory=lambda: os.getenv("SELF_RAG_COLLECTION", "documents"))
    bm25_dir: str = field(default_factory=lambda: _env_or_data_path("SELF_RAG_BM25_DIR", "bm25_store"))
    bm25_tokenizer: str = field(default_factory=lambda: os.getenv("SELF_RAG_BM25_TOKENIZER", "jieba_search"))
    bm25_domain_terms_path: str = field(
        default_factory=lambda: _env_or_data_path("SELF_RAG_BM25_DOMAIN_TERMS_PATH", "config", "domain_terms.txt")
    )
    bm25_enable_cjk_bigrams: bool = field(default_factory=lambda: _env_bool("SELF_RAG_BM25_ENABLE_CJK_BIGRAMS", True))
    sqlite_path: str = field(default_factory=lambda: _env_or_data_path("SELF_RAG_SQLITE_PATH", "rag_store", "rag.sqlite3"))
    parsed_dir: str = field(default_factory=lambda: _env_or_data_path("SELF_RAG_PARSED_DIR", "rag_store", "parsed"))
    assets_dir: str = field(default_factory=lambda: _env_or_data_path("SELF_RAG_ASSETS_DIR", "rag_store", "assets"))
    pdf_parser: str = field(default_factory=lambda: os.getenv("SELF_RAG_PDF_PARSER", "docling"))
    pdf_parse_mode: str = field(default_factory=lambda: os.getenv("SELF_RAG_PDF_PARSE_MODE", "docling_grobid"))
    grobid_enabled: bool = field(default_factory=lambda: _env_bool("SELF_RAG_GROBID_ENABLED", True))
    grobid_url: str = field(default_factory=lambda: os.getenv("SELF_RAG_GROBID_URL", "http://localhost:8070"))
    grobid_timeout: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_GROBID_TIMEOUT", "60")))
    grobid_include_raw_citations: bool = field(
        default_factory=lambda: _env_bool("SELF_RAG_GROBID_INCLUDE_RAW_CITATIONS", True)
    )
    grobid_use_coordinates: bool = field(
        default_factory=lambda: _env_bool("SELF_RAG_GROBID_USE_COORDINATES", True)
    )
    docling_generate_page_images: bool = field(
        default_factory=lambda: _env_bool("SELF_RAG_DOCLING_GENERATE_PAGE_IMAGES", False)
    )
    docling_generate_picture_images: bool = field(
        default_factory=lambda: _env_bool("SELF_RAG_DOCLING_GENERATE_PICTURE_IMAGES", False)
    )
    docling_image_scale: float = field(default_factory=lambda: float(os.getenv("SELF_RAG_DOCLING_IMAGE_SCALE", "2.0")))
    docling_enable_ocr: str = field(default_factory=lambda: os.getenv("SELF_RAG_DOCLING_ENABLE_OCR", "auto"))
    docling_auto_ocr_sample_pages: int = field(
        default_factory=lambda: int(os.getenv("SELF_RAG_DOCLING_AUTO_OCR_SAMPLE_PAGES", "5"))
    )
    docling_auto_ocr_min_text_chars: int = field(
        default_factory=lambda: int(os.getenv("SELF_RAG_DOCLING_AUTO_OCR_MIN_TEXT_CHARS", "300"))
    )
    use_hyde: bool = field(default_factory=lambda: _env_bool("SELF_RAG_USE_HYDE", True))
    use_rerank: bool = field(default_factory=lambda: _env_bool("SELF_RAG_USE_RERANK", True))
    rerank_backend: str = field(default_factory=lambda: os.getenv("SELF_RAG_RERANK_BACKEND", "local"))
    rerank_model: str = field(
        default_factory=lambda: os.getenv("SELF_RAG_RERANK_MODEL", "./models/ms-marco-MiniLM-L-6-v2")
    )
    rerank_instruction: str = field(
        default_factory=lambda: os.getenv(
            "SELF_RAG_RERANK_INSTRUCTION",
            "Given a search query, rerank the retrieved passages by relevance for answering the query.",
        )
    )
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_RETRIEVAL_TOP_K", "50")))
    final_k: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_FINAL_K", "10")))
    context_parent_top_k: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_CONTEXT_PARENT_TOP_K", "6")))
    expand_to_parent: bool = field(default_factory=lambda: _env_bool("SELF_RAG_EXPAND_TO_PARENT", True))
    max_retrieval_attempts: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_MAX_RETRIEVAL_ATTEMPTS", "2")))
    max_generation_attempts: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_MAX_GENERATION_ATTEMPTS", "2")))
    generation_max_tokens: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_GENERATION_MAX_TOKENS", "1200")))
    grading_max_tokens: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_GRADING_MAX_TOKENS", "400")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_CHUNK_OVERLAP", "100")))
    parent_chunk_max_tokens: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_PARENT_CHUNK_MAX_TOKENS", "1800")))
    parent_chunk_min_tokens: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_PARENT_CHUNK_MIN_TOKENS", "300")))
    child_chunk_size: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_CHILD_CHUNK_SIZE", "500")))
    child_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_CHILD_CHUNK_OVERLAP", "80")))
    enable_document_cleaning: bool = field(default_factory=lambda: _env_bool("SELF_RAG_ENABLE_DOCUMENT_CLEANING", True))
    clean_exclude_references: bool = field(default_factory=lambda: _env_bool("SELF_RAG_CLEAN_EXCLUDE_REFERENCES", True))
    clean_repeated_headers: bool = field(default_factory=lambda: _env_bool("SELF_RAG_CLEAN_REPEATED_HEADERS", True))
    repeated_header_min_pages: int = field(default_factory=lambda: int(os.getenv("SELF_RAG_REPEATED_HEADER_MIN_PAGES", "2")))
    mojibake_extra_chars: str = field(default_factory=lambda: os.getenv("SELF_RAG_MOJIBAKE_EXTRA_CHARS", ""))
    citation_style: str = field(default_factory=lambda: os.getenv("SELF_RAG_CITATION_STYLE", "coarse"))
    show_related_assets: bool = field(default_factory=lambda: _env_bool("SELF_RAG_SHOW_RELATED_ASSETS", True))
    send_assets_to_llm: bool = field(default_factory=lambda: _env_bool("SELF_RAG_SEND_ASSETS_TO_LLM", False))

    def cloud_chat_configured(self) -> bool:
        return bool(self.openai_api_key and self.chat_model)

    def cloud_embedding_configured(self) -> bool:
        return bool(self.openai_api_key and self.embedding_model)
