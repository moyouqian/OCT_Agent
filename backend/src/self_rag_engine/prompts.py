from typing import Dict, List

from .types import Document


CONTEXT_ONLY_SYSTEM_PROMPT = """You are a context-only RAG assistant.
Use only the supplied context.
If the context does not contain the answer, say that the answer cannot be determined from the provided context.
Prefer precise, source-grounded answers over broad background knowledge."""


def format_context(documents: List[Document]) -> str:
    blocks = []
    for index, doc in enumerate(documents, start=1):
        text = doc.get("text") or doc.get("content") or ""
        meta = doc.get("meta") or doc.get("metadata") or {}
        citation_id = meta.get("citation_id")
        if citation_id:
            title = meta.get("title", "unknown")
            section = meta.get("section_path", "unknown")
            page = _format_page(meta.get("page_start"), meta.get("page_end"))
            blocks.append(
                f"[{citation_id}]\n"
                f"Title: {title}\n"
                f"Page: {page or 'unknown'}\n"
                f"Section: {section or 'unknown'}\n"
                f"Content:\n{text.strip()}"
            )
            continue
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "unknown")
        score = doc.get("score")
        score_text = f" score={score}" if score is not None else ""
        blocks.append(f"Context {index}{score_text} source={source} chunk={chunk_index}:\n{text.strip()}")
    return "\n\n".join(blocks)


def hyde_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Write concise hypothetical passages for retrieval."},
        {
            "role": "user",
            "content": (
                "Write a short factual passage that would likely answer this question. "
                "The passage is only used as a search query.\n\n"
                f"Question: {question}"
            ),
        },
    ]


def relevance_messages(question: str, document: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You grade whether a retrieved document is relevant to a question. "
                "Return JSON with keys: binary_score, score, reason."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nDocument:\n{document}"},
    ]


def rewrite_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Rewrite questions for document retrieval. Preserve user intent, "
                "add useful retrieval terms, and do not introduce new facts."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite this question into a better search query. "
                "Return JSON with key rewritten_question.\n\n"
                f"Question: {question}"
            ),
        },
    ]


def generation_messages(
    question: str, documents: List[Document], feedback: str | None = None
) -> List[Dict[str, str]]:
    feedback_block = (
        f"Your previous answer was rejected for the following reason: {feedback}\n"
        "Please revise your answer using only the context below.\n\n"
        if feedback
        else ""
    )
    return [
        {"role": "system", "content": CONTEXT_ONLY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{feedback_block}"
                "Use the following context to answer the question.\n\n"
                f"{format_context(documents)}\n\n"
                f"Question: {question}\n\n"
                "Answer with grounded details and avoid unsupported claims. "
                "When context blocks have citation ids like [S1], cite key claims with those ids."
            ),
        },
    ]


def groundedness_messages(generation: str, documents: List[Document]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You grade whether an answer is fully supported by the provided context. "
                "Return JSON with keys: binary_score, score, reason."
            ),
        },
        {"role": "user", "content": f"Context:\n{format_context(documents)}\n\nAnswer:\n{generation}"},
    ]


def usefulness_messages(question: str, generation: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You grade whether an answer addresses the user's question. "
                "Return JSON with keys: binary_score, score, reason."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{generation}"},
    ]


def _format_page(page_start, page_end) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"
