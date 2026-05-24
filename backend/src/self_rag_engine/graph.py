from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from .citations import build_citations
from .config import SelfRagConfig
from .llm import build_chat_backend, parse_json_object
from .prompts import (
    generation_messages,
    groundedness_messages,
    relevance_messages,
    rewrite_messages,
    usefulness_messages,
)
from .retrieval import HybridRetriever
from .types import ChatBackend, Document, Retriever


class SelfRagState(TypedDict, total=False):
    question: str
    original_question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
    retrieval_trace: List[Dict[str, Any]]
    grading_trace: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    attempt_count: int
    retrieval_attempt_count: int
    generation_attempt_count: int
    error: str


def build_self_rag_graph(
    config: Optional[SelfRagConfig] = None,
    chat_backend: Optional[ChatBackend] = None,
    retriever: Optional[Retriever] = None,
):
    """Build the complete LangGraph Self-RAG graph."""
    from langgraph.graph import END, START, StateGraph

    config = config or SelfRagConfig()
    llm = chat_backend or build_chat_backend(config)
    active_retriever = retriever or HybridRetriever(config=config, llm=llm)

    def retrieve(state: SelfRagState) -> SelfRagState:
        question = state["question"]
        retrieval_trace = list(state.get("retrieval_trace", []))
        try:
            result = active_retriever.retrieve(question)
        except Exception as exc:
            return {**state, "documents": [], "error": f"retrieve failed: {exc}"}
        retrieval_trace.append(result.get("trace", {"query": question}))
        documents = result.get("documents", [])
        return {
            **state,
            "documents": documents,
            "citations": build_citations(documents),
            "retrieval_trace": retrieval_trace,
        }

    def grade_documents(state: SelfRagState) -> SelfRagState:
        question = state["question"]
        filtered = []
        grades = []
        for index, document in enumerate(state.get("documents", [])):
            fallback = {
                "binary_score": "yes",
                "score": document.get("score", 0),
                "reason": "Fallback accepted retrieved document because grading failed.",
            }
            try:
                raw = llm.generate(
                    relevance_messages(question, _document_text(document)),
                    temperature=0.0,
                    max_tokens=config.grading_max_tokens,
                    json_mode=True,
                )
                grade = parse_json_object(raw, fallback)
            except Exception as exc:
                grade = dict(fallback)
                grade["reason"] = f"Document grader failed: {exc}"

            grade["document_index"] = index
            grades.append(grade)
            if _as_yes(grade.get("binary_score")):
                filtered.append(document)

        grading_trace = list(state.get("grading_trace", []))
        grading_trace.append({"node": "grade_documents", "grades": grades})
        next_state: SelfRagState = {**state, "documents": filtered, "grading_trace": grading_trace}
        if not filtered and state.get("retrieval_attempt_count", 0) >= config.max_retrieval_attempts:
            next_state["error"] = "No relevant documents found after the maximum retrieval attempts."
        return next_state

    def transform_query(state: SelfRagState) -> SelfRagState:
        attempt_count = state.get("attempt_count", 0) + 1
        retrieval_attempt_count = state.get("retrieval_attempt_count", 0) + 1
        fallback_question = state.get("rewritten_question") or state["question"]
        try:
            raw = llm.generate(
                rewrite_messages(state["question"]),
                temperature=0.0,
                max_tokens=config.grading_max_tokens,
                json_mode=True,
            )
            parsed = parse_json_object(raw, {"rewritten_question": fallback_question})
            rewritten = parsed.get("rewritten_question") or fallback_question
        except Exception as exc:
            return {
                **state,
                "attempt_count": attempt_count,
                "retrieval_attempt_count": retrieval_attempt_count,
                "rewritten_question": fallback_question,
                "question": fallback_question,
                "error": f"query rewrite failed: {exc}",
            }

        return {
            **state,
            "attempt_count": attempt_count,
            "retrieval_attempt_count": retrieval_attempt_count,
            "rewritten_question": rewritten,
            "question": rewritten,
            "generation_attempt_count": 0,
        }

    def generate(state: SelfRagState) -> SelfRagState:
        attempt_count = state.get("attempt_count", 0) + 1
        generation_attempt_count = state.get("generation_attempt_count", 0) + 1
        grading_trace = state.get("grading_trace", [])
        feedback = None
        if grading_trace:
            last = grading_trace[-1]
            groundedness = last.get("groundedness", {})
            if not _as_yes(groundedness.get("binary_score")):
                feedback = groundedness.get("reason")
        try:
            generation = llm.generate(
                generation_messages(state["question"], state.get("documents", []), feedback=feedback),
                temperature=0.2,
                max_tokens=config.generation_max_tokens,
            )
        except Exception as exc:
            return {
                **state,
                "attempt_count": attempt_count,
                "generation_attempt_count": generation_attempt_count,
                "error": f"generation failed: {exc}",
            }
        return {
            **state,
            "attempt_count": attempt_count,
            "generation_attempt_count": generation_attempt_count,
            "generation": generation,
        }

    def grade_generation(state: SelfRagState) -> SelfRagState:
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        question = state["question"]

        grounded = _grade_with_fallback(
            llm,
            groundedness_messages(generation, documents),
            {"binary_score": "no", "score": 0, "reason": "Groundedness grading failed."},
            config.grading_max_tokens,
        )
        if _as_yes(grounded.get("binary_score")):
            useful = _grade_with_fallback(
                llm,
                usefulness_messages(question, generation),
                {"binary_score": "no", "score": 0, "reason": "Usefulness grading failed."},
                config.grading_max_tokens,
            )
        else:
            useful = {
                "binary_score": "no",
                "score": 0,
                "reason": "Skipped usefulness check because answer was not grounded.",
            }

        error = None
        if not _as_yes(grounded.get("binary_score")):
            if state.get("generation_attempt_count", 0) >= config.max_generation_attempts:
                error = "Generation was not grounded after the maximum generation attempts."
        elif not _as_yes(useful.get("binary_score")):
            if state.get("retrieval_attempt_count", 0) >= config.max_retrieval_attempts:
                error = "Generation was grounded but did not answer the question after the maximum retrieval attempts."

        grading_trace = list(state.get("grading_trace", []))
        grading_trace.append(
            {
                "node": "grade_generation_v_documents_and_question",
                "groundedness": grounded,
                "usefulness": useful,
            }
        )
        next_state: SelfRagState = {**state, "grading_trace": grading_trace}
        if error:
            next_state["error"] = error
        return next_state

    def decide_to_generate(state: SelfRagState) -> str:
        if state.get("error"):
            return "end"
        if state.get("documents"):
            return "generate"
        if state.get("retrieval_attempt_count", 0) >= config.max_retrieval_attempts:
            return "end"
        return "transform_query"

    def decide_after_generation(state: SelfRagState) -> str:
        if state.get("error"):
            return "end"
        latest = (state.get("grading_trace") or [{}])[-1]
        grounded = latest.get("groundedness", {})
        useful = latest.get("usefulness", {})
        if _as_yes(grounded.get("binary_score")) and _as_yes(useful.get("binary_score")):
            return "end"
        if not _as_yes(grounded.get("binary_score")):
            if state.get("generation_attempt_count", 0) < config.max_generation_attempts:
                return "generate"
            return "end"
        if state.get("retrieval_attempt_count", 0) < config.max_retrieval_attempts:
            return "transform_query"
        return "end"

    workflow = StateGraph(SelfRagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation_v_documents_and_question", grade_generation)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate", "end": END},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "grade_generation_v_documents_and_question")
    workflow.add_conditional_edges(
        "grade_generation_v_documents_and_question",
        decide_after_generation,
        {"generate": "generate", "transform_query": "transform_query", "end": END},
    )
    return workflow.compile()


def run_self_rag(question: str, config: Optional[SelfRagConfig] = None) -> SelfRagState:
    graph = build_self_rag_graph(config=config)
    return graph.invoke(initial_state(question))


def initial_state(question: str) -> SelfRagState:
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "retrieval_trace": [],
        "grading_trace": [],
        "citations": [],
        "attempt_count": 0,
        "retrieval_attempt_count": 0,
        "generation_attempt_count": 0,
    }


def _grade_with_fallback(llm: ChatBackend, messages: List[Dict[str, str]], fallback: Dict[str, Any], max_tokens: int):
    try:
        raw = llm.generate(messages, temperature=0.0, max_tokens=max_tokens, json_mode=True)
        return parse_json_object(raw, fallback)
    except Exception as exc:
        grade = dict(fallback)
        grade["reason"] = f"{grade['reason']} Error: {exc}"
        return grade


def _as_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "relevant", "grounded", "useful"}


def _document_text(document: Document) -> str:
    return document.get("text") or document.get("content") or ""
