"""Deep research LangGraph subgraph."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph

from agent.config import get_llm
from agent.prompts import (
    FINAL_REPORT_PROMPT,
    RESEARCH_BRIEF_PROMPT,
    RESEARCH_CLARIFY_PROMPT,
    RESEARCH_COMPRESS_PROMPT,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_QUERY_PROMPT,
)
from agent.research.schemas import ClarifyWithUser, ResearchPlan, ResearchQuestion, SearchQueries
from agent.research.sources import search_all_sources
from agent.schemas import OctGraphState
from agent.utils.structured import invoke_structured_json_schema


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _latest_user_text(state: OctGraphState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _invoke_text(prompt: str) -> str:
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return _content_to_text(response.content).strip()


def _parse_items(text: str, *, max_items: int) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", line).strip()
        cleaned = cleaned.strip("\"'` ")
        if cleaned and not cleaned.startswith(("以下", "查询", "子问题")):
            items.append(cleaned)
        if len(items) >= max_items:
            break
    if items:
        return items

    sentences = [part.strip() for part in re.split(r"[。；;\n]+", text) if part.strip()]
    return sentences[:max_items]


def _needs_research_clarification(text: str) -> bool:
    stripped = re.sub(r"^(?:deep\s*research|deep_research)[:：]?", "", text.strip(), flags=re.I).strip()
    lowered = stripped.lower()
    vague_requests = {
        "",
        "oct",
        "做个研究",
        "做研究",
        "研究",
        "调研",
        "文献",
        "检索",
        "综述",
        "deep research",
        "deep_research",
    }
    return lowered in vague_requests or len(stripped) <= 3


def _clarification_question(text: str) -> str:
    topic = text.strip() or "这个主题"
    return (
        f"您希望我围绕“{topic}”具体研究哪个方向？"
        "例如：技术原理、最新临床应用进展、市场趋势、特定疾病诊断案例，或它们的组合。"
    )


def _source_material(results: list[Any]) -> str:
    if not results:
        return "没有检索到可用资料。"
    parts = []
    for index, result in enumerate(results, start=1):
        parts.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Provider: {result.source}",
                    f"Title: {result.title}",
                    f"URL: {result.url}",
                    f"Content: {result.content}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _pending_research_state(message: str) -> OctGraphState:
    """构造"仍需用户补充信息"的状态：发出一条 AI 消息并保持 research_pending。"""
    return {
        "messages": [AIMessage(content=message)],
        "research_pending": True,
        "research_brief": "",
        "research_topics": [],
        "research_notes": [],
        "final_report": "",
        "requested_sub_agent": None,
    }


def _decide_clarification(llm: Any, buffer: str, latest_text: str) -> ClarifyWithUser:
    """判定研究请求是否需要先向用户澄清。"""
    if _needs_research_clarification(latest_text):
        return ClarifyWithUser(
            need_clarification=True,
            question=_clarification_question(latest_text),
            verification="",
        )
    clarify_messages = [
        HumanMessage(content=RESEARCH_CLARIFY_PROMPT.format(date=_today(), messages=buffer))
    ]
    return invoke_structured_json_schema(
        llm,
        ClarifyWithUser,
        clarify_messages,
        fallback_fn=lambda _: ClarifyWithUser(
            need_clarification=False,
            question="",
            verification="研究目标已足够清晰，将开始 deep research。",
        ),
    )


def _write_brief(llm: Any, buffer: str, latest_text: str) -> ResearchQuestion:
    """根据对话历史生成结构化研究 brief。"""
    brief_prompt = RESEARCH_BRIEF_PROMPT.format(date=_today(), messages=buffer)
    return invoke_structured_json_schema(
        llm,
        ResearchQuestion,
        [HumanMessage(content=brief_prompt)],
        fallback_fn=lambda _: ResearchQuestion(
            research_brief=_invoke_text(brief_prompt) or latest_text or "无法生成研究 brief。"
        ),
    )


def clarify_or_write_brief(state: OctGraphState) -> OctGraphState:
    """Clarify ambiguous research requests or produce a research brief."""

    buffer = get_buffer_string(state.get("messages", []))
    llm = get_llm()
    latest_text = _latest_user_text(state)

    if not state.get("research_pending"):
        decision = _decide_clarification(llm, buffer, latest_text)
        if decision.need_clarification:
            return _pending_research_state(decision.question)

    brief = _write_brief(llm, buffer, latest_text)
    if not brief.research_brief.strip():
        return _pending_research_state(
            "无法生成有效的 deep research brief，请补充更明确的研究主题。"
        )
    return {
        "research_brief": brief.research_brief,
        "research_topics": [],
        "research_notes": [],
        "final_report": "",
        "research_pending": False,
        "requested_sub_agent": None,
    }


def _fallback_research_plan(brief: str) -> ResearchPlan:
    try:
        plan_text = _invoke_text(RESEARCH_PLAN_PROMPT.format(research_brief=brief))
        topics = _parse_items(plan_text, max_items=4) or [brief]
    except Exception:
        topics = [brief]
    return ResearchPlan(topics=topics[:4])


def _fallback_search_queries(topic: str) -> SearchQueries:
    try:
        query_text = _invoke_text(RESEARCH_QUERY_PROMPT.format(topic=topic))
        queries = _parse_items(query_text, max_items=3) or [topic]
    except Exception:
        queries = [topic]
    return SearchQueries(queries=queries[:3])


def _safe_compress_note(topic: str, material: str) -> str:
    try:
        compressed = get_llm().invoke(
            [
                HumanMessage(
                    content=RESEARCH_COMPRESS_PROMPT.format(
                        topic=topic,
                        source_material=material,
                    )
                )
            ]
        )
        note = _content_to_text(compressed.content).strip()
    except Exception as exc:
        note = f"研究子问题“{topic}”的资料压缩失败：{exc}\n\n原始资料：\n{material}"
    return note or f"研究子问题“{topic}”没有生成有效笔记。\n\n原始资料：\n{material}"


def _safe_final_report(brief: str, notes: str) -> str:
    try:
        report = get_llm().invoke(
            [
                HumanMessage(
                    content=FINAL_REPORT_PROMPT.format(
                        date=_today(),
                        research_brief=brief,
                        notes=notes or "没有可用研究笔记。",
                    )
                )
            ]
        )
        content = _content_to_text(report.content).strip()
    except Exception as exc:
        content = f"## Deep Research 生成失败\n\n最终报告生成失败：{exc}\n\n## 已获得研究笔记\n\n{notes}"
    return content or f"## Deep Research 研究笔记\n\n{notes or '模型未返回报告内容，且没有可用研究笔记。'}"


def route_after_scope(state: OctGraphState) -> Literal["plan_research", "__end__"]:
    return "plan_research" if state.get("research_brief") else END


def plan_research(state: OctGraphState) -> OctGraphState:
    """Create parallel research topics from the brief."""

    brief = state.get("research_brief") or _latest_user_text(state)
    prompt = RESEARCH_PLAN_PROMPT.format(research_brief=brief)
    plan = invoke_structured_json_schema(
        get_llm(),
        ResearchPlan,
        [HumanMessage(content=prompt)],
        fallback_fn=lambda _: _fallback_research_plan(brief),
    )
    return {"research_topics": plan.topics}


def conduct_research(state: OctGraphState) -> OctGraphState:
    """Search sources for each topic and compress findings."""

    notes: list[str] = []
    topics = state.get("research_topics") or [state.get("research_brief", _latest_user_text(state))]

    for topic in topics:
        prompt = RESEARCH_QUERY_PROMPT.format(topic=topic)
        query_result = invoke_structured_json_schema(
            get_llm(),
            SearchQueries,
            [HumanMessage(content=prompt)],
            fallback_fn=lambda _, research_topic=topic: _fallback_search_queries(research_topic),
        )
        source_results = search_all_sources(query_result.queries, max_results=3)
        material = _source_material(source_results)
        notes.append(_safe_compress_note(topic, material))

    return {"research_notes": notes}


def final_report_generation(state: OctGraphState) -> OctGraphState:
    """Write the final deep research report."""

    brief = state.get("research_brief", "")
    notes = "\n\n".join(state.get("research_notes", []))
    content = _safe_final_report(brief, notes)
    return {
        "final_report": content,
        "messages": [AIMessage(content=content)],
    }


builder = StateGraph(OctGraphState)
builder.add_node("clarify_or_write_brief", clarify_or_write_brief)
builder.add_node("plan_research", plan_research)
builder.add_node("conduct_research", conduct_research)
builder.add_node("final_report_generation", final_report_generation)
builder.add_edge(START, "clarify_or_write_brief")
builder.add_conditional_edges(
    "clarify_or_write_brief",
    route_after_scope,
    {"plan_research": "plan_research", END: END},
)
builder.add_edge("plan_research", "conduct_research")
builder.add_edge("conduct_research", "final_report_generation")
builder.add_edge("final_report_generation", END)

deep_research = builder.compile(name="deep-research")
