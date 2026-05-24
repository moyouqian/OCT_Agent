"""OCT Agent 的 LangGraph 编排。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import get_llm
from agent.prompts import build_chat_prompt, build_strain_prompt, build_summarize_prompt
from agent.research import deep_research
from agent.schemas import OctGraphState
from agent.services.memory import forget, memory_summary, remember
from agent.services.models import DEFAULT_BANDWIDTH, DEFAULT_REFRACTIVE_INDEX, DEFAULT_WAVELENGTH
from agent.services.storage import make_run_dir
from agent.self_rag import decide_retrieval, run_knowledge_query, self_rag_message
from agent.tools import TOOLS
from agent.utils.messages import latest_user_text


_SUMMARIZE_THRESHOLD = 20   # 超过此消息数触发摘要
_KEEP_RECENT = 6            # LLM 调用时保留最近 N 条消息


def _default_strain_settings() -> dict[str, Any]:
    return {
        "vector": False,
        "cnn": False,
        "bnn": False,
        "Nx": 25,
        "Nz": 25,
        "g": 1,
        "MC_test": 50,
    }


def _default_physical_params() -> dict[str, float]:
    return {
        "wavelength": DEFAULT_WAVELENGTH,
        "bandwidth": DEFAULT_BANDWIDTH,
        "refractive_index": DEFAULT_REFRACTIVE_INDEX,
    }


def _latest_user_text(state: OctGraphState) -> str:
    return latest_user_text(state.get("messages", []))


def summarize_conversation(state: OctGraphState) -> OctGraphState:
    messages = state.get("messages", [])
    if len(messages) <= _SUMMARIZE_THRESHOLD:
        return {}

    last_count = state.get("summary_message_count", 0)
    new_compressible = len(messages) - _KEEP_RECENT
    if new_compressible <= last_count:
        return {}

    to_summarize = messages[last_count:new_compressible]
    existing_summary = state.get("conversation_summary", "")
    prompt = build_summarize_prompt(get_buffer_string(to_summarize), existing_summary)
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        new_summary = str(response.content).strip()
    except Exception:
        return {}

    return {
        "conversation_summary": new_summary,
        "summary_message_count": new_compressible,
    }


def init_run_context(state: OctGraphState) -> OctGraphState:
    return {
        "run_dir": make_run_dir("strain_estimation"),
        "result_refs": [],
        "file_ids": state.get("file_ids", []),
        "strain_settings": {**_default_strain_settings(), **state.get("strain_settings", {})},
        "physical_params": {**_default_physical_params(), **state.get("physical_params", {})},
        "visualization_enabled": state.get("visualization_enabled", True),
        "show_thinking": state.get("show_thinking", False),
    }


def strain_assistant(state: OctGraphState) -> OctGraphState:
    run_dir = state["run_dir"]
    result_refs = state.get("result_refs", [])
    file_ids = state.get("file_ids", [])
    settings = {**_default_strain_settings(), **state.get("strain_settings", {})}
    physical = {**_default_physical_params(), **state.get("physical_params", {})}
    selected_methods = [
        name
        for enabled, name in [
            (settings.get("vector"), f"矢量法 Nx={settings['Nx']} Nz={settings['Nz']} g={settings['g']}"),
            (settings.get("cnn"), "CNN"),
            (settings.get("bnn"), f"BNN MC_test={settings['MC_test']}"),
        ]
        if enabled
    ]
    summary = "当前还没有已完成的计算结果。"
    if result_refs:
        summary = "；".join(
            f"{Path(ref.get('file_path', '')).name}: {ref.get('result_key')}"
            for ref in result_refs
        )

    sys_msg = SystemMessage(
        content=build_strain_prompt(
            run_dir=run_dir,
            file_ids=file_ids,
            selected_methods=selected_methods,
            physical=physical,
            summary=summary,
            conversation_summary=state.get("conversation_summary", ""),
        )
    )
    messages = state.get("messages", [])
    recent = messages[-_KEEP_RECENT:] if len(messages) > _KEEP_RECENT else messages
    response = get_llm().bind_tools(TOOLS).invoke([sys_msg] + recent)
    return {"messages": [response]}


def collect_result_refs(state: OctGraphState) -> OctGraphState:
    # 从末尾向前收集连续的 tool 消息，覆盖一次并行调用产生的全部结果
    new_refs = []
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") != "tool":
            break
        try:
            data = json.loads(msg.content)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("status") == "success" and "ref" in data:
            new_refs.append(data["ref"])
    if not new_refs:
        return {}
    old_ids = {ref.get("result_id") for ref in state.get("result_refs", [])}
    new_refs = [ref for ref in new_refs if ref.get("result_id") not in old_ids]
    if not new_refs:
        return {}
    return {"result_refs": new_refs}


def visualize_node(state: OctGraphState) -> OctGraphState:
    result_refs = state.get("result_refs", [])
    if not result_refs:
        return {"messages": [AIMessage(content="没有可视化结果。")]}

    names = "；".join(ref.get("result_key", "结果") for ref in result_refs)
    return {
        "messages": [
            AIMessage(
                content=(
                    f"已生成 {len(result_refs)} 个结果：{names}。"
                    "右侧结果面板可查看交互热力图并下载 .mat 文件。"
                )
            )
        ]
    }


def route_after_assistant(state: OctGraphState) -> Literal["tools", "visualize", "__end__"]:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    if not state.get("visualization_enabled", True):
        return END
    for msg in reversed(state.get("messages", [])):
        msg_type = getattr(msg, "type", "")
        if msg_type == "human":
            return END
        if msg_type == "tool":
            return "visualize"
    return END


strain_builder = StateGraph(OctGraphState)
strain_builder.add_node("init_run_context", init_run_context)
strain_builder.add_node("strain_assistant", strain_assistant)
strain_builder.add_node("tools", ToolNode(TOOLS))
strain_builder.add_node("collect_result_refs", collect_result_refs)
strain_builder.add_node("visualize", visualize_node)
strain_builder.add_edge(START, "init_run_context")
strain_builder.add_edge("init_run_context", "strain_assistant")
strain_builder.add_conditional_edges(
    "strain_assistant",
    route_after_assistant,
    {"tools": "tools", "visualize": "visualize", END: END},
)
strain_builder.add_edge("tools", "collect_result_refs")
strain_builder.add_edge("collect_result_refs", "strain_assistant")
strain_builder.add_edge("visualize", END)
strain_estimation = strain_builder.compile(name="strain-estimation")


# 关键词路由表。每个子图一份关键词集合，去重后集中维护。
# 匹配顺序由 ROUTE_PRIORITY 决定：self_rag 在前，使"本地知识库 ... phase"
# 这类同时含多类关键词的请求优先归入知识库检索而非应变计算。
ROUTE_KEYWORDS: dict[str, list[str]] = {
    "self_rag": [
        "self_rag",
        "self-rag",
        "rag",
        "知识库",
        "知識庫",
        "本地检索",
        "本地檢索",
        "本地知识",
        "本地知識",
        "已索引",
        "论文库",
        "論文庫",
    ],
    "deep_research": [
        "deep research",
        "deep_research",
        "联网",
        "文献检索",
        "文献",
        "检索",
        "综述",
        "调研",
        "研究报告",
        "引用来源",
        "引用",
        "来源",
        "论文",
        "最新",
    ],
    "strain_estimation": [
        "应变",
        "strain",
        "cnn",
        "bnn",
        "矢量",
        "vector",
        ".mat",
        "phase",
        "热力图",
    ],
}

# 关键词匹配优先级。顺序敏感：靠前的子图先匹配。
ROUTE_PRIORITY: tuple[str, ...] = ("self_rag", "strain_estimation", "deep_research")

# 知识问答信号。命中说明用户想「了解」而非「计算」，
# 用于在方法名（bnn/cnn/...）触发应变路由前拦截，改走知识库。
KNOWLEDGE_QUESTION_KEYWORDS: tuple[str, ...] = (
    "介绍",
    "简介",
    "是什么",
    "什么是",
    "什么叫",
    "解释",
    "讲讲",
    "讲解",
    "说说",
    "原理",
    "区别",
    "差异",
    "对比",
    "比较",
    "定义",
    "概念",
    "了解",
    "explain",
    "what is",
    "what are",
    "what's",
    "introduce",
    "definition",
    "define",
    "difference",
)

# 应变计算的动作意图词。只有出现明确的「做计算」信号（或已上传文件），
# 才把含方法名的请求判定为应变计算，避免「介绍BNN」被误判。
STRAIN_COMPUTE_KEYWORDS: tuple[str, ...] = (
    "计算",
    "估计",
    "估算",
    "运算",
    "重建",
    "重构",
    "运行",
    "跑",
    "推理",
    "预测",
    "compute",
    "calculate",
    "run",
    "estimate",
    "estimation",
    "predict",
    "inference",
    "process",
    "reconstruct",
)


def _is_knowledge_question(lowered: str) -> bool:
    return any(token in lowered for token in KNOWLEDGE_QUESTION_KEYWORDS)


def _has_strain_compute_intent(lowered: str) -> bool:
    return any(token in lowered for token in STRAIN_COMPUTE_KEYWORDS)


def supervisor(state: OctGraphState) -> OctGraphState:
    """混合分层路由：显式请求 > 状态标志 > 强信号 > 关键词 > 兜底。

    优先级链（自上而下，命中即返回）：
      1. 显式请求 ``requested_sub_agent``（如 UI 按钮指定的子图）。
      2. 状态标志 ``research_pending``（深度研究多轮澄清进行中）。
      3. 强信号：已选定应变方法且已上传文件 -> strain；记忆命令 -> chat。
      4. 关键词快速匹配（见 ``infer_route_from_text``）。
      5. 兜底 -> self_rag（本地知识库检索）。
    当前不引入 LLM 路由层，关键词兜底足以覆盖科研原型场景。
    """
    requested = state.get("requested_sub_agent")
    if requested:
        # 消费后立即清除：requested_sub_agent 是「本轮一次性」的显式指令
        # （如 UI 按钮）。若不清除，在按 thread 持久化 state 的场景下它会
        # 跨轮次残留，把后续所有输入锁死在同一子图。deep_research 的多轮
        # 澄清改由 research_pending 续接，不依赖该字段。
        return {"sub_agent": requested, "requested_sub_agent": None}
    if state.get("research_pending"):
        return {"sub_agent": "deep_research"}

    settings = state.get("strain_settings", {})
    selected_method = any(settings.get(key) for key in ("vector", "cnn", "bnn"))
    if selected_method and state.get("file_ids"):
        return {"sub_agent": "strain_estimation"}

    text = _latest_user_text(state)
    if _is_memory_command(text):
        return {"sub_agent": "chat"}

    inferred = infer_route_from_text(text, has_files=bool(state.get("file_ids")))
    if inferred is not None:
        return {"sub_agent": inferred}
    return {"sub_agent": "self_rag"}


def infer_route_from_text(
    text: str, has_files: bool = False
) -> Literal["strain_estimation", "deep_research", "self_rag"] | None:
    """关键词路由，并对应变子图做意图门控。

    方法名（bnn/cnn/strain/...）既出现在计算请求里，也出现在知识问答里，
    因此命中应变关键词后还需确认是「计算意图」而非「了解意图」：
      - 明确的知识问句（介绍/是什么/解释...）且无计算动作词 -> 跳过应变，归知识库；
      - 出现计算动作词，或本轮已上传文件 -> 应变计算；
      - 仅裸方法名、既无计算意图也无附件 -> 跳过应变，交由兜底 self_rag。
    """
    lowered = text.lower()
    knowledge = _is_knowledge_question(lowered)
    compute = _has_strain_compute_intent(lowered)
    for route in ROUTE_PRIORITY:
        if not any(token in lowered for token in ROUTE_KEYWORDS[route]):
            continue
        if route == "strain_estimation":
            if knowledge and not compute:
                continue
            if has_files or compute:
                return "strain_estimation"
            continue
        return route  # type: ignore[return-value]
    return None


def route_for_subagent(state: OctGraphState) -> Literal["strain_estimation", "deep_research", "self_rag", "chat"]:
    return state["sub_agent"]


def _research_messages(state: OctGraphState) -> list[BaseMessage]:
    """Keep deep research focused on the current research turn."""

    messages = state.get("messages", [])
    if state.get("research_pending"):
        return messages[-4:]
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", "") == "human":
            return [messages[index]]
    return []


def deep_research_node(state: OctGraphState) -> OctGraphState:
    """Run deep research with unrelated conversation history trimmed away."""

    messages = _research_messages(state)
    child_state: OctGraphState = {
        **state,
        "messages": messages,
        "requested_sub_agent": None,
    }
    result = deep_research.invoke(child_state)
    result_messages = result.get("messages", [])
    new_messages = result_messages[len(messages) :]
    if not new_messages and result.get("final_report"):
        new_messages = [AIMessage(content=result["final_report"])]

    return {
        "messages": new_messages,
        "requested_sub_agent": None,
        "research_pending": result.get("research_pending", False),
        "research_brief": result.get("research_brief", ""),
        "research_topics": result.get("research_topics", []),
        "research_notes": result.get("research_notes", []),
        "final_report": result.get("final_report", ""),
    }


def _memory_command(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("记住"):
        content = re.sub(r"^记住[:：]?", "", stripped).strip()
        item = remember(content)
        return f"已记住：{item.get('content', '')}"
    if stripped.startswith("忘记"):
        content = re.sub(r"^忘记[:：]?", "", stripped).strip()
        removed = forget(content)
        return f"已删除 {removed} 条相关记忆。"
    if "查看记忆" in stripped:
        summary = memory_summary()
        return summary or "当前还没有长期记忆。"
    return None


def chat(state: OctGraphState) -> OctGraphState:
    text = _latest_user_text(state)

    if "清除摘要" in text.strip():
        return {
            "messages": [AIMessage(content="已清除对话摘要，后续对话将重新积累上下文。")],
            "conversation_summary": "",
            "summary_message_count": 0,
        }

    memory_result = _memory_command(text)
    if memory_result is not None:
        return {"messages": [AIMessage(content=memory_result)]}

    summary = memory_summary()
    conversation_summary = state.get("conversation_summary", "")
    sys_msg = SystemMessage(content=build_chat_prompt(summary, conversation_summary))
    messages = state.get("messages", [])
    recent = messages[-_KEEP_RECENT:] if len(messages) > _KEEP_RECENT else messages
    result = get_llm().invoke([sys_msg] + recent)
    return {"messages": [result]}


def _is_memory_command(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("记住", "忘记")) or "查看记忆" in stripped or "清除摘要" in stripped


def self_rag_node(state: OctGraphState) -> OctGraphState:
    # 检索闸门：闲聊 / 与知识库无关的问题在检索前直接走对话，
    # 避免空跑检索+grading，同时保留长期记忆（chat 注入 memory_summary）。
    should_retrieve, gate_trace = decide_retrieval(state.get("messages", []))
    if not should_retrieve:
        fallback = chat(state)
        return {
            **fallback,
            "self_rag_citations": [],
            "self_rag_trace": {"gate": gate_trace},
            "self_rag_error": "",
        }

    question = _latest_user_text(state)
    try:
        result = run_knowledge_query(question)
    except Exception as exc:
        result = {"error": str(exc), "documents": [], "generation": "", "_used_chat_fallback": True}

    if result.get("_used_chat_fallback"):
        fallback = chat(state)
        return {
            **fallback,
            "self_rag_citations": result.get("citations", []),
            "self_rag_trace": {"gate": gate_trace, "retrieval_trace": result.get("retrieval_trace", [])},
            "self_rag_error": str(result.get("error") or ""),
        }

    return {
        "messages": [self_rag_message(result)],
        "self_rag_citations": result.get("citations", []),
        "self_rag_trace": {"gate": gate_trace, "retrieval_trace": result.get("retrieval_trace", [])},
        "self_rag_error": str(result.get("error") or ""),
    }


agent_builder = StateGraph(OctGraphState)
agent_builder.add_node("summarize_conversation", summarize_conversation)
agent_builder.add_node("supervisor", supervisor)
agent_builder.add_node("chat", chat)
agent_builder.add_node("strain_estimation", strain_estimation)
agent_builder.add_node("deep_research", deep_research_node)
agent_builder.add_node("self_rag", self_rag_node)
agent_builder.add_edge(START, "summarize_conversation")
agent_builder.add_edge("summarize_conversation", "supervisor")
agent_builder.add_conditional_edges(
    "supervisor",
    route_for_subagent,
    {
        "strain_estimation": "strain_estimation",
        "deep_research": "deep_research",
        "self_rag": "self_rag",
        "chat": "chat",
    },
)
agent_builder.add_edge("chat", END)
agent_builder.add_edge("strain_estimation", END)
agent_builder.add_edge("deep_research", END)
agent_builder.add_edge("self_rag", END)

graph = agent_builder.compile(name="oct-agent")
