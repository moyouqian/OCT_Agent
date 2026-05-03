"""OCT Agent 的 LangGraph 编排。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import get_llm
from agent.schemas import OctGraphState, TaskAssignment
from agent.services.memory import forget, memory_summary, remember
from agent.services.models import DEFAULT_BANDWIDTH, DEFAULT_REFRACTIVE_INDEX, DEFAULT_WAVELENGTH
from agent.services.storage import make_run_dir
from agent.tools import TOOLS


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
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


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
        content=(
            "你是 OCT Agent 中负责应变计算的子助手。你拥有三种工具：\n"
            "1. vector_method_g：矢量法，可带参数 Nx、Nz、g。\n"
            "2. cnn_method：CNN / Unet 深度学习方法。\n"
            "3. bnn_method：BNN / 贝叶斯神经网络方法，可带参数 MC_test，输出 strain 和 epistemic_uncertainty。\n\n"
            "规则：\n"
            "- 前端上传文件后会提供 file_id；调用工具时优先使用 file_id，不要编造本地路径。\n"
            "- 如果只有一个 file_id，可以直接使用它；如果有多个文件，按用户文字选择对应 file_id。\n"
            "- 如果高级设置中选择了应变计算方法，优先按这些方法执行。\n"
            "- 如果没有选择方法，但用户明确要求某个方法，则按用户文字执行。\n"
            "- 对同一个文件、同一种方法、同一组参数，不要重复调用。\n"
            "- 工具完成后，用中文简要说明结果已生成，不要把大矩阵写进消息。\n\n"
            f"当前 run_dir：{run_dir}\n"
            f"当前 file_ids：{file_ids}\n"
            f"高级设置选择的方法：{selected_methods or '未选择'}\n"
            f"物理参数：波长={physical['wavelength']}，带宽={physical['bandwidth']}，折射率={physical['refractive_index']}\n"
            f"当前已完成结果：{summary}"
        )
    )
    response = get_llm().bind_tools(TOOLS).invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def collect_result_refs(state: OctGraphState) -> OctGraphState:
    refs = []
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") != "tool":
            continue
        try:
            import json

            data = json.loads(msg.content)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("status") == "success" and "ref" in data:
            refs.append(data["ref"])

    refs = list(reversed(refs))
    old_ids = {ref.get("result_id") for ref in state.get("result_refs", [])}
    new_refs = [ref for ref in refs if ref.get("result_id") not in old_ids]
    if not new_refs:
        return {}
    return {"result_refs": new_refs}


def judge_visualize(state: OctGraphState) -> OctGraphState:
    return {}


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


def route_after_assistant(state: OctGraphState) -> Literal["tools", "judge_visualize"]:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "judge_visualize"


def route_after_judge(state: OctGraphState) -> str:
    return "visualize" if state.get("visualization_enabled", True) else END


strain_builder = StateGraph(OctGraphState)
strain_builder.add_node("init_run_context", init_run_context)
strain_builder.add_node("strain_assistant", strain_assistant)
strain_builder.add_node("tools", ToolNode(TOOLS))
strain_builder.add_node("collect_result_refs", collect_result_refs)
strain_builder.add_node("judge_visualize", judge_visualize)
strain_builder.add_node("visualize", visualize_node)
strain_builder.add_edge(START, "init_run_context")
strain_builder.add_edge("init_run_context", "strain_assistant")
strain_builder.add_conditional_edges(
    "strain_assistant",
    route_after_assistant,
    {"tools": "tools", "judge_visualize": "judge_visualize"},
)
strain_builder.add_edge("tools", "collect_result_refs")
strain_builder.add_edge("collect_result_refs", "strain_assistant")
strain_builder.add_conditional_edges(
    "judge_visualize",
    route_after_judge,
    {"visualize": "visualize", END: END},
)
strain_builder.add_edge("visualize", END)
strain_estimation = strain_builder.compile(name="strain-estimation")


def supervisor(state: OctGraphState) -> OctGraphState:
    settings = state.get("strain_settings", {})
    selected_method = any(settings.get(key) for key in ("vector", "cnn", "bnn"))
    text = _latest_user_text(state)
    if selected_method and state.get("file_ids"):
        return {"sub_agent": "strain_estimation"}

    sys_msg = SystemMessage(
        content=(
            "你是 OCT Agent 的任务分发助手。当前已支持两个方向：\n"
            "1. strain_estimation：OCT 应变计算，包含矢量法、CNN、BNN、.mat 文件、phase 数据、热力图。\n"
            "2. chat：普通聊天、解释概念、问答、记忆管理，后续可扩展到文献检索与总结。\n\n"
            "如果用户明确要求应变计算、处理 .mat/phase 数据、使用 CNN/BNN/矢量法或查看热力图，选择 strain_estimation；"
            "其他情况选择 chat。只能调用 TaskAssignment。"
        )
    )
    response = get_llm().bind_tools([TaskAssignment]).invoke([sys_msg] + state["messages"])
    if not response.tool_calls:
        update_type = "strain_estimation" if any(
            token in text.lower()
            for token in ["应变", "strain", "cnn", "bnn", "矢量", "vector", ".mat", "phase", "热力图"]
        ) else "chat"
    else:
        update_type = response.tool_calls[0]["args"]["update_type"]
    return {"sub_agent": update_type}


def route_for_subagent(state: OctGraphState) -> Literal["strain_estimation", "chat"]:
    return state["sub_agent"]


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
    memory_result = _memory_command(text)
    if memory_result is not None:
        return {"messages": [AIMessage(content=memory_result)]}

    summary = memory_summary()
    memory_text = f"\n\n可参考的长期记忆：\n{summary}" if summary else ""
    sys_msg = SystemMessage(
        content=(
            "你是一个通用 OCT Agent，可以进行自然语言问答、解释 OCT 相关概念，"
            "未来还会扩展文献检索、文献总结和报告生成。请用中文回答。"
            "如果模型输出包含 <think>，前端会按用户设置决定是否展示。"
            f"{memory_text}"
        )
    )
    result = get_llm().invoke([sys_msg] + state["messages"])
    return {"messages": [result]}


agent_builder = StateGraph(OctGraphState)
agent_builder.add_node("supervisor", supervisor)
agent_builder.add_node("chat", chat)
agent_builder.add_node("strain_estimation", strain_estimation)
agent_builder.add_edge(START, "supervisor")
agent_builder.add_conditional_edges(
    "supervisor",
    route_for_subagent,
    {"strain_estimation": "strain_estimation", "chat": "chat"},
)
agent_builder.add_edge("chat", END)
agent_builder.add_edge("strain_estimation", END)

graph = agent_builder.compile(name="oct-agent")
