"""Shared state and response schemas."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResultRef(TypedDict, total=False):
    result_id: str
    file_id: str
    file_path: str
    result_key: str
    result_path: str
    kind: Literal["array", "bnn"]
    format: Literal["mat"]
    shape: list[int]
    outputs: dict[str, dict[str, Any]]


class StrainSettings(TypedDict, total=False):
    vector: bool
    cnn: bool
    bnn: bool
    Nx: int
    Nz: int
    g: int
    MC_test: int


class PhysicalParams(TypedDict, total=False):
    wavelength: float
    bandwidth: float
    refractive_index: float


class OctGraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    file_ids: list[str]
    run_dir: str
    result_refs: Annotated[list[ResultRef], operator.add]
    sub_agent: Literal["strain_estimation", "deep_research", "chat"]
    requested_sub_agent: Literal["deep_research"] | None
    strain_settings: StrainSettings
    physical_params: PhysicalParams
    visualization_enabled: bool
    show_thinking: bool
    research_pending: bool
    research_brief: str
    research_topics: list[str]
    research_notes: list[str]
    final_report: str


class TaskAssignment(TypedDict):
    """Decision on what task type to execute."""

    update_type: Literal["strain_estimation", "deep_research", "chat"]
