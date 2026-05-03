"""LangGraph tools and pure computation functions for OCT strain estimation."""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any

import numpy as np
import torch
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from scipy.signal import convolve2d

from agent.services.paths import (
    DEFAULT_BNN_MODEL_PATH,
    DEFAULT_CNN_MODEL_PATH,
    PROJECT_ROOT,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.bnn.bunetPP import UNetPlusPlus
from assets.cnn.Unet import Unet

from agent.services.mat_io import load_single_matrix
from agent.services.models import (
    DEFAULT_BANDWIDTH,
    DEFAULT_REFRACTIVE_INDEX,
    DEFAULT_WAVELENGTH,
    crop_to_divisible_by_32,
    get_inference_device,
    gradient_to_strain,
    load_model,
)
from agent.services.storage import (
    make_result_key,
    resolve_file_reference,
    save_array_result,
    save_bnn_result,
)


def _select_file_id(file_id: str | None, file_ids: list[str] | None) -> str | None:
    if file_id:
        return file_id
    if file_ids and len(file_ids) == 1:
        return file_ids[0]
    return None


def _physical_kwargs(physical_params: dict[str, Any] | None) -> dict[str, float]:
    physical_params = physical_params or {}
    return {
        "wavelength": float(physical_params.get("wavelength") or DEFAULT_WAVELENGTH),
        "bandwidth": float(physical_params.get("bandwidth") or DEFAULT_BANDWIDTH),
        "refractive_index": float(physical_params.get("refractive_index") or DEFAULT_REFRACTIVE_INDEX),
    }


def compute_vector_strain(
    file_path: str,
    Nx: int = 25,
    Nz: int = 25,
    g: int = 1,
    physical_params: dict[str, Any] | None = None,
) -> np.ndarray:
    _, matrix = load_single_matrix(file_path)
    phase_data = np.array(matrix, dtype=float)
    phase_data[phase_data == 0] = np.nan

    kernel_x = np.ones((1, Nx)) / Nx
    complex_phase = np.exp(1j * phase_data)
    b = convolve2d(complex_phase, kernel_x, mode="valid")
    b_model, b_angle = np.abs(b), np.angle(b)

    rows, _ = b_angle.shape
    valid_g = min(g, rows - 1)
    if valid_g < 1:
        raise ValueError("输入数据行数不足，无法执行矢量法应变计算。")

    phase_diff = b_angle[valid_g:, :] - b_angle[:-valid_g, :]
    c = b_model[:-valid_g, :] * b_model[valid_g:, :] * np.exp(1j * phase_diff)
    c_norm_angle = np.angle(c) / valid_g

    kernel_z = np.ones((Nz, 1)) / Nz
    avg_phase = np.angle(convolve2d(np.exp(1j * c_norm_angle), kernel_z, mode="valid"))
    return gradient_to_strain(avg_phase, **_physical_kwargs(physical_params))


def compute_cnn_strain(file_path: str, physical_params: dict[str, Any] | None = None) -> np.ndarray:
    _, wrapped_data = load_single_matrix(file_path)
    device = get_inference_device()
    net = load_model("cnn", Unet, DEFAULT_CNN_MODEL_PATH, device)

    image_np = np.transpose(wrapped_data)
    image = torch.from_numpy(image_np.reshape(1, 1, image_np.shape[0], image_np.shape[1])).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        avg_phase = net(image)

    avg_phase_np = avg_phase.squeeze().cpu().detach().numpy()
    avg_phase_np = np.transpose(np.array(avg_phase_np))
    return gradient_to_strain(avg_phase_np, **_physical_kwargs(physical_params))


def compute_bnn_strain(
    file_path: str,
    MC_test: int = 50,
    physical_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, wrapped_data = load_single_matrix(file_path)
    wrapped_data = crop_to_divisible_by_32(wrapped_data)
    device = get_inference_device()
    net = load_model("bnn", UNetPlusPlus, DEFAULT_BNN_MODEL_PATH, device)

    image_np = np.transpose(wrapped_data)
    image = torch.from_numpy(image_np.reshape(1, 1, image_np.shape[0], image_np.shape[1])).to(
        device=device,
        dtype=torch.float32,
    )

    means = []
    with torch.no_grad():
        for _ in range(MC_test):
            mean, _, _ = net(image)
            means.append(mean)

    stacked_means = torch.stack(means)
    predicts = torch.mean(stacked_means, dim=0).squeeze().cpu().detach().numpy()
    avg_phase = np.transpose(np.array(predicts))
    strain = gradient_to_strain(avg_phase, **_physical_kwargs(physical_params))

    epistemic_uncertainty = torch.var(stacked_means, dim=0).squeeze()
    epistemic_uncertainty = epistemic_uncertainty.cpu().detach().numpy() ** 0.5
    epistemic_uncertainty = np.transpose(np.array(epistemic_uncertainty))
    return strain, epistemic_uncertainty


@tool
def vector_method_g(
    run_dir: Annotated[str, InjectedState("run_dir")],
    file_ids: Annotated[list[str], InjectedState("file_ids")],
    physical_params: Annotated[dict[str, Any], InjectedState("physical_params")],
    file_id: str = "",
    file_path: str = "",
    Nx: int = 25,
    Nz: int = 25,
    g: int = 1,
) -> str:
    """使用矢量法对上传的 .mat 相位文件执行应变计算。"""

    try:
        selected_file_id = _select_file_id(file_id, file_ids)
        resolved_file_id, resolved_path = resolve_file_reference(selected_file_id, file_path or None)
        result_key = make_result_key("vector", Nx=Nx, Nz=Nz, g=g)
        strain = compute_vector_strain(
            resolved_path,
            Nx=Nx,
            Nz=Nz,
            g=g,
            physical_params=physical_params,
        )
        ref = save_array_result(run_dir, resolved_path, result_key, strain, resolved_file_id)
        return json.dumps({"status": "success", "method": "vector", "ref": ref}, ensure_ascii=False)
    except Exception as exc:
        return f"矢量法计算过程中发生错误: {exc}"


@tool
def cnn_method(
    run_dir: Annotated[str, InjectedState("run_dir")],
    file_ids: Annotated[list[str], InjectedState("file_ids")],
    physical_params: Annotated[dict[str, Any], InjectedState("physical_params")],
    file_id: str = "",
    file_path: str = "",
) -> str:
    """使用 CNN 方法对上传的 .mat 相位文件执行应变计算。"""

    try:
        selected_file_id = _select_file_id(file_id, file_ids)
        resolved_file_id, resolved_path = resolve_file_reference(selected_file_id, file_path or None)
        result_key = make_result_key("cnn")
        strain = compute_cnn_strain(resolved_path, physical_params=physical_params)
        ref = save_array_result(run_dir, resolved_path, result_key, strain, resolved_file_id)
        return json.dumps({"status": "success", "method": "cnn", "ref": ref}, ensure_ascii=False)
    except Exception as exc:
        return f"CNN 计算过程中发生错误: {exc}"


@tool
def bnn_method(
    run_dir: Annotated[str, InjectedState("run_dir")],
    file_ids: Annotated[list[str], InjectedState("file_ids")],
    physical_params: Annotated[dict[str, Any], InjectedState("physical_params")],
    file_id: str = "",
    file_path: str = "",
    MC_test: int = 50,
) -> str:
    """使用 BNN 方法对上传的 .mat 相位文件执行应变计算。"""

    try:
        selected_file_id = _select_file_id(file_id, file_ids)
        resolved_file_id, resolved_path = resolve_file_reference(selected_file_id, file_path or None)
        result_key = make_result_key("bnn", MC_test=MC_test)
        strain, epistemic_uncertainty = compute_bnn_strain(
            resolved_path,
            MC_test=MC_test,
            physical_params=physical_params,
        )
        ref = save_bnn_result(
            run_dir,
            resolved_path,
            result_key,
            strain,
            epistemic_uncertainty,
            resolved_file_id,
        )
        return json.dumps({"status": "success", "method": "bnn", "ref": ref}, ensure_ascii=False)
    except Exception as exc:
        return f"BNN 计算过程中发生错误: {exc}"


TOOLS = [vector_method_g, cnn_method, bnn_method]
