"""Model loading and numerical helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.config import INFERENCE_DEVICE, MIN_FREE_GPU_MEMORY_GB

_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}

DEFAULT_WAVELENGTH = 840e-9
DEFAULT_BANDWIDTH = 50e-9
DEFAULT_REFRACTIVE_INDEX = 1.0


def get_inference_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if INFERENCE_DEVICE != "auto":
        return torch.device(INFERENCE_DEVICE)

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx

    if best_free < MIN_FREE_GPU_MEMORY_GB * 1024**3:
        return torch.device("cpu")
    return torch.device(f"cuda:{best_idx}")


def crop_to_divisible_by_32(un_phase: np.ndarray) -> np.ndarray:
    target_rows = un_phase.shape[0] - (un_phase.shape[0] % 32)
    target_cols = un_phase.shape[1] - (un_phase.shape[1] % 32)
    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("BNN 推理要求输入矩阵至少为 32x32。")

    rows_to_crop = un_phase.shape[0] - target_rows
    cols_to_crop = un_phase.shape[1] - target_cols
    top_crop = rows_to_crop // 2
    bottom_crop = rows_to_crop - top_crop
    left_crop = cols_to_crop // 2
    right_crop = cols_to_crop - left_crop

    cropped = un_phase[top_crop:] if bottom_crop == 0 else un_phase[top_crop:-bottom_crop]
    cropped = cropped[:, left_crop:] if right_crop == 0 else cropped[:, left_crop:-right_crop]
    return cropped


def load_model(
    model_name: str,
    model_class: type[torch.nn.Module],
    model_path: str | Path,
    device: torch.device,
    *args: Any,
    **kwargs: Any,
) -> torch.nn.Module:
    cache_key = (model_name, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    net = model_class(*args, **kwargs)
    state = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    _MODEL_CACHE[cache_key] = net
    return net


def gradient_to_strain(
    avg_phase: np.ndarray,
    wavelength: float = DEFAULT_WAVELENGTH,
    bandwidth: float = DEFAULT_BANDWIDTH,
    refractive_index: float = DEFAULT_REFRACTIVE_INDEX,
) -> np.ndarray:
    p1 = (wavelength**2) / bandwidth
    displacement = wavelength * avg_phase / (4 * np.pi * refractive_index)
    strain = displacement * 1e3 / p1
    return np.nan_to_num(strain, nan=0.0)
