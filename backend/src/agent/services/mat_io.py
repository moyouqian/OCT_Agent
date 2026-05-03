"""MATLAB `.mat` file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


def get_public_variable_names(mat_data: dict[str, Any]) -> list[str]:
    return [name for name in mat_data if not name.startswith("__") and not name.startswith("_")]


def load_single_matrix(path: str | Path) -> tuple[str, np.ndarray]:
    mat_data = sio.loadmat(path)
    var_names = get_public_variable_names(mat_data)
    if not var_names:
        raise ValueError(f"No usable variables found in {path}.")
    if len(var_names) != 1:
        raise ValueError(
            f"Expected exactly one usable variable in {path}, found {len(var_names)}: {var_names}."
        )
    name = var_names[0]
    value = np.asarray(mat_data[name])
    if value.ndim != 2:
        raise ValueError(f"Variable {name} must be a 2D matrix, got shape {value.shape}.")
    return name, value


def inspect_mat_file(path: str | Path) -> dict[str, Any]:
    mat_data = sio.loadmat(path)
    variables = []
    for name in get_public_variable_names(mat_data):
        value = np.asarray(mat_data[name])
        variables.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    return {"variables": variables}

