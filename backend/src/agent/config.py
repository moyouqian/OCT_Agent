"""Runtime configuration for the OCT agent."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../..", ".env"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ALLOW_LOCAL_FILE_PATHS = env_bool("ALLOW_LOCAL_FILE_PATHS", False)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto")
MIN_FREE_GPU_MEMORY_GB = float(os.getenv("MIN_FREE_GPU_MEMORY_GB", "2"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    api_key = (
        os.getenv("SILICONFLOW_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("SILICONFLOW_API_BASE")
        or os.getenv("GROQ_API_BASE")
        or os.getenv("OPENAI_API_BASE")
    )
    model = (
        os.getenv("SILICONFLOW_API_MODEL")
        or os.getenv("GROQ_API_MODEL")
        or os.getenv("MODEL")
    )

    missing = [
        name
        for name, value in {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing chat model configuration. Prefer setting "
            "SILICONFLOW_API_KEY, SILICONFLOW_API_BASE, and "
            "SILICONFLOW_API_MODEL. Missing: "
            + ", ".join(missing)
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
