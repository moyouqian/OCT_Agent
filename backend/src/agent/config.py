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


# LLM provider 优先级。按整组选择，避免 api_key/base_url/model 跨厂商错配：
# 选第一个设置了 api_key 的 provider，其 base_url/model 取自同一组环境变量。
PROVIDERS: tuple[tuple[str, str, str, str], ...] = (
    ("SiliconFlow", "SILICONFLOW_API_KEY", "SILICONFLOW_API_BASE", "SILICONFLOW_API_MODEL"),
    ("Groq", "GROQ_API_KEY", "GROQ_API_BASE", "GROQ_API_MODEL"),
    ("OpenAI", "OPENAI_API_KEY", "OPENAI_API_BASE", "MODEL"),
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回按 provider 优先级选定的 ChatOpenAI 实例。

    结果被 lru_cache 缓存：provider 选择、base_url、model、LLM_TEMPERATURE
    等环境变量只在首次调用时读取并固化到该实例。运行期这是期望行为（复用连接、
    避免重复构造）；但测试或运行中若改了这些环境变量需手动 ``get_llm.cache_clear()``
    才会生效。
    """
    for name, key_env, base_env, model_env in PROVIDERS:
        api_key = os.getenv(key_env)
        if not api_key:
            continue
        base_url = os.getenv(base_env)
        model = os.getenv(model_env)
        missing = [
            env_name
            for env_name, value in ((base_env, base_url), (model_env, model))
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Provider {name} is selected (found {key_env}) but its "
                f"configuration is incomplete. Missing: {', '.join(missing)}"
            )
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise RuntimeError(
        "Missing chat model configuration. Set one provider's API key, base URL, "
        "and model. Prefer SILICONFLOW_API_KEY, SILICONFLOW_API_BASE, and "
        "SILICONFLOW_API_MODEL."
    )
