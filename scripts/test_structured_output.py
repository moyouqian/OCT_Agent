"""Smoke test SiliconFlow structured-output methods.

Run from the repository root:
    python scripts/test_structured_output.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class RouteDecision(BaseModel):
    decision: Literal["clarify", "write_brief"]
    reason: str


def _raw_text(raw: object) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    return repr(content)


def main() -> int:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_API_BASE") or os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("SILICONFLOW_API_MODEL")

    missing = [
        name
        for name, value in {
            "SILICONFLOW_API_KEY": api_key,
            "SILICONFLOW_API_BASE or SILICONFLOW_BASE_URL": base_url,
            "SILICONFLOW_API_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        print("Missing environment variables: " + ", ".join(missing))
        return 2

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    messages = [
        HumanMessage(
            content=(
                "Return JSON for this routing task. "
                "If the user asks only 'OCT', choose clarify. "
                "Schema fields: decision, reason."
            )
        )
    ]

    for method in ["json_schema", "json_mode", "function_calling"]:
        print(f"\n=== method={method} ===")
        try:
            structured = llm.with_structured_output(
                RouteDecision,
                method=method,
                include_raw=True,
            )
            result = structured.invoke(messages)
            print("parsed:", result.get("parsed"))
            print("parsing_error:", result.get("parsing_error"))
            print("raw:", _raw_text(result.get("raw")))
        except Exception as exc:
            print("error:", repr(exc))

    return 0


if __name__ == "__main__":
    sys.exit(main())
