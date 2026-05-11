"""Structured-output helpers that avoid function-calling by default."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _extract_result(result: Any) -> tuple[Any | None, Any | None]:
    if isinstance(result, dict):
        return result.get("parsed"), result.get("parsing_error")
    return getattr(result, "parsed", None), getattr(result, "parsing_error", None)


def _invoke_json_mode(model: Any, schema: type[SchemaT], messages: Sequence[BaseMessage]) -> SchemaT:
    structured_model = model.with_structured_output(
        schema,
        method="json_mode",
        include_raw=True,
    )
    result = structured_model.invoke(messages)
    parsed, parsing_error = _extract_result(result)
    if parsed is not None and parsing_error is None:
        return parsed
    raise ValueError(f"json_mode structured output parsing failed: {parsing_error}")


def invoke_structured_json_schema(
    model: Any,
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    fallback_fn: Callable[[Any], SchemaT] | None = None,
) -> SchemaT:
    """Invoke structured output through json_schema first, with safe fallbacks.

    The main path deliberately uses `method="json_schema"` so pure structured
    output does not masquerade as a function/tool call. If json_schema is not
    supported by an OpenAI-compatible provider, we try json_mode once before
    delegating to the caller's fallback.
    """

    try:
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        )
        result = structured_model.invoke(messages)
        parsed, parsing_error = _extract_result(result)

        if parsed is not None and parsing_error is None:
            return parsed

        try:
            return _invoke_json_mode(model, schema, messages)
        except Exception as json_mode_error:
            if fallback_fn is not None:
                return fallback_fn(
                    {
                        "json_schema_result": result,
                        "json_mode_error": json_mode_error,
                    }
                )
            raise ValueError(f"Structured output parsing failed: {parsing_error}") from json_mode_error

    except Exception as exc:
        try:
            return _invoke_json_mode(model, schema, messages)
        except Exception as json_mode_error:
            if fallback_fn is not None:
                return fallback_fn(
                    {
                        "json_schema_error": exc,
                        "json_mode_error": json_mode_error,
                    }
                )
            raise
