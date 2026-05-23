"""Message utility helpers shared across agent modules."""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the text content of the most recent human message, or empty string."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""
