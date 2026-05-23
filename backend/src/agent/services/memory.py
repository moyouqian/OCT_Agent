"""Lightweight local memory for the OCT agent."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from agent.services.paths import DATA_ROOT, ensure_data_dirs

MEMORY_PATH = DATA_ROOT / "memory.json"
_MEMORY_LOCK = threading.Lock()
MEMORY_CATEGORIES = {
    "preference": "用户偏好",
    "project": "项目事实",
    "physical": "常用物理参数",
    "file": "文件/实验摘要",
}


def _empty_memory() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in MEMORY_CATEGORIES}


def load_memory() -> dict[str, list[dict[str, Any]]]:
    ensure_data_dirs()
    if not MEMORY_PATH.exists():
        return _empty_memory()
    data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    memory = _empty_memory()
    for key, value in data.items():
        if key in memory and isinstance(value, list):
            memory[key] = value
    return memory


def save_memory(memory: dict[str, list[dict[str, Any]]]) -> None:
    ensure_data_dirs()
    MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def remember(content: str, category: str = "preference") -> dict[str, Any]:
    category = category if category in MEMORY_CATEGORIES else "preference"
    item = {
        "content": content.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if item["content"]:
        with _MEMORY_LOCK:
            memory = load_memory()
            memory[category].append(item)
            save_memory(memory)
    return item


def forget(content: str) -> int:
    with _MEMORY_LOCK:
        memory = load_memory()
        removed = 0
        for key, items in memory.items():
            kept = [item for item in items if content not in item.get("content", "")]
            removed += len(items) - len(kept)
            memory[key] = kept
        save_memory(memory)
    return removed


def memory_summary(limit: int = 8) -> str:
    memory = load_memory()
    lines: list[str] = []
    for key, label in MEMORY_CATEGORIES.items():
        for item in memory.get(key, [])[-limit:]:
            content = item.get("content", "").strip()
            if content:
                lines.append(f"- {label}: {content}")
    return "\n".join(lines[-limit:])
