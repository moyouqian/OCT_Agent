"""Research source abstractions.

Tavily is the active v1 source. Zotero MCP is represented as a stub so the
researcher can fan out across sources once that connector is wired in.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.research.schemas import SourceResult


class ResearchSource(ABC):
    """Interface for search-backed research sources."""

    name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        """Return normalized results for a query."""


class TavilyResearchSource(ResearchSource):
    """Tavily web search source."""

    name = "tavily"

    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return [
                SourceResult(
                    source=self.name,
                    title="Tavily 未配置",
                    url="",
                    content="缺少 TAVILY_API_KEY，无法执行联网检索。",
                )
            ]

        try:
            from tavily import TavilyClient
        except ImportError:
            return [
                SourceResult(
                    source=self.name,
                    title="Tavily 依赖未安装",
                    url="",
                    content="缺少 tavily-python 依赖，请先同步后端依赖。",
                )
            ]

        try:
            client = TavilyClient(api_key=api_key)
            payload = client.search(
                query,
                max_results=max_results,
                include_raw_content=True,
                topic="general",
            )
        except Exception as exc:
            return [
                SourceResult(
                    source=self.name,
                    title="Tavily 检索失败",
                    url="",
                    content=f"查询 {query!r} 失败：{exc}",
                )
            ]

        results = []
        for item in payload.get("results", []):
            content = item.get("raw_content") or item.get("content") or ""
            results.append(
                SourceResult(
                    source=self.name,
                    title=item.get("title") or "Untitled",
                    url=item.get("url") or "",
                    content=content[:6000],
                )
            )
        return results


class ZoteroMcpResearchSource(ResearchSource):
    """Placeholder for a future Zotero MCP source."""

    name = "zotero_mcp"

    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        return []


def default_sources() -> list[ResearchSource]:
    """Return enabled research sources."""

    return [TavilyResearchSource(), ZoteroMcpResearchSource()]


def search_all_sources(
    queries: list[str],
    *,
    sources: list[ResearchSource] | None = None,
    max_results: int = 3,
) -> list[SourceResult]:
    """Search all configured sources concurrently and deduplicate results."""

    active_sources = sources if sources is not None else default_sources()
    tasks = [(source, query) for source in active_sources for query in queries]
    if not tasks:
        return []

    collected: list[SourceResult] = []
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        future_map = {
            pool.submit(source.search, query, max_results): (source.name, query)
            for source, query in tasks
        }
        for future in as_completed(future_map):
            source_name, query = future_map[future]
            try:
                collected.extend(future.result())
            except Exception as exc:
                collected.append(
                    SourceResult(
                        source=source_name,
                        title=f"{source_name} 检索失败",
                        url="",
                        content=f"查询 {query!r} 失败：{exc}",
                    )
                )

    deduped: dict[str, SourceResult] = {}
    for item in collected:
        key = item.url or f"{item.source}:{item.title}:{item.content[:80]}"
        if key not in deduped:
            deduped[key] = item
    return list(deduped.values())
