"""Optional anonymous Parallel Search MCP backend for AgentRAG."""

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "AgentRAG/1.0 (+https://github.com/hari12314/RAG-Agent-Tool-Calling)"


async def _search(query, timeout=30):
    # Bound the whole operation, including initialization and transport cleanup.
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout
        ) as http:
            async with streamable_http_client(MCP_URL, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "web_search",
                        {"objective": query, "search_queries": [query]},
                    )

    text = "\n\n".join(
        block.text for block in result.content if block.type == "text"
    )
    if result.isError:
        raise RuntimeError(text or "Parallel search failed.")
    return text or "No results."


def search(query):
    """Return source excerpts and URLs as a ReAct observation (Streamlit thread)."""
    return asyncio.run(_search(query))
