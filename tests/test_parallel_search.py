"""Exercise the real MCP transport and app agent loop without loading FAISS/models."""

import ast
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

import parallel_search


def app_functions():
    # app.py starts the Streamlit UI at import time. Compile its unchanged function
    # definitions so dispatch tests need neither UI state nor embedding downloads.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
    names = {
        "t_kb", "t_web", "t_dt", "t_calc", "t_sum", "t_check", "execute",
        "parse_action", "build_system_prompt", "run_agent",
    }
    functions = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )
    namespace = dict(re=re, math=math, time=time, datetime=datetime)
    exec(compile(functions, "app.py", "exec"), namespace)
    return namespace


class MCPFixture:
    def __init__(self, text="LangChain docs: https://docs.langchain.com/ RAG uses retrieval.",
                 error=False, delay=False, status=200):
        self.text, self.error, self.delay, self.status = text, error, delay, status
        self.requests, self.clients = [], []
        self.called = asyncio.Event()
        self.real_client = httpx.AsyncClient

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append((request, body))
        method = body["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "web_search", "inputSchema": {
                    "type": "object", "required": ["objective", "search_queries"],
                    "properties": {"objective": {"type": "string"},
                                   "search_queries": {"type": "array", "items": {"type": "string"}}},
                },
            }]}
        elif method == "tools/call":
            self.called.set()
            if self.delay:
                await asyncio.Event().wait()
            if self.status != 200:
                return httpx.Response(self.status, text="Service unavailable")
            result = {"content": [{"type": "text", "text": self.text}], "isError": self.error}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def client(self, **kwargs):
        client = self.real_client(transport=httpx.MockTransport(self.respond), **kwargs)
        self.clients.append(client)
        return client

    def assert_requests(self, case, query):
        case.assertTrue(self.requests)
        for request, _ in self.requests:
            case.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            case.assertEqual(request.headers["User-Agent"], parallel_search.USER_AGENT)
            case.assertNotIn("Authorization", request.headers)
        calls = [b for _, b in self.requests if b["method"] == "tools/call"]
        case.assertEqual(len(calls), 1)
        case.assertEqual(calls[0]["params"]["name"], "web_search")
        case.assertEqual(calls[0]["params"]["arguments"],
                         {"objective": query, "search_queries": [query]})
        case.assertTrue(all(c.is_closed for c in self.clients))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.app = app_functions()

    def test_agent_consumes_parallel_observation(self):
        fixture = MCPFixture()
        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                text = 'Thought: Find documentation.\nAction: search_web("LangChain RAG")'
            else:
                observation = kwargs["messages"][-1]["content"]
                self.assertIn(fixture.text, observation)
                text = "Thought: Use the source.\nFinal Answer: RAG uses retrieval: https://docs.langchain.com/"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        # Even an unrelated saved API key must not be used on the anonymous path.
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "unused-test-key"}), \
                patch("parallel_search.httpx.AsyncClient", fixture.client):
            result = self.app["run_agent"](client, "fixture-model", None, "Find RAG docs",
                                           search_provider="Parallel")
        self.assertIn("https://docs.langchain.com/", result["answer"])
        self.assertEqual(result["tools_used"], ["search_web"])
        self.assertEqual(result["steps"][0]["observation"], fixture.text)
        fixture.assert_requests(self, "LangChain RAG")

    def test_duckduckgo_default_preserves_output_and_result_count(self):
        class DDGS:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def text(inner, query, max_results):
                self.assertEqual(query, "RAG")
                self.assertEqual(max_results, 3)
                return [{"title": "Default", "body": "Existing output"}]

        with patch.dict(sys.modules, {"duckduckgo_search": SimpleNamespace(DDGS=DDGS)}), \
                patch("parallel_search.search", side_effect=AssertionError("Unexpected MCP call")):
            self.assertEqual(self.app["execute"](None, None, None, "search_web", "RAG"),
                             "Default: Existing output")

    def test_tool_error_is_observation_without_fallback(self):
        fixture = MCPFixture(text="Free search quota exceeded", error=True)
        with patch("parallel_search.httpx.AsyncClient", fixture.client):
            result = self.app["execute"](None, None, None, "search_web", "RAG", "Parallel")
        self.assertEqual(result, "Web search error: Free search quota exceeded")
        fixture.assert_requests(self, "RAG")

    def test_transport_error_is_observation(self):
        fixture = MCPFixture(status=503)
        with patch("parallel_search.httpx.AsyncClient", fixture.client):
            result = self.app["t_web"]("RAG", search_provider="Parallel")
        self.assertTrue(result.startswith("Web search error:"), result)
        fixture.assert_requests(self, "RAG")

    def test_empty_result(self):
        fixture = MCPFixture(text="")
        with patch("parallel_search.httpx.AsyncClient", fixture.client):
            self.assertEqual(parallel_search.search("RAG"), "No results.")

    def test_missing_optional_dependencies(self):
        with patch.dict(sys.modules, {"parallel_search": None}):
            self.assertEqual(self.app["t_web"]("RAG", search_provider="Parallel"),
                             "Parallel search requires: pip install -r requirements-parallel.txt")

    def test_timeout_observation(self):
        with patch("parallel_search.search", side_effect=TimeoutError):
            self.assertEqual(self.app["t_web"]("RAG", search_provider="Parallel"),
                             "Web search error: Parallel search timed out.")

    def test_ui_entry_points_forward_selection(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "run_agent"]
        self.assertEqual(len(calls), 2)  # Send and example buttons.
        for call in calls:
            value = next(k.value for k in call.keywords if k.arg == "search_provider")
            self.assertIsInstance(value, ast.Name)
            self.assertEqual(value.id, "search_provider")


class AsyncSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_total_timeout_closes_client(self):
        fixture = MCPFixture(delay=True)
        with patch("parallel_search.httpx.AsyncClient", fixture.client):
            with self.assertRaises(TimeoutError):
                await parallel_search._search("RAG", timeout=0.1)
        fixture.assert_requests(self, "RAG")

    async def test_cancellation_propagates_and_closes_client(self):
        fixture = MCPFixture(delay=True)
        with patch("parallel_search.httpx.AsyncClient", fixture.client):
            task = asyncio.create_task(parallel_search._search("RAG"))
            await asyncio.wait_for(fixture.called.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        fixture.assert_requests(self, "RAG")


if __name__ == "__main__":
    unittest.main()
