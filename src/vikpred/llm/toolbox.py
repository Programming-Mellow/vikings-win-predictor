"""A synchronous handle on the MCP server, for callers like Streamlit that aren't async.

The MCP client is async and its session must stay on one event loop, so the
toolbox runs that loop in a background thread and keeps one stdio session open
for its whole life instead of spawning the server on every tool call.
"""

import asyncio
import json
import sys
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolbox:
    def __init__(self, command: str = sys.executable, args: list[str] | None = None, env: dict | None = None):
        self._params = StdioServerParameters(
            command=command, args=args or ["-m", "vikpred.mcp_server.server"], env=env)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._run(self._connect())
        self.tools = self._run(self._list_tools())

    def _run(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def _list_tools(self) -> list[dict]:
        result = await self._session.list_tools()
        # Shape each MCP tool as a Claude API tool definition.
        return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in result.tools]

    async def _call(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = await self._session.call_tool(name, arguments)
        if getattr(result, "structured_content", None) is not None:
            text = json.dumps(result.structured_content)
        else:
            text = "\n".join(c.text for c in result.content if c.type == "text")
        return text, bool(result.is_error)

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Run a tool; returns (text result, is_error)."""
        return self._run(self._call(name, arguments))

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._run(self._stack.aclose(), timeout=10)
            except Exception:
                pass  # the server process is gone either way
        self._loop.call_soon_threadsafe(self._loop.stop)
