from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import MCPServer
from mcp.types import TextContent

from mcp_codesearch import __version__
from mcp_codesearch.progress import ToolProgressMiddleware
from mcp_codesearch.server import EXPECTED_TOOLS, mcp

ROOT = Path(__file__).resolve().parents[1]


async def assert_safe_tool_dispatch(client: Client) -> None:
    result = await client.call_tool("code_search", {"query": ""})

    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "Query cannot be empty" in result.content[0].text


def test_server_uses_public_sdk_v2_metadata() -> None:
    assert isinstance(mcp, MCPServer)
    assert mcp.name == "codesearch"
    assert mcp.version == __version__


def test_progress_middleware_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        ToolProgressMiddleware(interval_seconds=0)


@pytest.mark.asyncio
async def test_long_tool_emits_repeated_progress_until_completion() -> None:
    server = MCPServer(
        "progress-test",
        middleware=[ToolProgressMiddleware(interval_seconds=0.01)],
    )

    @server.tool()
    async def slow_tool() -> str:
        await asyncio.sleep(0.08)
        return "complete"

    progress: list[tuple[float, float | None, str | None]] = []

    async def capture_progress(
        current: float,
        total: float | None,
        message: str | None,
    ) -> None:
        progress.append((current, total, message))

    async with Client(server, cache=None) as client:
        result = await client.call_tool(
            "slow_tool",
            progress_callback=capture_progress,
        )

    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "complete"
    assert len(progress) >= 3
    assert [current for current, _total, _message in progress] == sorted(
        current for current, _total, _message in progress
    )
    assert all(total is None for _current, total, _message in progress)
    assert all(message == "slow_tool is still running" for _current, _total, message in progress)


@pytest.mark.asyncio
async def test_server_supports_modern_and_legacy_protocols() -> None:
    async with Client(mcp, cache=None) as modern:
        assert modern.protocol_version == "2026-07-28"
        assert modern.server_info is not None
        assert modern.server_info.name == "codesearch"
        assert modern.server_info.version == __version__

        first = await modern.list_tools()
        second = await modern.list_tools()
        tool_names = [tool.name for tool in first.tools]

        assert tool_names == [tool.name for tool in second.tools] == EXPECTED_TOOLS
        assert first.result_type == "complete"
        assert first.ttl_ms == 0
        assert first.cache_scope == "private"
        assert first.meta == {
            "io.modelcontextprotocol/serverInfo": {
                "name": "codesearch",
                "version": __version__,
            }
        }
        await assert_safe_tool_dispatch(modern)

    async with Client(mcp, mode="legacy", cache=None) as legacy:
        assert legacy.protocol_version == "2025-11-25"
        assert legacy.server_info is not None
        assert legacy.server_info.name == "codesearch"
        assert legacy.server_info.version == __version__
        assert [tool.name for tool in (await legacy.list_tools()).tools] == EXPECTED_TOOLS
        await assert_safe_tool_dispatch(legacy)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_protocol"),
    [("auto", "2026-07-28"), ("legacy", "2025-11-25")],
)
async def test_stdio_entrypoint_supports_modern_and_legacy_wire_protocols(
    mode: str,
    expected_protocol: str,
) -> None:
    params = StdioServerParameters(
        command=str(Path(sys.executable).with_name("mcp-codesearch")),
        cwd=ROOT,
    )

    async with Client(stdio_client(params), mode=mode, cache=None) as client:
        assert client.protocol_version == expected_protocol
        assert client.server_info is not None
        assert client.server_info.name == "codesearch"
        assert client.server_info.version == __version__
        assert [tool.name for tool in (await client.list_tools()).tools] == EXPECTED_TOOLS
        await assert_safe_tool_dispatch(client)
