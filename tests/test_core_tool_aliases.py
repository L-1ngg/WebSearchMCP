import json

import pytest

from web_search import server


@pytest.mark.asyncio
async def test_search_alias_is_registered_with_mcp_defaults():
    tools = await server.mcp.list_tools(run_middleware=False)

    assert "search" in {tool.name for tool in tools}

    tool = await server.mcp.get_tool("search")

    assert tool is not None
    assert tool.name == "search"
    assert "web_search" in tool.description
    assert tool.parameters["required"] == ["query"]
    assert tool.parameters["properties"]["planning_session_id"]["default"] == ""
    assert tool.parameters["properties"]["planning_mode"]["default"] == "auto"
    assert tool.parameters["properties"]["platform"]["default"] == ""
    assert tool.parameters["properties"]["model"]["default"] == ""
    assert tool.parameters["properties"]["search_prompt"]["default"] == ""
    assert tool.parameters["properties"]["source_preference"]["default"] == "auto"
    assert tool.parameters["properties"]["answer_style"]["default"] == "auto"
    assert tool.parameters["properties"]["search_depth"]["default"] == "auto"
    assert tool.parameters["properties"]["extra_sources"]["default"] == 0


@pytest.mark.asyncio
async def test_search_alias_delegates_to_web_search_via_mcp_tool_call(monkeypatch):
    captured: dict[str, object] = {}
    expected = {
        "session_id": "session_123",
        "content": "Final answer",
        "sources_count": 2,
        "status": "ok",
        "answer_ready": True,
        "used_custom_search_prompt": False,
        "planning_applied": False,
        "planning_status": "not_provided",
    }

    async def fake_web_search(
        query: str,
        planning_session_id: str = "",
        planning_mode: str = "auto",
        platform: str = "",
        model: str = "",
        search_prompt: str = "",
        source_preference: str = "auto",
        answer_style: str = "auto",
        search_depth: str = "auto",
        extra_sources: int = 0,
        ctx=None,
    ) -> dict:
        captured.update(
            {
                "query": query,
                "planning_session_id": planning_session_id,
                "planning_mode": planning_mode,
                "platform": platform,
                "model": model,
                "search_prompt": search_prompt,
                "source_preference": source_preference,
                "answer_style": answer_style,
                "search_depth": search_depth,
                "extra_sources": extra_sources,
            }
        )
        return expected

    monkeypatch.setattr(server, "web_search", fake_web_search)

    result = await server.mcp.call_tool(
        "search",
        {"query": "latest status"},
        run_middleware=False,
    )

    assert result.structured_content == expected
    assert json.loads(result.content[0].text) == expected
    assert captured == {
        "query": "latest status",
        "planning_session_id": "",
        "planning_mode": "auto",
        "platform": "",
        "model": "",
        "search_prompt": "",
        "source_preference": "auto",
        "answer_style": "auto",
        "search_depth": "auto",
        "extra_sources": 0,
    }


@pytest.mark.asyncio
async def test_fetch_alias_returns_truncation_metadata(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_web_fetch(url: str, ctx=None) -> str:
        captured["url"] = url
        captured["ctx"] = ctx
        return "0123456789abcdef"

    monkeypatch.setattr(server, "web_fetch", fake_web_fetch)

    result = await server.fetch("https://example.com/docs", max_chars=12)

    assert result == {
        "status": "ok",
        "url": "https://example.com/docs",
        "content": "0123456789ab",
        "truncated": True,
        "content_length": 16,
        "returned_length": 12,
        "max_chars": 12,
    }
    assert captured == {"url": "https://example.com/docs", "ctx": None}


@pytest.mark.asyncio
async def test_fetch_alias_does_not_treat_page_text_prefix_as_error(monkeypatch):
    async def fake_log_info(ctx, message: str, enabled: bool):
        return None

    async def fake_call_tavily_extract(url: str) -> str:
        return "配置错误: 这是页面正文，不是服务错误。"

    async def should_not_call_firecrawl(url: str, ctx=None) -> str | None:
        raise AssertionError("firecrawl should not be used when tavily extract succeeds")

    monkeypatch.setattr(server, "log_info", fake_log_info)
    monkeypatch.setattr(server, "_call_tavily_extract", fake_call_tavily_extract)
    monkeypatch.setattr(server, "_call_firecrawl_scrape", should_not_call_firecrawl)

    result = await server.fetch("https://example.com/error-looking-page")

    assert result["status"] == "ok"
    assert result["url"] == "https://example.com/error-looking-page"
    assert result["content"] == "配置错误: 这是页面正文，不是服务错误。"
    assert result["truncated"] is False
    assert result["content_length"] == len("配置错误: 这是页面正文，不是服务错误。")
    assert result["returned_length"] == len("配置错误: 这是页面正文，不是服务错误。")
    assert result["max_chars"] == 12000


@pytest.mark.asyncio
async def test_fetch_alias_returns_error_status_when_all_extractors_fail(monkeypatch):
    async def fake_log_info(ctx, message: str, enabled: bool):
        return None

    async def fake_call_tavily_extract(url: str) -> str | None:
        return None

    async def fake_call_firecrawl_scrape(url: str, ctx=None) -> str | None:
        return None

    monkeypatch.setattr(server, "log_info", fake_log_info)
    monkeypatch.setattr(server, "_call_tavily_extract", fake_call_tavily_extract)
    monkeypatch.setattr(server, "_call_firecrawl_scrape", fake_call_firecrawl_scrape)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("TAVILY_API_KEYS", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    result = await server.fetch("https://example.com/missing")

    assert result["status"] == "error"
    assert result["url"] == "https://example.com/missing"
    assert result["content"] == "提取失败: 所有提取服务均未能获取内容"
    assert result["truncated"] is False
    assert result["content_length"] == len("提取失败: 所有提取服务均未能获取内容")
    assert result["returned_length"] == len("提取失败: 所有提取服务均未能获取内容")


@pytest.mark.asyncio
async def test_map_alias_returns_structured_payload(monkeypatch):
    captured: dict[str, object] = {}
    expected = {
        "base_url": "https://docs.example.com",
        "results": [{"url": "https://docs.example.com/guide"}],
        "response_time": 0.42,
    }

    async def fake_call_tavily_map_structured(
        url: str,
        instructions: str = "",
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        timeout: int = 150,
    ) -> dict:
        captured.update(
            {
                "url": url,
                "instructions": instructions,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "timeout": timeout,
            }
        )
        return expected

    monkeypatch.setattr(server, "_call_tavily_map_structured", fake_call_tavily_map_structured)

    result = await server.map(
        "https://docs.example.com",
        instructions="docs only",
        max_depth=2,
        max_breadth=5,
        limit=10,
        timeout=30,
    )

    assert result == expected
    assert captured == {
        "url": "https://docs.example.com",
        "instructions": "docs only",
        "max_depth": 2,
        "max_breadth": 5,
        "limit": 10,
        "timeout": 30,
    }


@pytest.mark.asyncio
async def test_web_map_preserves_legacy_string_output_after_helper_split(monkeypatch):
    expected = {
        "base_url": "https://docs.example.com",
        "results": [{"url": "https://docs.example.com/guide"}],
        "response_time": 0.42,
    }

    async def fake_call_tavily_map_structured(
        url: str,
        instructions: str = "",
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        timeout: int = 150,
    ) -> dict:
        return expected

    monkeypatch.setattr(server, "_call_tavily_map_structured", fake_call_tavily_map_structured)

    result = await server.web_map(
        "https://docs.example.com",
        instructions="docs only",
        max_depth=2,
        max_breadth=5,
        limit=10,
        timeout=30,
    )

    assert isinstance(result, str)
    assert json.loads(result) == expected
