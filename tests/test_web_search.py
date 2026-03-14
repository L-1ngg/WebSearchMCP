import pytest

from web_search import server
from web_search.sources import split_answer_and_sources


@pytest.mark.asyncio
async def test_web_search_returns_explicit_error_when_upstream_fails(monkeypatch):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "false")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    class FailingProvider:
        async def search(self, query, platform=""):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "GrokSearchProvider", lambda api_url, api_key, model: FailingProvider())

    result = await server.web_search("latest status")

    assert result["status"] == "error"
    assert result["answer_ready"] is False
    assert result["error"]["code"] == "upstream_search_failed"
    assert result["error"]["retry_same_query"] is False
    assert "boom" in result["content"]


@pytest.mark.asyncio
async def test_web_search_returns_explicit_error_for_empty_answer(monkeypatch):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "false")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    class EmptyProvider:
        async def search(self, query, platform=""):
            return ""

    monkeypatch.setattr(server, "GrokSearchProvider", lambda api_url, api_key, model: EmptyProvider())

    result = await server.web_search("latest status")

    assert result["status"] == "error"
    assert result["error"]["code"] == "empty_answer"
    assert result["answer_ready"] is False


def test_split_answer_and_sources_keeps_raw_when_split_would_remove_everything():
    raw = """Sources:
- [Example](https://example.com)
- [Docs](https://docs.example.com)
"""

    answer, sources = split_answer_and_sources(raw)

    assert answer == raw.strip()
    assert sources == []
