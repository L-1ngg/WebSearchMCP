import pytest

from web_search import server
from web_search.sources import build_get_sources_response


def _build_sources(count: int) -> list[dict]:
    return [
        {
            "title": f"Source {index}",
            "url": f"https://example.com/{index}",
            "provider": "test",
        }
        for index in range(count)
    ]


def test_build_get_sources_response_adds_metadata_and_optional_error():
    response = build_get_sources_response(
        "demo-session",
        {
            "sources": [{"url": "https://example.com"}],
            "sources_count": 1,
            "next_cursor": "",
            "has_more": False,
        },
        error="session_id_not_found_or_expired",
    )

    assert response == {
        "session_id": "demo-session",
        "sources": [{"url": "https://example.com"}],
        "sources_count": 1,
        "returned_count": 1,
        "next_cursor": "",
        "has_more": False,
        "error": "session_id_not_found_or_expired",
    }


@pytest.mark.asyncio
async def test_get_sources_supports_cached_source_pagination():
    session_id = "pagination-session"
    await server._SOURCES_CACHE.set(session_id, _build_sources(4))

    first_page = await server.get_sources(session_id, limit=2)
    second_page = await server.get_sources(session_id, limit=2, cursor="2")

    assert first_page == {
        "session_id": session_id,
        "sources": _build_sources(2),
        "sources_count": 4,
        "returned_count": 2,
        "next_cursor": "2",
        "has_more": True,
    }
    assert second_page == {
        "session_id": session_id,
        "sources": _build_sources(4)[2:],
        "sources_count": 4,
        "returned_count": 2,
        "next_cursor": "",
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_get_sources_keeps_legacy_full_list_behavior_when_limit_is_omitted_or_zero():
    session_id = "legacy-session"
    sources = _build_sources(3)
    await server._SOURCES_CACHE.set(session_id, sources)

    omitted_limit = await server.get_sources(session_id)
    zero_limit = await server.get_sources(session_id, limit=0)

    expected = {
        "session_id": session_id,
        "sources": sources,
        "sources_count": 3,
        "returned_count": 3,
        "next_cursor": "",
        "has_more": False,
    }

    assert omitted_limit == expected
    assert zero_limit == expected


@pytest.mark.asyncio
async def test_get_sources_keeps_missing_session_structured():
    result = await server.get_sources("missing-session", limit=2, cursor="2")

    assert result == {
        "session_id": "missing-session",
        "sources": [],
        "sources_count": 0,
        "returned_count": 0,
        "next_cursor": "",
        "has_more": False,
        "error": "session_id_not_found_or_expired",
    }
