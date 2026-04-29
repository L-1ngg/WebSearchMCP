from web_search import retrieval_service, server


def test_fetch_status_contextvar_defaults_to_unknown():
    assert retrieval_service._FETCH_STATUS.get() == "unknown"


def test_truncate_content_reports_lengths_and_truncation():
    result = retrieval_service._truncate_content("abcdefghij", 4)
    assert result == {
        "content": "abcd",
        "truncated": True,
        "content_length": 10,
        "returned_length": 4,
        "max_chars": 4,
    }


def test_build_tavily_map_payload_preserves_expected_keys():
    payload = retrieval_service._build_tavily_map_payload(
        {
            "base_url": "https://docs.example.com",
            "results": [{"url": "https://docs.example.com/intro"}],
            "response_time": 0.42,
        }
    )
    assert payload == {
        "base_url": "https://docs.example.com",
        "results": [{"url": "https://docs.example.com/intro"}],
        "response_time": 0.42,
    }


def test_server_reexports_tavily_map_helper_from_retrieval_service():
    assert server._call_tavily_map is retrieval_service._call_tavily_map
