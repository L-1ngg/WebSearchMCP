from web_search import planning_adapter
from web_search.planning import PlanningSession


def test_query_fingerprint_preserves_technical_symbol_semantics():
    assert planning_adapter._query_fingerprint("C# async await") != planning_adapter._query_fingerprint("C async await")
    assert planning_adapter._query_fingerprint("C++ coroutine") != planning_adapter._query_fingerprint("C coroutine")
    assert planning_adapter._query_fingerprint("ASP.NET routing") != planning_adapter._query_fingerprint("asp net routing")


def test_query_fingerprint_allows_minimal_formatting_normalization():
    assert planning_adapter._query_fingerprint("gpt - 4.1 pricing") == planning_adapter._query_fingerprint("GPT-4.1   pricing")


def test_missing_required_phases_reports_only_required_missing_steps():
    session = PlanningSession("plan_demo")
    session.complexity_level = 2
    session.phases["intent_analysis"] = object()  # type: ignore[assignment]
    assert planning_adapter._missing_required_phases(session) == [
        "complexity_assessment",
        "query_decomposition",
        "search_strategy",
        "tool_selection",
    ]
